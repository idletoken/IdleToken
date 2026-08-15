// Cluster pairing (acceptance P3). Interface-first (philosophy 14): the UI talks
// to a PairingProvider. Two entry modes — same account, or a one-time code —
// both funnel into one LAN cluster. The real provider talks to the coordinator
// over the local RPC and reflects the actual engine cluster (members + stage
// topology); in a plain browser (no Tauri, so no LAN access) a clearly-labeled
// dev-sim drives the UI so the flow can be built and reviewed.
//
// Anything from the sim is marked `source: "dev-sim"` and the UI badges it, so
// simulated peers can never be mistaken for a real cluster (philosophy 9/15).

import type { EngineTuning } from "./settings";
import { forgetSimLoadedModel } from "./clusterStats";

export type NodeRole = "coordinator" | "worker";

// Per-node lifecycle during auto-orchestration (P4): once a cluster forms, every
// node probes, gets a layer range assigned, loads it, then is ready.
export type NodeStage = "joined" | "probing" | "assigned" | "loading" | "ready" | "error";

export interface PeerNode {
  id: string;
  hostname: string;
  gpu: string;
  role: NodeRole;
  self: boolean;
  stage: NodeStage;
  layerLo?: number; // assigned pipeline layer range [lo, hi)
  layerHi?: number;
  /** false = the creator has not heard this member's roster poll within its
   *  timeout (audit 2.8). Optional so absence (older snapshot shapes) reads as
   *  online — which is what absence used to mean. */
  online?: boolean;
}

export interface SelfInfo {
  hostname: string;
  gpu: string;
  // This machine's local GGUF, as resolved by weights.resolveLocalWeights.
  // Empty = nothing local, which is correct for a joiner (the coordinator's
  // shard service feeds its layers) and a mock load in P3. Non-empty = real
  // weights the engines should load from disk (P4/P6).
  modelPath?: string;
  // Settings-derived engine tuning (API bind/token, inter-stage port,
  // discovery port). Omitted = the Rust side's defaults (the historical
  // hard-coded ports). See settings.engineTuning().
  tuning?: EngineTuning;
}

// Cluster-wide orchestration phase (P4) and the exposed API (P6).
export type OrchestrationPhase = "idle" | "probing" | "splitting" | "loading" | "ready";
export type ApiStatus = "offline" | "starting" | "online";

export interface ClusterApi {
  baseUrl: string;
  status: ApiStatus;
}

/** Why the last join attempt failed (pairing.rs `last_error`). `code` maps to
 *  a localized sentence in the UI (pairing.err.*); `detail` carries the
 *  variable part — the discovery port for "notFound", the creator's verbatim
 *  rejection for "rejected". */
export interface PairingError {
  code: string;
  detail: string;
}

export interface PairingSnapshot {
  code: string | null; // the cluster's join code (held by whoever created it)
  // True when this cluster was formed in account mode (integration plan 3.3):
  // the join proof is a secret derived from the platform account, not a
  // human-shareable code — `code` stays null and the UI labels it accordingly.
  accountMode?: boolean;
  peers: PeerNode[];
  coordinatorId: string | null;
  phase: OrchestrationPhase;
  api: ClusterApi | null;
  source: "engine" | "dev-sim";
  // True on the creator while the roster is still open and big enough to
  // launch — the UI shows the "start cluster" button. (Real provider only;
  // the dev-sim auto-orchestrates.)
  canStart?: boolean;
  // Why the last join attempt failed; null/absent while nothing has. (Real
  // provider only — the dev-sim never fails.)
  lastError?: PairingError | null;
}

export const DS4_TOTAL_LAYERS = 43;

// Contiguous PP split of `total` layers across `n` nodes: even base with the
// remainder going to the earlier (stronger) stages. Mirrors the coordinator's
// v0.1 plan_layers shape; the real split is resource-proportional on-engine.
export function splitLayers(n: number, total = DS4_TOTAL_LAYERS): Array<[number, number]> {
  if (n <= 0) return [];
  const base = Math.floor(total / n);
  const rem = total % n;
  const ranges: Array<[number, number]> = [];
  let lo = 0;
  for (let i = 0; i < n; i++) {
    const hi = lo + base + (i < rem ? 1 : 0);
    ranges.push([lo, hi]);
    lo = hi;
  }
  return ranges;
}

export interface PairingProvider {
  // Start a cluster here. `code` reuses an existing join code instead of
  // minting a fresh one — the model-switch restart (App.switchModel) needs it:
  // switching means tearing the cluster down and building it again, and a new
  // code would strand every other machine, which is holding the old one.
  create(self: SelfInfo, code?: string): Promise<void>;
  join(code: string, self: SelfInfo): Promise<void>; // join an existing cluster
  // Account mode (integration plan 3.3): same LAN mechanics as create/join, but
  // the join proof is `secret` — derived from the signed-in platform account
  // via accountPairSecret() — so no code is typed or displayed. LAN-only:
  // machines must hear each other's UDP beacon (no cloud rendezvous here; the
  // engine's --pair-account/rendezvous path is a later convergence).
  createAccount(self: SelfInfo, secret: string): Promise<void>;
  joinAccount(self: SelfInfo, secret: string): Promise<void>;
  // creator: freeze the roster, launch the engines (P4 entry).
  // allowSolo=true is the single-machine flow saying it really does mean one
  // machine; without it the engine enforces a 2-machine floor.
  start(allowSolo?: boolean): Promise<void>;
  leave(): Promise<void>;
  setCoordinator(peerId: string): Promise<void>;
  subscribe(cb: (s: PairingSnapshot) => void): () => void;
}

export function isValidCode(code: string): boolean {
  return /^[A-Z0-9]{6}$/.test(code.trim().toUpperCase());
}

// Account-mode pair secret (integration plan 3.3): a deterministic secret every
// machine signed in to the same platform account derives locally, so they can
// find each other over the existing beacon/roster mechanics without a typed
// code. Material = platform user id (stable, account-scoped — deliberately NOT
// the raw email, and NOT the JWT which differs per login) + the normalized
// platform URL (accounts from different platforms never collide) + the cluster
// name (lets one account run separate clusters side by side; machines must
// share the setting — default "home"). The secret itself is never broadcast:
// the UDP beacon carries only its FNV-1a hash, and the full value travels only
// inside the LAN TCP join as proof — the same trust level as a shared code.
// Honesty: this proves "derived from the same account material", which matches
// the code-mode trust bar; it is not a platform-verified JWT handshake (that is
// the engine's --pair-account path, a later convergence).
export async function accountPairSecret(
  userId: string,
  platformUrl: string,
  clusterName: string
): Promise<string> {
  const url = platformUrl.trim().replace(/\/+$/, "").toLowerCase();
  const name = clusterName.trim() || "home";
  const material = `idletoken-account-pair|v1|${userId}|${url}|${name}`;
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(material));
  const hex = Array.from(new Uint8Array(buf))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
  return `ACCT-${hex}`;
}

function mintCode(): string {
  // Unambiguous alphabet (no O/0/I/1) for a code people read aloud.
  const alpha = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
  const buf = new Uint8Array(6);
  crypto.getRandomValues(buf);
  return Array.from(buf, (b) => alpha[b % alpha.length]).join("");
}

// ---- dev-sim provider -----------------------------------------------------
// In-memory cluster that simulates peers arriving after a short delay so the
// pairing UI (peer list, coordinator pick, leave/rejoin) can be exercised
// without a network. NOT a real cluster.
class DevSimPairing implements PairingProvider {
  private state: PairingSnapshot = {
    code: null,
    peers: [],
    coordinatorId: null,
    phase: "idle",
    api: null,
    source: "dev-sim",
  };
  private subs = new Set<(s: PairingSnapshot) => void>();
  private timers: ReturnType<typeof setTimeout>[] = [];
  private seq = 0;

  private emit() {
    const snap = { ...this.state, peers: [...this.state.peers] };
    this.subs.forEach((cb) => cb(snap));
  }

  private clearTimers() {
    this.timers.forEach(clearTimeout);
    this.timers = [];
  }

  private selfPeer(self: SelfInfo, role: NodeRole): PeerNode {
    return { id: "self", hostname: self.hostname, gpu: self.gpu, role, self: true, stage: "joined", online: true };
  }

  async create(self: SelfInfo, code?: string): Promise<void> {
    this.clearTimers();
    this.state = {
      code: code ?? mintCode(),
      peers: [this.selfPeer(self, "coordinator")],
      coordinatorId: "self",
      phase: "idle",
      api: null,
      source: "dev-sim",
    };
    this.emit();
    // Simulate two machines joining, then auto-orchestrate (P4).
    this.timers.push(
      setTimeout(() => this.addSimPeer("dgx-spark", "GB10 (unified)"), 1400),
      setTimeout(() => this.addSimPeer("win-laptop", "RTX 2070"), 2800),
      setTimeout(() => this.orchestrate(), 3600)
    );
  }

  async join(_code: string, self: SelfInfo): Promise<void> {
    this.clearTimers();
    // Joining an existing cluster: a coordinator is already present.
    this.state = {
      code: _code.trim().toUpperCase(),
      peers: [
        { id: "peer-coord", hostname: "win-pc-01", gpu: "RTX 5060 Ti", role: "coordinator", self: false, stage: "joined", online: true },
        this.selfPeer(self, "worker"),
      ],
      coordinatorId: "peer-coord",
      phase: "idle",
      api: null,
      source: "dev-sim",
    };
    this.emit();
    this.timers.push(setTimeout(() => this.orchestrate(), 900));
  }

  async createAccount(self: SelfInfo, _secret: string): Promise<void> {
    // Same simulated flow as create, but account-formed: no shareable code.
    await this.create(self);
    this.state.code = null;
    this.state.accountMode = true;
    this.emit();
  }

  async joinAccount(self: SelfInfo, _secret: string): Promise<void> {
    await this.join("", self);
    this.state.code = null;
    this.state.accountMode = true;
    this.emit();
  }

  async start(allowSolo?: boolean): Promise<void> {
    // allowSolo is not decoration: it is how "run on this machine alone"
    // differs from "create a cluster". The sim used to ignore it and always
    // fabricate two joiners, so a one-machine deployment — a headline mode —
    // could not be seen in the browser build at all, and anything specific to
    // it was unreviewable until it reached a real machine.
    if (!allowSolo) return;   // otherwise the sim orchestrates on its own timers
    this.clearTimers();       // cancel the simulated joiners; go with just this machine
    this.orchestrate();
  }

  async leave(): Promise<void> {
    this.clearTimers();
    // The simulated coordinator is gone, so what it "loaded" is gone with it —
    // the next create latches the model that is selected then. (A real
    // coordinator gets this for free by exiting.)
    forgetSimLoadedModel();
    this.state = { code: null, peers: [], coordinatorId: null, phase: "idle", api: null, source: "dev-sim" };
    this.emit();
  }

  async setCoordinator(peerId: string): Promise<void> {
    this.state.coordinatorId = peerId;
    this.state.peers = this.state.peers.map((p) => ({
      ...p,
      role: p.id === peerId ? "coordinator" : "worker",
    }));
    this.emit();
  }

  private addSimPeer(hostname: string, gpu: string) {
    this.seq += 1;
    this.state.peers.push({ id: `sim-${this.seq}`, hostname, gpu, role: "worker", self: false, stage: "joined", online: true });
    this.emit();
  }

  // P4 auto-orchestration: probe -> split layers -> load -> ready -> API online.
  private orchestrate() {
    const setPeers = (stage: NodeStage) => {
      this.state.peers = this.state.peers.map((p) => ({ ...p, stage }));
    };

    this.state.phase = "probing";
    setPeers("probing");
    this.emit();

    this.timers.push(
      setTimeout(() => {
        // split contiguous layer ranges across the cluster, coordinator first
        const ordered = [...this.state.peers].sort((a, b) =>
          a.role === "coordinator" ? -1 : b.role === "coordinator" ? 1 : 0
        );
        const ranges = splitLayers(ordered.length);
        const byId = new Map(ordered.map((p, i) => [p.id, ranges[i]]));
        this.state.phase = "splitting";
        this.state.peers = this.state.peers.map((p) => {
          const r = byId.get(p.id);
          return { ...p, stage: "assigned", layerLo: r?.[0], layerHi: r?.[1] };
        });
        this.emit();
      }, 700),
      setTimeout(() => {
        this.state.phase = "loading";
        setPeers("loading");
        this.emit();
      }, 1500),
      setTimeout(() => {
        this.state.phase = "ready";
        setPeers("ready");
        // P6: API auto-listens on the coordinator once the cluster is ready.
        this.state.api = { baseUrl: "http://192.168.1.100:8000", status: "online" };
        this.emit();
      }, 2600)
    );
  }

  subscribe(cb: (s: PairingSnapshot) => void): () => void {
    this.subs.add(cb);
    cb({ ...this.state, peers: [...this.state.peers] });
    return () => this.subs.delete(cb);
  }
}

const devSim = new DevSimPairing();

// ---- real provider ----------------------------------------------------------
// Thin RPC shim over the Rust pairing layer (src-tauri/src/pairing.rs): UDP
// beacon discovery + TCP roster on the LAN, then engine materialization via
// the same supervisor the engine card uses. Snapshots arrive as
// `pairing:status` events.
class EnginePairing implements PairingProvider {
  private subs = new Set<(s: PairingSnapshot) => void>();
  private listening = false;

  private async ensureListen() {
    if (this.listening) return;
    this.listening = true;
    const { listen } = await import("@tauri-apps/api/event");
    await listen<PairingSnapshot>("pairing:status", (e) => {
      this.subs.forEach((cb) => cb(e.payload));
    });
  }

  private async call<T = void>(cmd: string, args?: Record<string, unknown>): Promise<T> {
    const { invoke } = await import("@tauri-apps/api/core");
    return invoke<T>(cmd, args);
  }

  async create(self: SelfInfo, code?: string): Promise<void> {
    await this.call("pairing_create", {
      code: code || mintCode(),
      hostname: self.hostname,
      gpu: self.gpu,
      modelPath: self.modelPath ?? "",
      tuning: self.tuning ?? null,
    });
  }

  async join(code: string, self: SelfInfo): Promise<void> {
    await this.call("pairing_join", {
      code: code.trim().toUpperCase(),
      hostname: self.hostname,
      gpu: self.gpu,
      modelPath: self.modelPath ?? "",
      tuning: self.tuning ?? null,
    });
  }

  async createAccount(self: SelfInfo, secret: string): Promise<void> {
    await this.call("pairing_create", {
      code: secret,
      hostname: self.hostname,
      gpu: self.gpu,
      modelPath: self.modelPath ?? "",
      tuning: self.tuning ?? null,
      account: true,
    });
  }

  async joinAccount(self: SelfInfo, secret: string): Promise<void> {
    await this.call("pairing_join", {
      code: secret,
      hostname: self.hostname,
      gpu: self.gpu,
      modelPath: self.modelPath ?? "",
      tuning: self.tuning ?? null,
      account: true,
    });
  }

  async start(allowSolo?: boolean): Promise<void> {
    await this.call("pairing_start", { allowSolo: allowSolo ?? false });
  }

  async leave(): Promise<void> {
    await this.call("pairing_leave");
  }

  async setCoordinator(peerId: string): Promise<void> {
    await this.call("pairing_set_coordinator", { peerId });
  }

  subscribe(cb: (s: PairingSnapshot) => void): () => void {
    this.subs.add(cb);
    this.ensureListen();
    this.call<PairingSnapshot>("pairing_status")
      .then((s) => cb(s))
      .catch(() => {});
    return () => this.subs.delete(cb);
  }
}

const enginePairing = new EnginePairing();

// Selector: the real engine-backed provider inside Tauri, the labeled dev-sim
// in a plain browser. The UI is identical either way (philosophy 14).
export function getPairingProvider(): PairingProvider {
  const inTauri = typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
  return inTauri ? enginePairing : devSim;
}
