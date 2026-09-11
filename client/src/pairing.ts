// Cluster pairing (acceptance P3). Interface-first (philosophy 14): the UI talks
// to a PairingProvider. Two entry modes — same account, or a one-time code —
// both funnel into one LAN cluster. The real provider talks to the coordinator
// over the local RPC and reflects the actual engine cluster (members + stage
// topology); in a plain browser (no Tauri, so no LAN access) a clearly-labeled
// dev-sim drives the UI so the flow can be built and reviewed.
//
// Anything from the sim is marked `source: "dev-sim"` and the UI badges it, so
// simulated peers can never be mistaken for a real cluster (philosophy 9/15).

import type { EngineTuning, OverflowTuning } from "./settings";
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
  /** What this machine brings to the pool, as its OWN uncapped probe measured
   *  it: real currently-free VRAM in bytes, ordinary usable RAM, the smaller
   *  fast expert-RAM pool, and
   *  whether the two are one physical pool (Apple Silicon). The wire names
   *  stay vramFree/ramFree for compatibility. Carried by the roster so any member
   *  can total the cluster and answer "does the model fit on all of us" before
   *  anyone presses Start. 0/absent = a member that does not report it — the
   *  total is then incomplete and the UI must say so, not guess. */
  vramFree?: number;
  ramFree?: number;
  /** RAM that may actually back fast node-local MoE experts. Windows applies
   * its per-process WDDM page-lock budget before publishing this number. */
  ramExpertFree?: number;
  unifiedMemory?: boolean;
  /** True only when this node has the cluster's exact model+precision as a
   * complete, integrity-checked local GGUF. Start is gated on every member. */
  modelReady?: boolean;
  /** Creator-local continuity label for this install. It is a digest prefix,
   *  never the raw device id, and is absent from member roster broadcasts.
   *  It proves only "the same roster identity asked again" — not official
   *  software, honest hardware, or remote attestation. */
  deviceIdentity?: string;
  /** This member is currently asking the creator for the coordinator role.
   *  A request never changes the role by itself. */
  wantsCoordinator?: boolean;
}

export interface SelfInfo {
  hostname: string;
  gpu: string;
  // This machine's local GGUF, as resolved by weights.resolveLocalWeights.
  // Empty = not ready. Cluster joiners also keep a complete local GGUF; the
  // worker imports only its assigned tensors from that file.
  modelPath?: string;
  // Settings-derived engine tuning (API bind/token, inter-stage port,
  // discovery port). Omitted = the Rust side's defaults (the historical
  // hard-coded ports). See settings.engineTuning().
  tuning?: EngineTuning;
}

// Cluster-wide orchestration phase (P4) and the exposed API (P6).
export type OrchestrationPhase = "idle" | "starting" | "probing" | "splitting" | "loading" | "ready";
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
  // The creator-selected identity every joining machine must prepare.
  modelId?: string;
  quant?: string;
  // True on the creator while the roster is still open and big enough to
  // launch — the UI shows the "start cluster" button. (Real provider only;
  // the dev-sim auto-orchestrates.)
  canStart?: boolean;
  // Why the last join attempt failed; null/absent while nothing has. (Real
  // provider only — the dev-sim never fails.)
  lastError?: PairingError | null;
  /** Set together with `lastError.code === "modelNotReady"`: the model and
   *  precision the cluster refused us for not having. A refused joiner never
   *  reaches the roster, so this refusal is the only place it can learn what to
   *  fetch. Two fields, not a sentence — the UI turns them into one exact
   *  download. */
  /** What the cluster demanded when it refused this machine. `ctx` is 0 from a
   *  pre-2026-09-02 creator that did not declare its window. */
  requiredModel?: { modelId: string; quant: string; ctx?: number } | null;
  /** True only in the creator process. Missing values from an older native
   *  backend fail closed and never expose a role-approval action. */
  isCreator?: boolean;
}

export const COORDINATOR_ROLE_RISK =
  "The coordinator can see local prompts and responses in plaintext and controls the cluster. " +
  "The device label shows roster continuity only; it does not prove official software or honest hardware.";

/** Whether this exact snapshot is eligible to present an approval action. */
export function canApproveCoordinatorRequest(
  snapshot: PairingSnapshot,
  peer: PeerNode | undefined,
): peer is PeerNode {
  return snapshot.isCreator === true
    && snapshot.phase === "idle"
    && !!peer
    && peer.self === false
    && peer.role !== "coordinator"
    && peer.online !== false
    && peer.wantsCoordinator === true
    && typeof peer.deviceIdentity === "string"
    && peer.deviceIdentity.length > 0;
}

export function coordinatorApprovalPrompt(peer: PeerNode): string {
  const device = peer.deviceIdentity || "unavailable";
  return `Approve ${peer.hostname} (device ${device}) as coordinator?\n\n${COORDINATOR_ROLE_RISK}`;
}

export type CoordinatorApprovalOutcome = "approved" | "cancelled" | "stale";

/**
 * UI-side confirmation flow. The native `pairing_set_coordinator` command is
 * still the enforcement boundary and re-checks creator/request/member state;
 * this helper only makes the security decision explicit to the person.
 */
export async function approveCoordinatorRequest(
  provider: Pick<PairingProvider, "setCoordinator">,
  snapshot: PairingSnapshot,
  peerId: string,
  confirmApproval: (message: string) => boolean | Promise<boolean>,
): Promise<CoordinatorApprovalOutcome> {
  const peer = snapshot.peers.find((candidate) => candidate.id === peerId);
  if (!canApproveCoordinatorRequest(snapshot, peer)) return "stale";
  if (!await confirmApproval(coordinatorApprovalPrompt(peer))) return "cancelled";
  // The snapshot captured by this click may become stale while the modal is
  // open. Do not pretend a second read of that same object closes the race:
  // the native command performs the authoritative current-roster check across
  // the IPC boundary and rejects a withdrawal or leave that won the race.
  await provider.setCoordinator(peerId);
  return "approved";
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
  start(allowSolo?: boolean, modelPath?: string, overflow?: OverflowTuning): Promise<void>;
  // Publish a newly downloaded and verified local copy to the roster. The
  // path stays on this machine; peers receive only modelReady.
  updateModel(modelId: string, quant: string, modelPath: string): Promise<void>;
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
// platform URL (accounts from different platforms never collide). Cluster
// names were removed from the product in 2026-08; a stale stored name must not
// split two machines that are signed in to the same account. The fixed final
// field below preserves wire compatibility with older clients that kept the
// former default, while giving the stored setting no meaning at all. The
// secret itself is never
// broadcast, and since 2026-08-20 (audit A-P0-3) neither is anything derived
// from it: the UDP beacon carries a random session id and a per-packet nonce
// only, and the full value travels solely inside the LAN TCP join as proof —
// the same trust level as a shared code.
// Honesty: this proves "derived from the same account material", which matches
// the code-mode trust bar; it is not a platform-verified JWT handshake (that is
// the engine's --pair-account path, a later convergence).
export async function accountPairSecret(
  userId: string,
  platformUrl: string
): Promise<string> {
  const url = platformUrl.trim().replace(/\/+$/, "").toLowerCase();
  // Protocol constant, not a cluster name. Keeping the old default bytes lets
  // a newly updated client still find an older client that never changed the
  // now-retired setting.
  const material = `idletoken-account-pair|v1|${userId}|${url}|IdleToken-Home`;
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
    isCreator: false,
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
    // Memory the dev-sim reports for this machine, so the pooled verdict can
    // be exercised in a browser (the real path fills these from the probe).
    return { id: "self", hostname: self.hostname, gpu: self.gpu, role, self: true, stage: "joined", online: true, modelReady: true,
             vramFree: 13.2 * 1024 ** 3, ramFree: 20.6 * 1024 ** 3,
             ramExpertFree: 15.25 * 1024 ** 3, unifiedMemory: false };
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
      isCreator: true,
    };
    this.emit();
    // Simulate two machines joining, then auto-orchestrate (P4).
    // Generic names on purpose (2026-08-20 audit, D-P1-5's client half): these
    // used to be the maintainers' own test-bed machines, which then shipped in
    // the public bundle as a list of somebody's real computers.
    this.timers.push(
      setTimeout(() => this.addSimPeer("machine-b", "GB10 (unified)"), 1400),
      setTimeout(() => this.addSimPeer("machine-c", "RTX 2070"), 2800),
      setTimeout(() => this.orchestrate(), 3600)
    );
  }

  async join(_code: string, self: SelfInfo): Promise<void> {
    this.clearTimers();
    // Joining an existing cluster: a coordinator is already present.
    this.state = {
      code: _code.trim().toUpperCase(),
      peers: [
        { id: "peer-coord", hostname: "machine-a", gpu: "RTX 5060 Ti", role: "coordinator", self: false, stage: "joined", online: true, modelReady: true,
          vramFree: 13.2 * 1024 ** 3, ramFree: 20.6 * 1024 ** 3,
          ramExpertFree: 15.25 * 1024 ** 3, unifiedMemory: false },
        this.selfPeer(self, "worker"),
      ],
      coordinatorId: "peer-coord",
      phase: "idle",
      api: null,
      source: "dev-sim",
      isCreator: false,
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

  async start(allowSolo?: boolean, _modelPath?: string, _overflow?: OverflowTuning): Promise<void> {
    // allowSolo is not decoration: it is how "run on this machine alone"
    // differs from "create a cluster". The sim used to ignore it and always
    // fabricate two joiners, so a one-machine deployment — a headline mode —
    // could not be seen in the browser build at all, and anything specific to
    // it was unreviewable until it reached a real machine.
    if (!allowSolo) return;   // otherwise the sim orchestrates on its own timers
    this.clearTimers();       // cancel the simulated joiners; go with just this machine
    this.orchestrate();
  }

  async updateModel(_modelId: string, _quant: string, _modelPath: string): Promise<void> {}

  async leave(): Promise<void> {
    this.clearTimers();
    // The simulated coordinator is gone, so what it "loaded" is gone with it —
    // the next create latches the model that is selected then. (A real
    // coordinator gets this for free by exiting.)
    forgetSimLoadedModel();
    this.state = { code: null, peers: [], coordinatorId: null, phase: "idle", api: null, source: "dev-sim", isCreator: false };
    this.emit();
  }

  async setCoordinator(peerId: string): Promise<void> {
    const peer = this.state.peers.find((candidate) => candidate.id === peerId);
    if (!canApproveCoordinatorRequest(this.state, peer)) {
      throw new Error("[PAIR_ROLE_NOT_REQUESTED] that machine is not requesting the coordinator role");
    }
    this.state.coordinatorId = peerId;
    this.state.peers = this.state.peers.map((p) => ({
      ...p,
      role: p.id === peerId ? "coordinator" : "worker",
      wantsCoordinator: p.id === peerId ? false : p.wantsCoordinator,
    }));
    this.emit();
  }

  private addSimPeer(hostname: string, gpu: string) {
    this.seq += 1;
    // Unified-memory machines report one pool (the max, not the sum) — the
    // same rule the engine applies, exercised here by the DGX fixture.
    const unified = /unified/i.test(gpu);
    this.state.peers.push({
      id: `sim-${this.seq}`,
      hostname,
      gpu,
      role: "worker",
      self: false,
      stage: "joined",
      online: true,
      modelReady: true,
      deviceIdentity: `sim-${this.seq.toString().padStart(4, "0")}`,
      // One simulated request keeps the browser-only review path observable.
      wantsCoordinator: this.seq === 1,
      vramFree: (unified ? 96 : 6.5) * 1024 ** 3,
      ramFree: (unified ? 96 : 12) * 1024 ** 3,
      ramExpertFree: unified ? 0 : 12 * 1024 ** 3,
      unifiedMemory: unified,
    });
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
        // Loopback, like the real thing: the inference API is coordinator-local
        // (2026-08-15), so a LAN address here was both a stale claim and a real
        // internal address in a shipped bundle.
        this.state.api = { baseUrl: "http://127.0.0.1:8000", status: "online" };
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

  async start(allowSolo?: boolean, modelPath?: string, overflow?: OverflowTuning): Promise<void> {
    await this.call("pairing_start", {
      allowSolo: allowSolo ?? false,
      modelPath: modelPath ?? null,
      overflowTuning: overflow ?? null,
    });
  }

  async updateModel(modelId: string, quant: string, modelPath: string): Promise<void> {
    await this.call("pairing_update_model", { modelId, quant, modelPath });
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
