import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { LangContext, STRINGS, useI18n, type Lang } from "./i18n";
import { getResourceProvider } from "./provider";
import { getEngineProvider, type EngineLogLine, type EngineRole, type EngineStatus } from "./provider/engine";
import { uiTestDirectives } from "./testHooks";
import type { NodeSnapshot, ClusterState } from "./types";
import { HW_NO_GPU, HW_CC_TOO_LOW, HW_DRIVER_TOO_OLD, HW_VRAM_TOO_SMALL } from "./types";
import { getModel, getManifest, defaultQuant, estimateClusterCapacity, pickBestFittingModel, type ModelSpec } from "./models";
import { resolveLocalWeights, fetchWeights, onFetchProgress, defaultModelDir, cancelFetch, resolveDownload, weightsState } from "./weights";
import { loadSettings, saveSettings, settingsWerePersisted, effectiveCaps, engineTuning, TIERS, type AppSettings } from "./settings";
import { buildDiagnosticsBundle } from "./diagnostics";
import { getAuthProvider, type Session } from "./auth";
import SettingsPanel from "./SettingsPanel";
import Capability from "./Capability";
import AuthScreen from "./AuthScreen";
import PairingPanel from "./PairingPanel";
import Chat from "./Chat";
import { inTauri, platformGate } from "./platform";
import { accountPairSecret, getPairingProvider, type PairingSnapshot, type ClusterApi } from "./pairing";
import { fmtBytes, fmtGiB, pct } from "./format";

type Theme = "dark" | "light";

let uiTestRan = false;

// Ship a UI-test assertion result to the shell's stderr (see ui_test_report
// in src-tauri/src/main.rs). No-op outside Tauri.
function reportTest(tag: string, data: unknown) {
  import("@tauri-apps/api/core").then(({ invoke }) =>
    invoke("ui_test_report", { tag, data: JSON.stringify(data) }).catch(() => {})
  );
}

function usePersisted<T extends string>(key: string, initial: T): [T, (v: T) => void] {
  const [v, setV] = useState<T>(() => (localStorage.getItem(key) as T) || initial);
  const set = (next: T) => {
    setV(next);
    localStorage.setItem(key, next);
  };
  return [v, set];
}

// ---- top bar --------------------------------------------------------------
// Slim since the sidebar IA: places (chat/cluster/settings) live in the
// sidebar; the topbar keeps identity, live status and quick toggles. Account
// moved here from the page footer — "who am I" belongs top-right by habit.
function AccountMenu(props: { session: Session | null; onSignIn: () => void; onSignOut: () => void }) {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent | KeyboardEvent) => {
      if (e instanceof KeyboardEvent && e.key !== "Escape") return;
      setOpen(false);
    };
    document.addEventListener("click", close);
    document.addEventListener("keydown", close);
    return () => {
      document.removeEventListener("click", close);
      document.removeEventListener("keydown", close);
    };
  }, [open]);
  if (!props.session) {
    return (
      <button className="iconbtn" onClick={props.onSignIn}>
        {t("auth.submitSignIn")}
      </button>
    );
  }
  return (
    <div className="acct" onClick={(e) => e.stopPropagation()}>
      <button className="iconbtn acct__btn" aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen(!open)}>
        {props.session.email}
      </button>
      {open ? (
        <div className="acct__menu" role="menu">
          <button
            className="acct__item"
            role="menuitem"
            onClick={() => {
              setOpen(false);
              props.onSignOut();
            }}
          >
            {t("auth.signOut")}
          </button>
        </div>
      ) : null}
    </div>
  );
}

// ---- primary places (top-nav; matches portal pv-topnav convention) --------
type View = "cluster" | "chat" | "settings";

// Three places, in the order they get reached for: talk to the cluster, look
// at the machines behind it, turn knobs.
//
// The marketplace used to be a fourth place (2026-08-10: removed). Browsing
// other people's compute is a browser job — the portal already does it, and
// duplicating it here meant two IAs to keep in sync. What genuinely needs THIS
// machine — listing this cluster, the credit balance, API keys — is a settings
// category now (Settings → Sharing & earnings).
const NAV: Array<{ id: View; icon: string }> = [
  // Simple geometric glyphs drawn inline — consistent 18px stroke icons.
  { id: "chat", icon: "M4 5h16v11H9l-5 4z" },
  { id: "cluster", icon: "M4 5h6v6H4zM14 5h6v6h-6zM4 15h6v6H4zM14 15h6v6h-6z" },
  { id: "settings", icon: "M12 8a4 4 0 100 8 4 4 0 000-8zM12 2v3M12 19v3M2 12h3M19 12h3M5 5l2 2M17 17l2 2M19 5l-2 2M7 17l-2 2" },
];

function TopBar(props: {
  cluster: ClusterState;
  theme: Theme;
  session: Session | null;
  view: View;
  onView: (v: View) => void;
  onToggleTheme: () => void;
  onToggleLang: () => void;
  onGoCluster: () => void;
  onSignIn: () => void;
  onSignOut: () => void;
}) {
  const { t } = useI18n();
  const clusterKey =
    props.cluster === "ready" ? "cluster.ready" : props.cluster === "joining" ? "cluster.joining" : "cluster.standalone";
  return (
    <header className="topbar">
      <div className="brand">
        <span className="brand__mark">IdleToken</span>
      </div>
      <nav className="topnav" aria-label="primary">
        {NAV.map((item) => (
          <button
            key={item.id}
            className={`topnav__item${props.view === item.id ? " is-on" : ""}`}
            aria-current={props.view === item.id ? "page" : undefined}
            onClick={() => props.onView(item.id)}
          >
            <svg viewBox="0 0 24 24" className="topnav__icon" aria-hidden="true">
              <path d={item.icon} fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinejoin="round" strokeLinecap="round" />
            </svg>
            <span className="topnav__label">{t(`nav.${item.id}` as const)}</span>
          </button>
        ))}
      </nav>
      <span className="topbar__spacer" />
      <button className={`pill pill--${props.cluster}`} onClick={props.onGoCluster} title={t("nav.cluster")}>
        <span className="pill__dot" />
        {t(clusterKey)}
      </button>
      <button className="iconbtn" onClick={props.onToggleLang} aria-label={t("lang.switch")}>
        {t("lang.switch")}
      </button>
      <button
        className="iconbtn"
        onClick={props.onToggleTheme}
        aria-label={props.theme === "dark" ? t("theme.toLight") : t("theme.toDark")}
        title={props.theme === "dark" ? t("theme.toLight") : t("theme.toDark")}
      >
        {props.theme === "dark" ? "☾" : "☀"}
      </button>
      <AccountMenu session={props.session} onSignIn={props.onSignIn} onSignOut={props.onSignOut} />
    </header>
  );
}

// ---- this machine: one dense card = probe strip + capacity guidance --------
// Replaces the old GPU card + stat tiles + layer spine trio (2026-07 UX pass):
// the probe is supporting detail, so it gets one row; the pixels go to the
// question that actually matters before pairing — "is my hardware enough,
// and how far off am I?" (estimateClusterCapacity, engine-estimate mirror).
function NodeCapacityCard(props: {
  snap: NodeSnapshot;
  model: ModelSpec;
  quant: string; // selected precision → sizes the weight bytes in the estimate
  tier: { id: number; ctx: number };
  nNodes: number; // known cluster size when paired; nominal estimate otherwise
  // Once the cluster is READY the layer plan is decided — the capacity pitch
  // has done its job, so it collapses to one line (lifecycle-aware layout).
  compact?: boolean;
}) {
  const { t } = useI18n();
  const s = props.snap;
  const cap = estimateClusterCapacity(props.model, s, props.tier.ctx, props.nNodes, props.quant);
  const GB = (b: number) => Math.round(b / 1024 ** 3);
  const vUsable = fmtGiB(s.vram_usable);
  const vTotal = fmtGiB(s.vram_total);
  const ram = fmtGiB(s.ram_usable);
  const disk = fmtBytes(s.disk_avail);
  const total = props.model.totalLayers;
  const ticks = useMemo(() => Array.from({ length: total }), [total]);
  const ctxLabel = props.tier.ctx >= 1048576 ? "1M" : `${props.tier.ctx / 1024}K`;
  // Hardware floor: the engine decided, the UI only renders the verdict. A
  // blocked machine must SAY SO up front — otherwise the card looks healthy and
  // the failure surfaces much later as a mock fallback or garbage tokens.
  const hwMsg =
    s.hw_status === HW_NO_GPU ? t("node.hw.noGpu")
    : s.hw_status === HW_CC_TOO_LOW ? t("node.hw.ccLow")
    : s.hw_status === HW_DRIVER_TOO_OLD ? t("node.hw.driverOld")
    : s.hw_status === HW_VRAM_TOO_SMALL ? t("node.hw.vramSmall")
    : "";
  return (
    <section className="card node-card">
      {hwMsg && (
        <div className="hw-blocked" role="alert">
          <strong>{t("node.hw.blocked")}</strong>
          <span>{hwMsg}</span>
          {s.hw_reason && <span className="hw-blocked__detail">{s.hw_reason}</span>}
        </div>
      )}
      <div className="node-strip">
        <div className="nstat nstat--gpu">
          <span className="nstat__k">{t("node.gpu")}</span>
          <span className="nstat__v">{s.gpu_name || "—"}</span>
          <span className="nstat__sub">
            cc {s.cc_major}.{s.cc_minor}
            {s.driver_version ? ` · ${t("node.driver")} ${s.driver_version}` : ""}
            {s.unified_memory ? ` · ${t("node.unified")}` : ""}
          </span>
        </div>
        <div className="nstat nstat--bar">
          <span className="nstat__k">{t("node.vram")}</span>
          <span className="nstat__v">
            {vUsable.value}
            <span className="unit">/ {vTotal.value} {vTotal.unit}</span>
          </span>
          <div className="track track--mini">
            <span className="track__usable" style={{ width: `${pct(s.vram_usable, s.vram_total)}%` }} />
            <span className="track__used" style={{ width: `${pct(s.vram_used_other, s.vram_total)}%` }} />
          </div>
        </div>
        <div className="nstat">
          <span className="nstat__k">{t("node.ram")}</span>
          <span className="nstat__v">
            {ram.value}
            <span className="unit">{ram.unit}</span>
          </span>
        </div>
        <div className="nstat">
          <span className="nstat__k">{t("node.disk")}</span>
          <span className="nstat__v">
            {disk.split(" ")[0]}
            <span className="unit">{disk.split(" ")[1]}</span>
          </span>
        </div>
        <div className="nstat">
          <span className="nstat__k">{t("node.cpu")}</span>
          <span className="nstat__v">{s.cpu_count}</span>
        </div>
      </div>

      {props.compact ? (
        <p className="capacity__text capacity__text--compact">
          {t("capacity.have", { have: GB(cap.haveBytes) })}
        </p>
      ) : (
      <div className="capacity">
        <div className="capacity__head">
          <span className="card__label">{t("capacity.title")}</span>
          <span className="capacity__need">
            {t("capacity.need", { need: GB(cap.needBytes), model: props.model.label, tier: ctxLabel, n: props.nNodes })}
          </span>
        </div>
        <div className="spine" role="img" aria-label={t("spine.caption", { n: cap.hostableLayers, total })}>
          {ticks.map((_, i) => (
            <span key={i} className={`tick${i < cap.hostableLayers ? " tick--on" : ""}`} />
          ))}
        </div>
        <div className="spine-scale">
          <span>layer 0</span>
          <span>{total - 1}</span>
        </div>
        <p className="capacity__text">
          {t("capacity.have", { have: GB(cap.haveBytes) })}{" "}
          {cap.gapBytes > 0 ? (
            <span className="capacity__gap">{t("capacity.gap", { gap: GB(cap.gapBytes) })}</span>
          ) : (
            <span className="capacity__ok">{t("capacity.enough")}</span>
          )}
        </p>
      </div>
      )}
    </section>
  );
}

// ---- engine lifecycle card (P1) --------------------------------------------
// The native engine is a separate process the client supervises (philosophy
// 17). This card shows its live state and lets the user start/stop it; crash →
// backoff restarts are the supervisor's job and surface here via events.
const LOG_TAIL = 6;

function EngineCard() {
  const { t } = useI18n();
  const [st, setSt] = useState<EngineStatus | null>(null);
  const [lines, setLines] = useState<EngineLogLine[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    const eng = getEngineProvider();
    let live = true;
    eng.status().then((s) => live && setSt(s)).catch((e) => live && setErr(String(e?.message ?? e)));
    eng.logs(LOG_TAIL).then((ls) => live && setLines(ls)).catch(() => {});
    const unStatus = eng.onStatus((s) => setSt(s));
    const unLog = eng.onLog((l) => setLines((prev) => [...prev.slice(-(LOG_TAIL - 1)), l]));
    return () => {
      live = false;
      unStatus();
      unLog();
    };
  }, []);

  const active = st !== null && st.state !== "stopped" && st.state !== "crashed";
  const toggle = async () => {
    if (busy || !st) return;
    setBusy(true);
    setErr(null);
    try {
      if (active) await getEngineProvider().stop();
      else await getEngineProvider().start("worker");
    } catch (e) {
      setErr(String((e as Error)?.message ?? e));
    }
    setBusy(false);
  };

  const state = st?.state ?? "stopped";
  // Collapsed by default: the engine is auto-managed by pairing orchestration —
  // manual start/stop is a troubleshooting surface, not a daily control. It
  // auto-expands when something is actually wrong (crash) so failures are never
  // hidden behind the fold.
  return (
    <details className="engine-details" open={state === "crashed" || undefined}>
      <summary className="engine-details__summary">
        <span className="engine-details__label">{t("engine.title")}</span>
        <span className={`stage-tag stage-tag--${state === "running" ? "ready" : state === "crashed" ? "bad" : "idle"}`}>
          <span className="stage-tag__dot" />
          {t(`engine.state.${state}` as const)}
        </span>
      </summary>
      <div className="engine-details__body">
        <div className="engine-card__head">
          <span className="engine-hint">{t("engine.advancedNote")}</span>
          <button className="btn-secondary engine-card__btn" disabled={busy || st === null} onClick={toggle}>
            {active ? t("engine.stop") : t("engine.start")}
        </button>
        </div>
        {st?.state === "running" && st.pid ? (
          <div className="engine-meta">
            pid {st.pid}
            {st.role ? <> · {st.role}</> : null}
            {st.restarts > 0 ? <> · {t("engine.restarts", { n: st.restarts })}</> : null}
          </div>
        ) : null}
        {st?.state === "crashed" ? (
          <div className="engine-meta engine-meta--bad">
            {t("engine.crashedHint", { code: st.lastExitCode ?? "?" })}
          </div>
        ) : null}
        {err ? <div className="engine-meta engine-meta--bad">{err}</div> : null}
        {lines.length > 0 ? (
          <pre className="engine-log" aria-label={t("engine.logs")}>
            {lines.map((l) => l.line).join("\n")}
          </pre>
        ) : null}
      </div>
    </details>
  );
}

function FixtureBanner() {
  const { t } = useI18n();
  return (
    <div className="fixture">
      <span className="fixture__badge">{t("fixture.badge")}</span>
      <div className="fixture__text">
        <b>{t("fixture.title")}</b>
        <p>{t("fixture.body")}</p>
      </div>
    </div>
  );
}

// ---- cluster activity row (engine GET /v1/stats) ---------------------------
// "What did my cluster do" at a glance: served requests, total tokens, last
// decode speed, uptime. Polled from the engine's stats endpoint in Tauri; the
// browser dev build synthesizes numbers only inside the already-simulated
// dev-sim cluster (consistent with its labeling).
interface ClusterStats {
  requests: number;
  input_tokens: number;
  output_tokens: number;
  cache_hits?: number;     // KV prefix reuse hits (undefined on older engines)
  cached_tokens?: number;  // cumulative prefill tokens saved
  uptime_s: number;
  last_tok_per_s: number;
}

function ActivityRow(props: { api: ClusterApi; source: "engine" | "dev-sim" }) {
  const { t } = useI18n();
  const [stats, setStats] = useState<ClusterStats | null>(null);

  useEffect(() => {
    if (props.api.status !== "online") return;
    let live = true;
    let simBase = { requests: 0, tokens: 0 };
    const tick = async () => {
      if (!live) return;
      if (inTauri()) {
        try {
          const { invoke } = await import("@tauri-apps/api/core");
          const v = await invoke<ClusterStats>("api_stats", { baseUrl: props.api.baseUrl });
          if (live) setStats(v);
        } catch {
          /* stats are best-effort; the row just stays as-is */
        }
      } else if (props.source === "dev-sim") {
        simBase = { requests: simBase.requests + (Math.random() < 0.4 ? 1 : 0), tokens: simBase.tokens + 40 };
        if (live)
          setStats({
            requests: simBase.requests,
            input_tokens: Math.floor(simBase.tokens * 0.6),
            output_tokens: Math.floor(simBase.tokens * 0.4),
            uptime_s: Math.floor(performance.now() / 1000) + 60,
            last_tok_per_s: 9.5,
          });
      }
    };
    tick();
    const timer = setInterval(tick, 5000);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, [props.api.status, props.api.baseUrl, props.source]);

  if (!stats) return null;
  const up = stats.uptime_s;
  const uptimeLabel =
    up >= 86400
      ? t("stats.days", { n: Math.floor(up / 86400) })
      : up >= 3600
        ? t("stats.hours", { n: Math.floor(up / 3600) })
        : t("stats.mins", { n: Math.max(1, Math.floor(up / 60)) });
  return (
    <div className="activity">
      <span className="activity__item">
        <b>{stats.requests.toLocaleString()}</b> {t("stats.requests")}
      </span>
      <span className="activity__item">
        <b>{(stats.input_tokens + stats.output_tokens).toLocaleString()}</b> tokens
      </span>
      {stats.last_tok_per_s > 0 ? (
        <span className="activity__item">
          <b>{stats.last_tok_per_s.toFixed(1)}</b> tok/s
        </span>
      ) : null}
      {(stats.cached_tokens ?? 0) > 0 ? (
        <span className="activity__item" title={t("stats.cacheTitle")}>
          {t("stats.cache")} <b>{stats.cached_tokens!.toLocaleString()}</b> tok
        </span>
      ) : null}
      <span className="activity__item">
        {t("stats.uptime")} <b>{uptimeLabel}</b>
      </span>
    </div>
  );
}

// ---- chat launcher: the cluster card's quick box now LEADS to the chat view
// (a box that looks like chat must be chat — the one-shot answer lived here
// before and violated that expectation).
function ChatLauncher(props: { onStart: (q: string) => void }) {
  const { t } = useI18n();
  const [q, setQ] = useState("");
  const go = () => {
    if (q.trim()) props.onStart(q.trim());
  };
  return (
    <div className="tryit__row cluster-chatlaunch">
      <input
        className="tryit__input"
        value={q}
        placeholder={t("tryit.placeholder")}
        onChange={(e) => setQ(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && !e.nativeEvent.isComposing && go()}
      />
      <button className="btn-primary tryit__send" disabled={!q.trim()} onClick={go}>
        {t("tryit.send")}
      </button>
    </div>
  );
}

// ---- cluster: the product's home on the dashboard --------------------------
// The cluster (not this machine) is what the user is here for. Empty state =
// the onboarding hero; active state = members, API address and the try-it box,
// with the pairing panel as the management surface.
function ClusterCard(props: {
  pair: PairingSnapshot | null;
  // Does the selected model+precision fit THIS machine alone? The local row
  // stays on screen either way and says why it's unavailable — "the option is
  // missing" and "the option is unavailable to you, here's why" look identical
  // to a user, and only one of them is true.
  //
  // Deliberately no byte figure here: NodeCapacityCard sits beside this card
  // quoting the shortfall for an N-machine cluster (≈61 GB), while this row's
  // question is N=1 (≈54.5 GiB). Both are right, they answer different
  // questions, and two nearly-equal numbers side by side just read as a bug.
  // The card owns the quantities; this row owns the choice.
  fitsStandalone?: boolean;
  onServeStandalone?: () => void;
  onCreate: () => void;
  onJoin: () => void;
  onManage: () => void;
  onOpenSharing: () => void;
  onStartChat: (q: string) => void;
}) {
  const { t } = useI18n();
  const [copiedApi, setCopiedApi] = useState(false);
  const [copiedCode, setCopiedCode] = useState(false);
  const snap = props.pair;
  const active = snap !== null && snap.peers.length > 0;
  const anyError = active && snap.peers.some((p) => p.stage === "error");
  const fits = !!props.fitsStandalone;

  if (!active) {
    return (
      // No pitch here (2026-08-10): whoever is looking at this screen already
      // installed the thing. The sell lives on the portal landing page; this
      // card only answers "what do I press now".
      <section className="card cluster-card cluster-card--empty">
        <h2 className="cluster-empty__title">{t("cluster.emptyTitle")}</h2>

        {/* Two ways to deploy, always both on screen. They used to be one
            either/or row driven by fitsStandalone, which meant a machine big
            enough to go solo was never offered "join someone else's cluster",
            and a machine too small never saw the local option at all — the
            path you can't take should say why, not disappear. */}
        <div className="deploy-opt">
          <div className="deploy-opt__text">
            <h3 className="deploy-opt__title">{t("deploy.local")}</h3>
            <p className="deploy-opt__body">
              {fits ? t("deploy.local.fits") : t("deploy.local.tooBig")}
            </p>
          </div>
          <button
            className={fits ? "btn-primary" : "btn-secondary"}
            disabled={!fits}
            onClick={props.onServeStandalone}
          >
            {t("cluster.serveLocal")}
          </button>
        </div>

        <div className="deploy-opt">
          <div className="deploy-opt__text">
            <h3 className="deploy-opt__title">{t("deploy.cluster")}</h3>
            <p className="deploy-opt__body">{t("deploy.cluster.body")}</p>
          </div>
          <div className="deploy-opt__actions">
            <button className={fits ? "btn-secondary" : "btn-primary"} onClick={props.onCreate}>
              {t("cluster.create")}
            </button>
            <button className="btn-secondary" onClick={props.onJoin}>
              {t("cluster.joinCode")}
            </button>
          </div>
        </div>
      </section>
    );
  }

  // idle-with-roster = the cluster is forming, not "standalone": while the
  // roster is open the card must carry everything the creator needs — the
  // join code and the start button — so closing the modal loses nothing.
  const phaseKey =
    snap.phase === "ready" ? "cluster.ready" : snap.phase === "idle" ? "cluster.waiting" : (`pairing.phase.${snap.phase}` as const);
  const copyApi = async () => {
    if (!snap.api) return;
    try {
      await navigator.clipboard.writeText(snap.api.baseUrl);
      setCopiedApi(true);
      setTimeout(() => setCopiedApi(false), 1500);
    } catch {
      /* shown regardless */
    }
  };
  const copyCode = async () => {
    if (!snap.code) return;
    try {
      await navigator.clipboard.writeText(snap.code);
      setCopiedCode(true);
      setTimeout(() => setCopiedCode(false), 1500);
    } catch {
      /* shown regardless */
    }
  };

  return (
    <section className={`card cluster-card${snap.phase === "ready" ? " cluster-card--ready" : ""}`}>
      <div className="cluster-head">
        <span className="card__label">{t("cluster.title")}</span>
        <span className={`stage-tag stage-tag--${snap.phase === "ready" ? "ready" : "busy"}`}>
          <span className="stage-tag__dot" />
          {t(phaseKey)}
        </span>
        <span className="cluster-head__count">{t("cluster.machines", { n: snap.peers.length })}</span>
        <button className="linkbtn cluster-head__manage" onClick={props.onManage}>
          {t("cluster.manage")}
        </button>
      </div>

      {snap.phase === "idle" && snap.code ? (
        <div className="cluster-code">
          <span className="cluster-code__label">{t("pairing.yourCode")}</span>
          <span className="cluster-code__code">{snap.code}</span>
          <button className="iconbtn" onClick={copyCode}>
            {copiedCode ? t("pairing.copied") : t("pairing.copy")}
          </button>
        </div>
      ) : null}

      <div className="cluster-peers">
        {snap.peers.map((p) => (
          <div key={p.id} className="cpeer">
            <span className={`cpeer__dot cpeer__dot--${p.stage}`} />
            <span className="cpeer__host">
              {p.hostname}
              {p.self ? <span className="cpeer__you"> · {t("pairing.you")}</span> : null}
            </span>
            {p.role === "coordinator" ? <span className="role-tag">{t("pairing.coordinator")}</span> : null}
            <span className="cpeer__meta">
              {p.layerLo !== undefined && p.layerHi !== undefined
                ? t("pairing.layers", { lo: p.layerLo, hi: p.layerHi - 1 })
                : t(`pairing.stage.${p.stage}` as const)}
            </span>
          </div>
        ))}
      </div>

      {anyError ? <p className="cluster-hint cluster-hint--bad">{t("cluster.errorHint")}</p> : null}

      {snap.phase === "loading" ? <p className="cluster-hint">{t("cluster.loadingHint")}</p> : null}

      {snap.canStart ? (
        <button className="btn-primary btn-block cluster-start" onClick={() => getPairingProvider().start()}>
          {t("pairing.startCluster", { n: snap.peers.length })} →
        </button>
      ) : null}

      {snap.api ? (
        <div className="cluster-api">
          <div className="cluster-api__label">
            {t("cluster.api")}
            <span
              className={`stage-tag stage-tag--${
                snap.api.status === "online" ? "ready" : snap.api.status === "offline" ? "bad" : "busy"
              }`}
            >
              <span className="stage-tag__dot" />
              {snap.api.status === "online"
                ? t("pairing.api.online")
                : snap.api.status === "offline"
                  ? t("cluster.apiOffline")
                  : t("pairing.api.starting")}
            </span>
          </div>
          <div className="cluster-api__row">
            <code className="cluster-api__url">{snap.api.baseUrl}</code>
            <button className="iconbtn" onClick={copyApi}>
              {copiedApi ? t("pairing.copied") : t("pairing.copy")}
            </button>
          </div>
          <p className="cluster-api__hint">{t("cluster.apiHint")}</p>
          <ActivityRow api={snap.api} source={snap.source} />
        </div>
      ) : null}

      {snap.api?.status === "online" ? <ChatLauncher onStart={props.onStartChat} /> : null}
      {/* "What can I run?" — the first question after installing. Uses the
          cluster endpoint once an API is up (whole pool), the local engine
          otherwise (this machine alone). */}
      <Capability apiBaseUrl={snap.api?.status === "online" ? snap.api.baseUrl : null} />

      {snap.phase === "ready" ? (
        <div className="cluster-share">
          <button className="linkbtn" onClick={props.onOpenSharing}>
            {t("cluster.share")} →
          </button>
        </div>
      ) : null}
    </section>
  );
}

// ---- dashboard ------------------------------------------------------------
// The cluster leads; this machine + engine diagnostics sit beside it on wide
// windows (>=1180px two-column grid) and below it on narrow ones.
function Dashboard(props: {
  snap: NodeSnapshot;
  model: ModelSpec;
  quant: string;
  tier: { id: number; ctx: number };
  pair: PairingSnapshot | null;
  onServeStandalone: () => void;
  onCreateCluster: () => void;
  onJoinCluster: () => void;
  onManageCluster: () => void;
  onOpenSharing: () => void;
  onStartChat: (q: string) => void;
}) {
  const { t } = useI18n();
  const s = props.snap;
  const nNodes = props.pair && props.pair.peers.length > 0 ? props.pair.peers.length : 3;
  // Single-node-first: does the selected model+precision fit THIS machine alone
  // (N=1)? If so the empty-state leads with "serve locally" instead of pairing.
  // The shortfall goes down with it — the local row reports how far off it is
  // rather than just going quiet.
  const standalone = estimateClusterCapacity(props.model, s, props.tier.ctx, 1, props.quant);
  return (
    <main className="main main--wide">
      {s.source === "dev-fixture" ? <FixtureBanner /> : null}
      <div className="dash-grid">
        <div className="dash-col dash-col--cluster">
          <ClusterCard
            pair={props.pair}
            fitsStandalone={standalone.gapBytes === 0}
            onServeStandalone={props.onServeStandalone}
            onCreate={props.onCreateCluster}
            onJoin={props.onJoinCluster}
            onManage={props.onManageCluster}
            onOpenSharing={props.onOpenSharing}
            onStartChat={props.onStartChat}
          />
        </div>
        <div className="dash-col">
          <div className="node-section__head">
            <span className="eyebrow">{t("node.thisNode")}</span>
            <span className="node-section__host">{s.hostname || "—"}</span>
            <span className="os">{s.os}</span>
          </div>
          <NodeCapacityCard
            snap={s}
            model={props.model}
            quant={props.quant}
            tier={props.tier}
            nNodes={nNodes}
            compact={props.pair?.phase === "ready"}
          />
          <EngineCard />
        </div>
      </div>
    </main>
  );
}

// ---- root -----------------------------------------------------------------
// ---- Weight download banner -----------------------------------------------
// The most fragile step of the supply-side funnel: the user installs, picks a
// model, presses start, and then waits for a file of 0.5 to 80 GiB. The three
// rules here come from what makes people give up:
//   1. Always show the **source** -- HF or a mirror. Silently switching over when
//      the direct connection fails makes it look stuck.
//   2. A failure must offer **a next step they can follow**, and say that what has
//      been downloaded is kept (otherwise users delete it and start over).
//   3. Cancel is a first-class citizen. A long task without a cancel button gets
//      cancelled by quitting the whole application.
function WeightsBanner(props: {
  dl: { have: number; total: number; endpoint?: string; note?: string; error?: string } | null;
  needsWeights: boolean;
  onCancel: () => void;
  onStart: () => void;
  onDismissError: () => void;
}) {
  const { t } = useI18n();
  const d = props.dl;
  if (d?.error) {
    return (
      <div className="wbanner wbanner--err" role="alert">
        <strong>{t("weights.failed")}</strong>
        <span className="wbanner__msg">{d.error}</span>
        <span className="wbanner__hint">{t("weights.resumeHint")}</span>
        <button className="btn-secondary" onClick={props.onStart}>{t("weights.retry")}</button>
        <button className="iconbtn" onClick={props.onDismissError} aria-label="dismiss">✕</button>
      </div>
    );
  }
  if (d) {
    const pctv = d.total > 0 ? Math.min(100, (d.have / d.total) * 100) : 0;
    return (
      <div className="wbanner" role="status">
        <strong>{t("weights.downloading")}</strong>
        <span className="wbanner__bar"><i style={{ width: `${pctv}%` }} /></span>
        <span className="wbanner__msg">
          {fmtBytes(d.have)}{d.total > 0 ? ` / ${fmtBytes(d.total)}` : ""}
          {d.endpoint ? ` · ${t("weights.from")} ${new URL(d.endpoint).host}` : ""}
        </span>
        {d.note ? <span className="wbanner__hint">{d.note}</span> : null}
        <button className="iconbtn" onClick={props.onCancel}>{t("weights.cancel")}</button>
      </div>
    );
  }
  if (props.needsWeights) {
    return (
      <div className="wbanner" role="status">
        <strong>{t("weights.needed")}</strong>
        {/* the ACTION, not the state — this button read "Downloading weights"
            while nothing was downloading, which is what it turns into. */}
        <button className="btn-secondary" onClick={props.onStart}>{t("weights.download")}</button>
      </div>
    );
  }
  return null;
}

export default function App() {
  const [theme, setTheme] = usePersisted<Theme>("idletoken.theme", "light");
  const [lang, setLang] = usePersisted<Lang>("idletoken.lang", "en");
  const [settings, setSettings] = useState<AppSettings>(() => loadSettings());
  // Automatic first-start model selection happens once: the probe re-runs when
  // the VRAM cap changes, and a re-run must not override the choice again.
  const firstRun = useRef(true);
  // Sidebar IA (2026-07 audit): chat / market / cluster / settings are stable
  // PLACES, not modal tasks. The active place persists across restarts.
  // Chat is the default place — talking to the cluster is the point; the
  // machines behind it are supporting detail. Before there is a cluster the
  // chat view is a one-click ramp into Cluster, so first run still lands right.
  const [view, setView] = usePersisted<View>("idletoken.view", "chat");
  // Deep link into a settings category ("share this cluster →" must land ON
  // the sharing page, not on whatever category was open last). null = keep
  // whatever the panel had; consumed by SettingsPanel on mount.
  const [settingsCategory, setSettingsCategory] = useState<string | null>(null);
  const openSettings = (category: string) => {
    setSettingsCategory(category);
    setView("settings");
  };
  // A prompt typed into the cluster card's quick box carries over into the
  // chat view and auto-sends there (one action, one home).
  const [chatInitial, setChatInitial] = useState<string | null>(null);
  const [session, setSession] = useState<Session | null>(() => getAuthProvider().currentSession());
  const [showAuth, setShowAuth] = useState(false);
  const [showPairing, setShowPairing] = useState(false);
  const [pairingView, setPairingView] = useState<"choose" | "join">("choose");
  // Single source of truth for cluster state: one subscription here feeds the
  // topbar pill AND the dashboard's cluster card (the cluster must be visible
  // without opening the pairing panel).
  const [pairSnap, setPairSnap] = useState<PairingSnapshot | null>(null);
  useEffect(() => getPairingProvider().subscribe(setPairSnap), []);
  const [snap, setSnap] = useState<NodeSnapshot | null>(null);
  const [totals, setTotals] = useState<{ vram_total: number; ram_total: number } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);

  const model = getModel(settings.modelId);

  // --- Getting the weights in place (B1/B2) --------------------------------
  // `weightsSource: "auto"` used to pass an empty string to the engine in every
  // case, and on the engine side an empty string takes the mock branch, which no
  // longer falls back automatically -- so a freshly installed client was certain
  // to fail loading once a model was picked, with nobody telling the user where to
  // get the weights. "auto" now has three real paths; see resolveLocalWeights in
  // weights.ts.
  const [weightsPath, setWeightsPath] = useState("");
  const [needsWeights, setNeedsWeights] = useState(false);
  const [dl, setDl] = useState<{
    have: number; total: number; endpoint?: string; note?: string; error?: string;
  } | null>(null);

  const refreshWeights = useCallback(async () => {
    try {
      const r = await resolveLocalWeights({
        weightsSource: settings.weightsSource === "local" ? "local" : "auto",
        ggufPath: settings.ggufPath,
        modelDir: settings.modelDir,
        manifest: getManifest(settings.modelId),
        quant: settings.quant,
      });
      setWeightsPath(r.path);
      setNeedsWeights(r.needsDownload);
    } catch {
      // A failed probe must not block the UI: treat it as "needs downloading", and
      // the user gets the real error when they press download.
      setWeightsPath("");
      setNeedsWeights(true);
    }
  }, [settings.weightsSource, settings.ggufPath, settings.modelDir, settings.modelId, settings.quant]);

  useEffect(() => { void refreshWeights(); }, [refreshWeights]);

  // Progress event subscription. Subscribed once for the whole application;
  // filtering by id is left for future multi-download support.
  useEffect(() => {
    let un: (() => void) | null = null;
    let dead = false;
    onFetchProgress((p) => {
      if (p.kind === "progress" || p.kind === "probe") {
        setDl((prev) => ({
          have: p.have ?? prev?.have ?? 0,
          total: p.total ?? prev?.total ?? 0,
          endpoint: p.endpoint ?? prev?.endpoint,
          note: p.note ?? prev?.note,
        }));
      } else if (p.kind === "done") {
        setDl(null);
        void refreshWeights();
      } else if (p.kind === "error") {
        setDl({ have: 0, total: 0, error: p.message ?? "download failed" });
      }
    }).then((f) => (dead ? f() : (un = f)));
    return () => { dead = true; un?.(); };
  }, [refreshWeights]);

  /** Ensure usable weights are present locally, downloading them when needed, and
   *  return the final path (an empty string means the cluster feeds the shards). */
  const ensureWeights = useCallback(async (): Promise<string> => {
    const r = await resolveLocalWeights({
      weightsSource: settings.weightsSource === "local" ? "local" : "auto",
      ggufPath: settings.ggufPath,
      modelDir: settings.modelDir,
      manifest: getManifest(settings.modelId),
      quant: settings.quant,
    });
    if (!r.needsDownload || !r.target) return r.path;
    const dir = settings.modelDir || (await defaultModelDir());
    setDl({ have: 0, total: r.target.expectBytes });
    await fetchWeights({ id: "primary", target: r.target, destDir: dir });
    const after = await resolveLocalWeights({
      weightsSource: "auto",
      ggufPath: "",
      modelDir: dir,
      manifest: getManifest(settings.modelId),
      quant: settings.quant,
    });
    return after.path;
  }, [settings.weightsSource, settings.ggufPath, settings.modelDir, settings.modelId, settings.quant]);

  // Preset caps derive from the machine's totals (learned from the first probe);
  // "custom" uses the precise sliders directly.
  const caps = useMemo(
    () => effectiveCaps(settings, totals),
    [settings.resourcePreset, settings.maxVramMb, settings.maxRamMb, totals]
  );

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

  // UI-test channel (see testHooks.ts): execute launcher-provided directives
  // through the same providers user actions use. No-op in normal runs.
  useEffect(() => {
    // StrictMode double-mounts effects in dev; directives must run once.
    if (uiTestRan) return;
    uiTestRan = true;
    uiTestDirectives().then((list) => {
      for (const d of list) {
        const m = d.match(/^engine-start:(worker|coordinator)$/);
        if (m) getEngineProvider().start(m[1] as EngineRole).catch(() => {});
        const q = d.match(/^quit:(\d+)$/); // close the window after N ms (graceful-exit tests)
        if (q)
          setTimeout(() => {
            import("@tauri-apps/api/window").then(({ getCurrentWindow }) => getCurrentWindow().close());
          }, Number(q[1]));

        // Pairing flow (P3/P4/P5/P6 gates). `:as=<name>` overrides the advertised
        // hostname so two instances on one test box get distinct roster ids;
        // `:apiPort=<n>` / `:apiToken=<s>` override the settings-derived engine
        // tuning (P5 settings gate); `:model=<path>` passes a real GGUF so the
        // engines load actual weights (P6 real-reply gate) instead of the mock
        // load (P3/P4). Token order is fixed: as, apiPort, apiToken, model.
        const pm = d.match(
          /^pairing-(create|join):([A-Z0-9]{6})(?::as=([^:,\s]+))?(?::apiPort=(\d+))?(?::apiToken=([^:,\s]+))?(?::model=(\S+))?$/
        );
        if (pm) {
          const [, op, code, as, apiPort, apiToken, model] = pm;
          const tuning = {
            ...engineTuning(settings),
            ...(apiPort ? { apiPort: Number(apiPort) } : {}),
            ...(apiToken ? { apiToken } : {}),
          };
          import("@tauri-apps/api/core").then(({ invoke }) =>
            invoke(op === "create" ? "pairing_create" : "pairing_join", {
              code,
              hostname: as || window.location.hostname || "test-node",
              gpu: "test",
              modelPath: model || "",
              tuning,
            }).catch((e) => console.error("uiTest pairing:", e))
          );
        }
        // Account-mode pairing (integration plan 3.3): derive the pair secret
        // from the real signed-in platform session — no fixed test material,
        // so an unauthenticated run reports the failure honestly instead of
        // faking a pairing. `:as=<name>` = distinct roster id per instance.
        const am = d.match(/^pairing-account-(create|join)(?::as=([^:,\s]+))?$/);
        if (am) {
          const [, op, as] = am;
          (async () => {
            const gate = platformGate();
            if (!gate.ok || !gate.session.userId) {
              reportTest("pairing-account", {
                ok: false,
                op,
                reason: gate.ok ? "session-missing-user-id" : gate.reason,
              });
              return;
            }
            const secret = await accountPairSecret(
              gate.session.userId,
              gate.url,
              loadSettings().clusterName || "home"
            );
            const { invoke } = await import("@tauri-apps/api/core");
            try {
              await invoke(op === "create" ? "pairing_create" : "pairing_join", {
                code: secret,
                hostname: as || window.location.hostname || "test-node",
                gpu: "test",
                modelPath: "",
                tuning: engineTuning(settings),
                account: true,
              });
              reportTest("pairing-account", { ok: true, op, email: gate.session.email });
            } catch (e) {
              reportTest("pairing-account", { ok: false, op, reason: String(e) });
            }
          })();
        }
        // Platform panel state report (integration plan 3.1/3.2 acceptance
        // hook): what the panel would render for the current auth/platform
        // config — connection gate, agent start params, Tauri availability.
        // Reads live settings/session (not the mount-time closure) and does
        // NOT start a real agent.
        if (d === "report-platform-panel") {
          const s = loadSettings();
          const sess = getAuthProvider().currentSession();
          const gate = platformGate();
          reportTest("platform", {
            platformUrl: s.platformUrl || null,
            signedIn: sess !== null,
            sessionProvider: sess?.provider ?? null,
            gate: gate.ok ? "ok" : gate.reason,
            agentName: s.clusterName || "home",
            coordApiPort: s.apiPort || 8000,
            tauri: inTauri(),
          });
        }
        // Self-check for the diagnostics bundle (support): produce a report
        // through the **real** provider path and assert two things -- that it
        // contains hardware facts, and that it contains **no token whatsoever**.
        // Redaction cannot be watched by eye; it needs a gate.
        const dg = d.match(/^report-diagnostics(?::token=([^,\s]+))?$/);
        if (dg) {
          // With :token=<sentinel>, write it into settings before producing the
          // report -- only then is "the report contains no token" a statement that
          // was actually verified. Without a sentinel, tokenChecked below honestly
          // records false.
          const original = loadSettings();
          if (dg[1]) saveSettings({ ...original, apiToken: dg[1] });
          const s = dg[1] ? { ...original, apiToken: dg[1] } : original;
          getEngineProvider()
            .diagnostics(undefined)
            .then((rep) => {
              const bundle = buildDiagnosticsBundle(rep, s);
              const r = bundle as any;
              // The redaction assertion is verified with a sentinel **known to be
              // present in settings**: with no token available to leak, "nothing
              // leaked" says nothing. tokenChecked records honestly whether this
              // cell was actually tested (not covered is not the same as passed).
              const token = s.apiToken || "";
              const text = JSON.stringify(bundle);
              reportTest("diagnostics", {
                schema: r.schema ?? null,
                probeGpu: r.probe?.gpu_name ?? null,
                probeError: r.probe?.error ?? null,
                hasAdvise: Boolean(r.advise && !r.advise.error),
                tokenChecked: token.length > 0,
                leaksApiToken: token.length > 0 && text.includes(token),
              });
            })
            .catch((e) => reportTest("diagnostics", { error: String(e) }))
            // The test sentinel token is not left behind in the user's settings.
            .finally(() => { if (dg[1]) saveSettings(original); });
        }
        // Report a fresh probe through the real provider path so acceptance
        // can diff it against `--probe-json` (P1: no fake data in the UI).
        // An executable oracle for weight downloading (B1). The product gate
        // cannot run it -- that needs a desktop capable of drawing -- while the
        // download itself needs no window. It runs a real download of the smallest
        // model (0.8B / 0.49 GiB): endpoint probing, Range resumption, size
        // validation and rename, all through the same production path.
        //   weights-fetch:<modelId>[:dir=<path>]
        const wf = d.match(/^weights-fetch:([a-z0-9.\-]+)(?::dir=(\S+))?$/);
        if (wf) {
          (async () => {
            const t0 = Date.now();
            try {
              const man = getManifest(wf[1]);
              const target = resolveDownload(man, defaultQuant(wf[1]));
              if (!target) { reportTest("weightsFetch", { error: "manifest has no repo/gguf" }); return; }
              const dir = wf[2] || (await defaultModelDir());
              await fetchWeights({ id: "uitest", target, destDir: dir });
              const st = await weightsState(dir, target.file, target.expectBytes);
              reportTest("weightsFetch", {
                model: wf[1], file: target.file, dir,
                complete: st.complete, haveBytes: st.have_bytes,
                expectBytes: target.expectBytes, seconds: Math.round((Date.now() - t0) / 1000),
              });
            } catch (e) {
              reportTest("weightsFetch", { model: wf[1], error: String(e) });
            }
          })();
        }
        if (d === "report-probe") {
          getResourceProvider()
            .probe({})
            .then((s) => reportTest("probe", s))
            .catch((e) => reportTest("probe", { error: String(e) }));
        }
        // Drive the auth lifecycle end to end (P2): fresh signup, signout,
        // wrong-password rejection, re-signin — all through the provider.
        if (d === "auth-flow") {
          (async () => {
            const em = `t${Date.now()}@test.local`;
            const pw = "test-pass-123";
            const auth = getAuthProvider();
            const r: Record<string, unknown> = {};
            try {
              await auth.signUp(em, pw);
              r.signup = "ok";
            } catch (e) {
              r.signup = String(e);
            }
            auth.signOut();
            r.signedOutNull = auth.currentSession() === null;
            try {
              await auth.signIn(em, "wrong-pass-000");
              r.wrongPwRejected = false;
            } catch {
              r.wrongPwRejected = true;
            }
            try {
              const s = await auth.signIn(em, pw);
              r.reSignin = s.email === em ? "ok" : "email-mismatch";
            } catch (e) {
              r.reSignin = String(e);
            }
            reportTest("auth", r);
          })();
        }
        if (d === "pairing-auto-start") {
          // creator: fire start exactly once, as soon as the roster allows it
          import("./pairing").then(({ getPairingProvider }) => {
            let fired = false;
            const un = getPairingProvider().subscribe((s) => {
              if (s.canStart && !fired) {
                fired = true;
                getPairingProvider().start().catch((e) => console.error("uiTest start:", e));
                un();
              }
            });
          });
        }
        // Record the live pairing:status trace from idle to ready (P4/P6). Proves
        // auto-orchestration progresses without any further user action and that
        // the API address comes online in the snapshot. Per-node layer display
        // needs matched hostnames (a real 2-machine LAN); here we assert the
        // orchestration + API-exposure signals achievable on one test box.
        if (d === "report-pairing-phases") {
          import("./pairing").then(({ getPairingProvider }) => {
            const phases: string[] = [];
            let reported = false;
            const un = getPairingProvider().subscribe((s) => {
              if (phases[phases.length - 1] !== s.phase) phases.push(s.phase);
              if (s.phase === "ready" && !reported) {
                reported = true;
                const coordPeer = s.peers.find((p) => p.role === "coordinator");
                reportTest("pairing-phases", {
                  phases,
                  sawStarting: phases.includes("starting"),
                  endedReady: s.phase === "ready",
                  apiBaseUrl: s.api?.baseUrl ?? null,
                  apiOnline: s.api?.status === "online",
                  coordinatorReady: coordPeer?.stage === "ready",
                  peersReady: s.peers.length > 0 && s.peers.every((p) => p.stage === "ready"),
                  // true only when the engine layer plan is attributed to a UI
                  // peer (needs a matched real hostname); informational here.
                  anyLayered: s.peers.some((p) => p.layerLo !== undefined && p.layerHi !== undefined),
                  peerStages: s.peers.map((p) => p.stage),
                });
                un();
              }
            });
          });
        }
      }
    });
  }, []);

  // Client-side UI settings that take effect immediately.
  useEffect(() => {
    const r = document.documentElement;
    r.setAttribute("data-accent", settings.accent);
    r.setAttribute("data-density", settings.density);
    if (settings.reduceMotion) r.setAttribute("data-reduce-motion", "");
    else r.removeAttribute("data-reduce-motion");
    (document.body.style as { zoom?: string }).zoom = String(settings.uiScale);
  }, [settings.accent, settings.density, settings.reduceMotion, settings.uiScale]);

  // Re-probe whenever the effective caps change so the dashboard reflects the
  // new limit immediately (the setting visibly takes effect). We keep the old
  // snapshot visible during a re-probe so only the first load shows a spinner.
  useEffect(() => {
    let live = true;
    setError(null);
    getResourceProvider()
      .probe({ maxVramMb: caps.maxVramMb, maxRamMb: caps.maxRamMb })
      .then((s) => {
        if (!live) return;
        setSnap(s);
        setTotals({ vram_total: s.vram_total, ram_total: s.ram_total });
        // First-start selection (B3): the default model used to be hardcoded to
        // DSv4-Flash (80.76 GiB), so the first thing a new user with an 8 GB card
        // saw was "does not fit on one machine, please build a cluster" -- with one
        // machine to their name. Given a real probe, recommend **the largest model
        // this machine can actually run** instead.
        // Only when settings have never been saved: a choice the user made
        // themselves must not be overwritten.
        if (firstRun.current && !settingsWerePersisted()) {
          firstRun.current = false;
          const ctx = (TIERS.find((x) => x.id === settings.tier) ?? TIERS[1]).ctx;
          const pick = pickBestFittingModel(s, ctx);
          if (pick.modelId !== settings.modelId || pick.quant !== settings.quant) {
            const next = { ...settings, modelId: pick.modelId, quant: pick.quant };
            setSettings(next);
            saveSettings(next);
            reportTest("firstRunModel", { picked: pick.modelId, quant: pick.quant });
          }
        }
      })
      .catch((e) => live && setError(String(e?.message ?? e)));
    return () => {
      live = false;
    };
  }, [nonce, caps.maxVramMb, caps.maxRamMb]);

  const updateSettings = (s: AppSettings) => {
    setSettings(s);
    saveSettings(s);
  };

  const cluster: ClusterState =
    pairSnap === null || pairSnap.phase === "idle"
      ? pairSnap && pairSnap.peers.length > 0
        ? "joining"
        : "standalone"
      : pairSnap.phase === "ready"
        ? "ready"
        : "joining";

  return (
    <LangContext.Provider value={{ lang, setLang }}>
      <div className={`app${dl || needsWeights ? " app--wbanner" : ""}`}>
        <TopBar
          cluster={cluster}
          theme={theme}
          session={session}
          view={view}
          onView={setView}
          onToggleTheme={() => setTheme(theme === "dark" ? "light" : "dark")}
          onToggleLang={() => setLang(lang === "en" ? "zh" : "en")}
          onGoCluster={() => setView("cluster")}
          onSignIn={() => setShowAuth(true)}
          onSignOut={() => {
            getAuthProvider().signOut();
            setSession(null);
          }}
        />
        <div className="app__body app__body--topnav">
          <div className="app__content">
            {error ? (
              <div className="center-state">
                <div className="msg">{t2("state.errorTitle", lang)}</div>
                <div className="err">{error}</div>
                <button className="iconbtn" onClick={() => setNonce((n) => n + 1)}>
                  {t2("state.retry", lang)}
                </button>
              </div>
            ) : !snap ? (
              <div className="center-state">
                <div className="spinner" />
                <div className="msg">{t2("state.loading", lang)}</div>
              </div>
            ) : view === "chat" ? (
              <Chat
                api={pairSnap?.api ?? null}
                apiToken={settings.apiToken}
                modelId={settings.modelId}
                initialPrompt={chatInitial}
                onInitialConsumed={() => setChatInitial(null)}
                onGoCluster={() => setView("cluster")}
              />
            ) : view === "settings" ? (
              <SettingsPanel
                asPage
                apiBaseUrl={pairSnap?.api?.status === "online" ? pairSnap.api.baseUrl : null}
                settings={settings}
                onChange={updateSettings}
                snap={snap}
                theme={theme}
                onTheme={setTheme}
                lang={lang}
                onLang={setLang}
                session={session}
                onSignIn={() => setShowAuth(true)}
                initialCategory={settingsCategory}
                onClose={() => setView("cluster")}
              />
            ) : (
              <Dashboard
                snap={snap}
                model={model}
                quant={settings.quant}
                tier={TIERS.find((x) => x.id === settings.tier) ?? TIERS[1]}
                pair={pairSnap}
                onServeStandalone={async () => {
                  // Single-node-first (small-model-design §4): the model+precision
                  // fits this one machine → serve immediately, no pairing. A solo
                  // create (num_workers=1, co-located worker) + start = a 1-node
                  // serving cluster; others can still be added later via Manage.
                  try {
                    // Standalone means this machine is the coordinator and nobody
                    // else can feed it shards, so a complete local copy of the
                    // weights is **required**. If it is missing, download it first
                    // (progress reaches the banner below through weights-fetch
                    // events). This step used to be absent: auto passed an empty
                    // string, the engine took the mock branch, mock no longer falls
                    // back, loading failed outright -- and all the user saw was
                    // "failed to start".
                    const path = await ensureWeights();
                    await getPairingProvider().create({
                      hostname: snap.hostname,
                      gpu: snap.gpu_name,
                      modelPath: path,
                      tuning: engineTuning(settings),
                    });
                    await getPairingProvider().start();
                  } catch (e) {
                    console.error("serve-standalone:", e);
                    setDl({ have: 0, total: 0, error: String(e) });
                  }
                }}
                onCreateCluster={() => {
                  setPairingView("choose");
                  setShowPairing(true);
                }}
                onJoinCluster={() => {
                  setPairingView("join");
                  setShowPairing(true);
                }}
                onManageCluster={() => {
                  setPairingView("choose");
                  setShowPairing(true);
                }}
                onOpenSharing={() => openSettings("platform")}
                onStartChat={(q) => {
                  setChatInitial(q);
                  setView("chat");
                }}
              />
            )}
          </div>
        </div>
      </div>
      {/* The weight download banner. It appears only when weights are genuinely
          missing or a download is running -- a permanent notice bar gets ignored. */}
      {dl || needsWeights ? (
        <WeightsBanner
          dl={dl}
          needsWeights={needsWeights}
          onCancel={() => void cancelFetch("primary")}
          onStart={() => void ensureWeights().catch((e) => setDl({ have: 0, total: 0, error: String(e) }))}
          onDismissError={() => setDl(null)}
        />
      ) : null}
      {showAuth ? (
        <AuthScreen
          onAuthed={(s) => {
            setSession(s);
            setShowAuth(false);
          }}
          onClose={() => setShowAuth(false)}
        />
      ) : null}
      {showPairing && snap ? (
        <PairingPanel
          self={{
            hostname: snap.hostname,
            gpu: snap.gpu_name,
            modelPath: weightsPath,
            tuning: engineTuning(settings),
          }}
          session={session}
          initialView={pairingView}
          onSignIn={() => {
            setShowPairing(false);
            setShowAuth(true);
          }}
          onClose={() => setShowPairing(false)}
        />
      ) : null}
    </LangContext.Provider>
  );
}

// Loading/error strings render in App itself, ABOVE the LangContext provider
// — a tiny direct lookup keeps them translated without hook gymnastics.
function t2(key: keyof (typeof STRINGS)["en"], lang: Lang): string {
  return (STRINGS[lang] as Record<string, string>)[key] ?? key;
}
