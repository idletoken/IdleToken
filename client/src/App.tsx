import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { LangContext, STRINGS, useI18n, type Lang } from "./i18n";
import { getResourceProvider } from "./provider";
import { getEngineProvider, type EngineLogLine, type EngineRole, type EngineStatus } from "./provider/engine";
import { uiTestDirectives } from "./testHooks";
import type { NodeSnapshot, ClusterState } from "./types";
import { HW_NO_GPU, HW_CC_TOO_LOW, HW_DRIVER_TOO_OLD, HW_VRAM_TOO_SMALL, HW_GPU_UNSUPPORTED, HW_MACOS_SEALED } from "./types";
import { getModel, getManifest, defaultQuant, estimateClusterCapacity, isSingleNode, isLocalGguf, localGgufSpec, LOCAL_GGUF_ID, pickBestFittingModel, type ModelSpec } from "./models";
import { resolveLocalWeights, resolveCustomWeights, customGgufName, fetchWeights, onFetchProgress, defaultModelDir, cancelFetch, resolveDownload, weightsState, isWeightsCancelled, WeightsCancelled, type CustomModelSource, type DownloadTarget } from "./weights";
import { loadSettings, saveSettings, settingsWerePersisted, effectiveCaps, engineTuning, autoUiScale, TIERS, type AppSettings } from "./settings";
import { buildDiagnosticsBundle } from "./diagnostics";
import { getAuthProvider, type Session } from "./auth";
import SettingsPanel from "./SettingsPanel";
import AuthScreen from "./AuthScreen";
import PairingPanel from "./PairingPanel";
import Chat from "./Chat";
import ModelPicker from "./ModelPicker";
import WeightsRow, { type WeightsInfo } from "./WeightsRow";
import { inTauri, platformGate, getMe } from "./platform";
import { identityFrom, type UserIdentity } from "./Avatar";
import { accountPairSecret, getPairingProvider, type PairingSnapshot, type ClusterApi } from "./pairing";
import { recordProblem } from "./problems";
import { useClusterStats, servedModelOf, type ClusterStats } from "./clusterStats";
import { fmtBytes, fmtGiB, pct } from "./format";
import UpdateDialog, { type UpdateResult } from "./UpdateDialog";
import { getUpdateProvider } from "./provider/update";
import { quitApp, setAutostart, syncTray, syncWindowPrefs, windowState } from "./system";

type Theme = "dark" | "light";

let uiTestRan = false;
// Identifies THIS JS context in UI-test reports. Diagnostic for the one class
// of double-execution the module guard above cannot stop: the webview loading
// the page twice (fresh module state each time). Two reports with different
// ids = two page loads, not a re-run within one.
const uiTestCtx = Math.random().toString(36).slice(2, 8);

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
      <button className="iconbtn iconbtn--signin" onClick={props.onSignIn}>
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
  // "pill.serving", not "cluster.ready": in single-machine mode there is no
  // cluster, and the pill's claim is about the service, not the topology.
  const clusterKey =
    props.cluster === "ready" ? "pill.serving" : props.cluster === "joining" ? "cluster.joining" : "cluster.standalone";
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
      <button className="iconbtn iconbtn--lang" onClick={props.onToggleLang} aria-label={t("lang.switch")}>
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
  // An open GGUF has no manifest, so there is nothing honest to estimate from:
  // the engine measures this machine at start and refuses with a reason when
  // the model does not fit (WS-B2). The card says exactly that instead of
  // rendering figures derived from another model's manifest.
  const open = isLocalGguf(props.model.id);
  const cap = open ? null : estimateClusterCapacity(props.model, s, props.tier.ctx, props.nNodes, props.quant);
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
    : s.hw_status === HW_GPU_UNSUPPORTED ? t("node.hw.gpuUnsupported")
    : s.hw_status === HW_MACOS_SEALED ? t("node.hw.macosSealed")
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

      {cap === null ? (
        <p className="capacity__text">{t("capacity.custom")}</p>
      ) : props.compact ? (
        <p className="capacity__text capacity__text--compact">
          {t("capacity.have", { have: GB(cap.haveBytes) })}
        </p>
      ) : (
      <div className="capacity">
        <div className="capacity__head">
          <span className="card__label">{t("capacity.title")}</span>
          <span className="capacity__need">
            {/* cap.nodes, not props.nNodes: a single-node model is sized for
                one machine however many are paired, and quoting the roster
                size here would describe a cluster the engine will not form. */}
            {props.model.singleNode
              ? t("capacity.need.single", { need: GB(cap.needBytes), model: props.model.label, tier: ctxLabel })
              : t("capacity.need", { need: GB(cap.needBytes), model: props.model.label, tier: ctxLabel, n: cap.nodes })}
          </span>
        </div>
        <div className="spine" role="img" aria-label={t("spine.caption", { n: cap.hostableLayers, total })}>
          {ticks.map((_, i) => (
            <span key={i} className={`tick${i < cap.hostableLayers ? " tick--on" : ""}`} />
          ))}
        </div>
        <div className="spine-scale">
          <span>{t("spine.layer0")}</span>
          <span>{total - 1}</span>
        </div>
        <p className="capacity__text">
          {t("capacity.have", { have: GB(cap.haveBytes) })}{" "}
          {cap.gapBytes > 0 ? (
            <span className="capacity__gap">
              {t(props.model.singleNode ? "capacity.gap.single" : "capacity.gap", { gap: GB(cap.gapBytes) })}
            </span>
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
  const { t, tErr } = useI18n();
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
        {/* A refusal is not a crash: the engine decided this machine will not
            join, and the supervisor did not retry. Showing the crash hint here
            ("kept crashing, exit code 2") would describe the wrong problem and
            hide the one sentence that says what to change. */}
        {st?.refusedReason ? (
          <div className="engine-meta engine-meta--bad">
            {/* tErr: client-authored refusals carry a code and localize;
                engine verbatim sentences pass through untouched. */}
            <strong>{t("engine.refused")}</strong> {tErr(st.refusedReason)}
          </div>
        ) : st?.state === "crashed" ? (
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

// ---- cluster activity row (engine GET /idletoken/v1/stats) ---------------------------
// "What did my cluster do" at a glance: served requests, total tokens, last
// decode speed, uptime. The poll itself lives in ./clusterStats — the chat page
// needs the served model out of the same endpoint.
function ActivityRow(props: { stats: ClusterStats | null }) {
  const { t } = useI18n();
  const stats = props.stats;
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
        <b>{(stats.input_tokens + stats.output_tokens).toLocaleString()}</b> {t("stats.tokens")}
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
  /** The selection is a user-supplied GGUF (open intake): the local option is
   *  the llama.cpp engine and the cluster options do not apply (WS-C). */
  openModel?: boolean;
  onServeStandalone?: () => void;
  // "Run it here" downloads the weights first when they are missing — 4.7 GB
  // for an 8B, 80 GB for DSv4. Without this the button looked broken: it really
  // had started a multi-GB download, and the only place that said so was a
  // different page. Whatever a button sets in motion has to report back next to
  // that button.
  weights?: WeightsInfo;
  onCreate: () => void;
  onJoin: () => void;
  onManage: () => void;
  onOpenSharing: () => void;
  // The LOCAL setting, used only to detect disagreement with what the cluster
  // reports it is serving. Never used as the displayed value.
  settingModelId: string;
  /** Display label for the setting — getModel() cannot name an open GGUF. */
  settingModelLabel: string;
  settingQuant: string;
  onOpenModelSetting: () => void;
  // Save a pick and rebuild whatever is running around it (App.switchModel).
  onSwitchModel: (modelId: string, quant: string) => void;
  /** Open-intake pick (local file / HF) — threaded into the picker. */
  onPickCustom: (c: CustomModelSource) => void;
  customName: string;
}) {
  const { t, tErr } = useI18n();
  const [copiedApi, setCopiedApi] = useState(false);
  const [copiedCode, setCopiedCode] = useState(false);
  const [pickOpen, setPickOpen] = useState(false);
  // A refused start ("need at least 2 machines", …) used to be an unhandled
  // rejection: the button did nothing on screen. Rendered under the button.
  const [startErr, setStartErr] = useState<string | null>(null);
  const snap = props.pair;
  // Before the early return below: hooks cannot be conditional.
  const stats = useClusterStats(snap?.api ?? null, snap?.source ?? "engine", {
    simModel: {
      id: props.settingModelId,
      label: props.settingModelLabel,
      quant: props.settingQuant,
    },
  });
  const active = snap !== null && snap.peers.length > 0;
  // Reported by the coordinator. Absent on an older engine -> show nothing;
  // substituting the local setting would answer a different question.
  const served = servedModelOf(stats);
  const anyError = active && snap.peers.some((p) => p.stage === "error");
  const fits = !!props.fitsStandalone;
  // The selected model decides whether "across several machines" is even a
  // path. Offering it for a single-node model would walk the user through
  // pairing, downloading and starting, only for the coordinator to refuse the
  // second worker — so the option stays visible (it explains itself) but its
  // buttons do not. An open GGUF is single-machine too, for now (WS-C wires
  // open models across machines).
  const clusterable = !props.openModel && !isSingleNode(props.settingModelId);

  if (!active) {
    return (
      // No pitch here (2026-08-10): whoever is looking at this screen already
      // installed the thing. The sell lives on the portal landing page; this
      // card only answers "what do I press now".
      <section className="card cluster-card cluster-card--empty">
        <h2 className="cluster-empty__title">{t("cluster.emptyTitle")}</h2>

        {/* WHICH model both options below are about. Every sentence on this
            card ("the selected model and precision fit on this machine") and
            every figure on the capacity card next to it is a consequence of
            this one choice, and it was the only thing on screen that never
            named it — the user had to open Settings to find out what "the
            selected model" currently is. Nothing is running yet, so unlike the
            served row further down this is the local setting, and it says so. */}
        <div className="cluster-model cluster-model--pick">
          <span className="cluster-model__label">{t("model.selected")}</span>
          <span className="cluster-model__name">{props.settingModelLabel}</span>
          {props.settingQuant ? <span className="cluster-model__quant">{props.settingQuant}</span> : null}
          {props.openModel ? <span className="cluster-model__quant">{t("model.open.badge")}</span> : null}
          {/* Nothing is running yet, so this pick is free: it writes the
              setting and the two options below re-read it. */}
          <button className="linkbtn cluster-model__change" onClick={() => setPickOpen((v) => !v)}>
            {t("model.change")}
          </button>
          {pickOpen ? (
            <ModelPicker
              modelId={props.settingModelId}
              quant={props.settingQuant}
              running={null}
              onPick={props.onSwitchModel}
              onPickCustom={props.onPickCustom}
              customName={props.customName}
              onClose={() => setPickOpen(false)}
            />
          ) : null}
        </div>

        {/* Two ways to deploy, always both on screen. They used to be one
            either/or row driven by fitsStandalone, which meant a machine big
            enough to go solo was never offered "join someone else's cluster",
            and a machine too small never saw the local option at all — the
            path you can't take should say why, not disappear. */}
        <div className="deploy-opt">
          <div className="deploy-opt__text">
            <h3 className="deploy-opt__title">{t("deploy.local")}</h3>
            {/* Both branches promise something about the OTHER option ("add
                machines later", "or pool machines below"), and neither is true
                for a single-node model — so each has a solo wording. */}
            <p className="deploy-opt__body">
              {props.openModel
                ? t("deploy.local.open")
                : fits
                  ? t(clusterable ? "deploy.local.fits" : "deploy.local.fitsSolo")
                  : t(clusterable ? "deploy.local.tooBig" : "deploy.local.tooBigSolo")}
            </p>
          </div>
          <button
            className={fits ? "btn-primary" : "btn-secondary"}
            // Also disabled while its own download runs — pressing it again
            // would start the sequence a second time with nothing to show for it.
            disabled={!fits || !!props.weights?.dl}
            onClick={props.onServeStandalone}
          >
            {props.weights?.dl ? t("weights.downloading") : t("cluster.serveLocal")}
          </button>
          {/* Progress for the download this button starts, plus anything the
              button does not already imply (a resumable partial, why the last
              attempt stopped). `idle=hide` drops the plain "no weights yet",
              which the button next to it already says. */}
          {props.weights ? <WeightsRow w={props.weights} idle="hide" /> : null}
        </div>

        <div className="deploy-opt">
          <div className="deploy-opt__text">
            <h3 className="deploy-opt__title">{t("deploy.cluster")}</h3>
            <p className="deploy-opt__body">
              {clusterable
                ? t("deploy.cluster.body")
                : props.openModel
                  ? t("deploy.cluster.openGguf")
                  : t("deploy.cluster.singleNode", { model: props.settingModelLabel })}
            </p>
            {!clusterable && (
              <button className="linkbtn" onClick={props.onOpenModelSetting}>
                {t("deploy.cluster.pickOther")}
              </button>
            )}
          </div>
          <div className="deploy-opt__actions">
            <button
              className={fits || !clusterable ? "btn-secondary" : "btn-primary"}
              disabled={!clusterable}
              onClick={props.onCreate}
            >
              {t("cluster.create")}
            </button>
            <button className="btn-secondary" disabled={!clusterable} onClick={props.onJoin}>
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

      {/* What is loaded, reported by the coordinator that loaded it — read only.
          Everything else on this card (the layer ranges, the capacity figures)
          is a consequence of this one fact, and it was the only one not shown. */}
      {served ? (
        <div className="cluster-model">
          <span className="cluster-model__label">{t("cluster.serving")}</span>
          <span className="cluster-model__name">{served.label}</span>
          {served.quant ? <span className="cluster-model__quant">{served.quant}</span> : null}
          {/* Switching from here is a cluster operation — it stops what is
              running and builds it again — so the row is still read-only until
              you ask, and the picker spells out the cost before it acts. */}
          <button className="linkbtn cluster-model__change" onClick={() => setPickOpen((v) => !v)}>
            {t("model.change")}
          </button>
          {pickOpen ? (
            <ModelPicker
              modelId={props.settingModelId}
              quant={props.settingQuant}
              running={{ modelId: served.id, quant: served.quant, machines: snap.peers.length }}
              onPick={props.onSwitchModel}
              onPickCustom={props.onPickCustom}
              customName={props.customName}
              onClose={() => setPickOpen(false)}
            />
          ) : null}
        </div>
      ) : null}
      {/* Inference-engine health (v2, llamacpp mode): mirrored from the
          coordinator's /health via stats. Chat answers 503 until "ready", so a
          quiet panel over a 503ing API would be a lie; absent on the legacy
          path, where per-peer stages carry the same news. */}
      {stats?.engine_state && stats.engine_state !== "ready" ? (
        <p
          className={`cluster-hint ${
            stats.engine_state === "failed" ? "cluster-hint--bad" : "cluster-hint--warn"
          }`}
        >
          {t(`cluster.engine.${stats.engine_state}` as const)}
          {(stats.engine_restarts ?? 0) > 0
            ? ` · ${t("cluster.engine.restarts", { n: stats.engine_restarts! })}`
            : ""}
        </p>
      ) : null}
      {/* No "setting disagrees with the cluster" hint here any more
          (2026-08-15): selecting a model IS the switch, everywhere — Settings
          included — so the disagreement the hint warned about can no longer be
          reached by picking; it only ever flickers mid-rebuild. One semantic,
          one path, nothing to warn about. */}

      <div className="cluster-peers">
        {snap.peers.map((p) => (
          <div key={p.id} className={`cpeer${p.online === false ? " cpeer--offline" : ""}`}>
            <span className={`cpeer__dot cpeer__dot--${p.stage}`} />
            <span className="cpeer__host">
              {p.hostname}
              {p.self ? <span className="cpeer__you"> · {t("pairing.you")}</span> : null}
              {p.online === false ? <span className="offline-tag">{t("pairing.offline")}</span> : null}
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

      {/* Joiner side: the CREATOR stopped answering. Every row is grayed by
          the same event, so the member-offline hint below would only repeat
          it — this one names the machine that matters. */}
      {snap.lastError?.code === "creatorLost" ? (
        <p className="cluster-hint cluster-hint--warn">{t("pairing.err.creatorLost")}</p>
      ) : null}

      {/* A member stopped answering while the cluster is up: the all-green
          card was a lie (audit 2.8). Point at the machine, not the cluster. */}
      {snap.lastError?.code !== "creatorLost" &&
      snap.phase !== "idle" &&
      snap.peers.some((p) => p.online === false) ? (
        <p className="cluster-hint cluster-hint--warn">{t("cluster.offlineHint")}</p>
      ) : null}

      {anyError ? <p className="cluster-hint cluster-hint--bad">{t("cluster.errorHint")}</p> : null}

      {snap.phase === "loading" ? <p className="cluster-hint">{t("cluster.loadingHint")}</p> : null}

      {snap.canStart ? (
        <button
          className="btn-primary btn-block cluster-start"
          onClick={() => {
            setStartErr(null);
            getPairingProvider()
              .start()
              .catch((e) => setStartErr(tErr(String(e))));
          }}
        >
          {t("pairing.startCluster", { n: snap.peers.length })} →
        </button>
      ) : null}
      {startErr ? <p className="cluster-hint cluster-hint--bad">{startErr}</p> : null}

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
          <ActivityRow stats={stats} />
        </div>
      ) : null}

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

// ---- local llama.cpp engine card (v2 WS-D1/D2) -----------------------------
// Shown in the cluster column while THIS machine serves a user-supplied GGUF
// through the coordinator's llamacpp single-machine mode. Three truths it must
// carry, all from the engine rather than guessed client-side:
//   - the scheduler's verdict (the "fits / how" sentence the coordinator
//     prints at start) and the auto-manifest line (what the GGUF header says);
//   - live engine health from /idletoken/v1/stats (engine_state mirrors
//     /health; chat 503s until "ready" — no fake green);
//   - a refusal, verbatim (exit 3 + worded stderr, latched by the supervisor).
function LocalEngineCard(props: {
  label: string; // the file the user picked
  api: ClusterApi | null;
  engStatus: EngineStatus | null;
  settingModelId: string;
  settingQuant: string;
  customName: string;
  onSwitchModel: (modelId: string, quant: string) => void;
  onPickCustom: (c: CustomModelSource) => void;
  onStop: () => void;
  /** Polled app-level (App owns one poll shared with the topbar pill — the
   *  coordinator serves requests serially, so pollers are not free). */
  stats: ClusterStats | null;
  weights?: WeightsInfo;
}) {
  const { t, tErr } = useI18n();
  const [pickOpen, setPickOpen] = useState(false);
  const [copiedApi, setCopiedApi] = useState(false);
  // The coordinator's startup sentences, latched from the supervisor's log
  // stream. Scraping a log is deliberate: the verdict is printed once at
  // start, exists nowhere else, and IS the scheduler's answer (B2 contract).
  const [verdict, setVerdict] = useState<string | null>(null);
  const [manifest, setManifest] = useState<string | null>(null);
  useEffect(() => {
    const latch = (l: EngineLogLine) => {
      const v = l.line.indexOf("coord: scheduler: ");
      if (v >= 0) setVerdict(l.line.slice(v + "coord: scheduler: ".length));
      const m = l.line.indexOf("coord: auto manifest: ");
      if (m >= 0) setManifest(l.line.slice(m + "coord: auto manifest: ".length));
    };
    const eng = getEngineProvider();
    eng.logs(100).then((ls) => ls.forEach(latch)).catch(() => {});
    return eng.onLog(latch);
  }, []);

  const stats = props.stats;
  const refused = props.engStatus?.refusedReason ?? null;
  // Engine health, most specific source first: the coordinator's own
  // engine_state once the API answers, the supervisor's process state before.
  const es = stats?.engine_state ?? null;
  const stateKey = refused
    ? ("local.state.failed" as const)
    : es
      ? (`local.state.${es}` as const)
      : props.engStatus?.state === "crashed"
        ? ("local.state.failed" as const)
        : props.engStatus?.state === "restarting"
          ? ("local.state.restarting" as const)
          : ("local.starting" as const);
  const tone = refused || es === "failed" || props.engStatus?.state === "crashed"
    ? "bad"
    : es === "ready"
      ? "ready"
      : "busy";

  const copyApi = async () => {
    if (!props.api) return;
    try {
      await navigator.clipboard.writeText(props.api.baseUrl);
      setCopiedApi(true);
      setTimeout(() => setCopiedApi(false), 1500);
    } catch {
      /* shown regardless */
    }
  };

  return (
    <section className="card cluster-card">
      <div className="cluster-head">
        <span className="card__label">{t("local.title")}</span>
        <span className={`stage-tag stage-tag--${tone}`}>
          <span className="stage-tag__dot" />
          {t(stateKey)}
        </span>
        {(stats?.engine_restarts ?? 0) > 0 ? (
          <span className="cluster-head__count">{t("local.restarts", { n: stats!.engine_restarts! })}</span>
        ) : null}
      </div>

      <div className="cluster-model">
        <span className="cluster-model__label">{t("cluster.serving")}</span>
        {/* The engine's own id once it has read the GGUF header; the picked
            file name until then. */}
        <span className="cluster-model__name">{stats?.model_label || stats?.model || props.label}</span>
        <span className="cluster-model__quant">{t("model.open.badge")}</span>
        <button className="linkbtn cluster-model__change" onClick={() => setPickOpen((v) => !v)}>
          {t("model.change")}
        </button>
        {pickOpen ? (
          <ModelPicker
            modelId={props.settingModelId}
            quant={props.settingQuant}
            running={{ modelId: props.settingModelId, quant: "", machines: 1 }}
            onPick={props.onSwitchModel}
            onPickCustom={props.onPickCustom}
            customName={props.customName}
            onClose={() => setPickOpen(false)}
          />
        ) : null}
      </div>

      {/* Download progress for an HF-sourced GGUF lives next to the card that
          started it, like every other weight download. */}
      {props.weights ? <WeightsRow w={props.weights} idle="hide" /> : null}

      {refused ? (
        <div className="hw-blocked" role="alert">
          <strong>{t("local.refusedTitle")}</strong>
          <span className="hw-blocked__detail">{tErr(refused)}</span>
        </div>
      ) : null}

      {es === "failed" ? <p className="cluster-hint cluster-hint--bad">{t("cluster.engine.failed")}</p> : null}
      {es === "starting" ? <p className="cluster-hint">{t("cluster.engine.starting")}</p> : null}
      {es === "restarting" ? <p className="cluster-hint cluster-hint--warn">{t("cluster.engine.restarting")}</p> : null}

      {verdict ? (
        <p className="cluster-hint local-verdict">
          <b>{t("local.verdict")}:</b> {verdict}
        </p>
      ) : null}
      {manifest ? <p className="cluster-hint local-manifest">{manifest}</p> : null}

      {props.api && !refused ? (
        <div className="cluster-api">
          <div className="cluster-api__label">
            {t("cluster.api")}
            <span
              className={`stage-tag stage-tag--${
                props.api.status === "online" ? "ready" : props.api.status === "offline" ? "bad" : "busy"
              }`}
            >
              <span className="stage-tag__dot" />
              {props.api.status === "online"
                ? t("pairing.api.online")
                : props.api.status === "offline"
                  ? t("cluster.apiOffline")
                  : t("pairing.api.starting")}
            </span>
          </div>
          <div className="cluster-api__row">
            <code className="cluster-api__url">{props.api.baseUrl}</code>
            <button className="iconbtn" onClick={copyApi}>
              {copiedApi ? t("pairing.copied") : t("pairing.copy")}
            </button>
          </div>
          <p className="cluster-api__hint">{t("cluster.apiHint")}</p>
          <ActivityRow stats={stats} />
        </div>
      ) : null}

      <p className="cluster-hint">{t("local.note")}</p>

      <div className="cluster-share">
        <button className="linkbtn" onClick={props.onStop}>
          {t("local.stop")}
        </button>
      </div>
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
  /** Local llama.cpp engine (open-GGUF serving) — replaces the cluster card
   *  while it runs; this machine IS the whole deployment. */
  localEngine: { gguf: string; label: string } | null;
  localApi: ClusterApi | null;
  localStats: ClusterStats | null;
  engStatus: EngineStatus | null;
  customName: string;
  onStopLocal: () => void;
  onPickCustom: (c: CustomModelSource) => void;
  onServeStandalone: () => void;
  onCreateCluster: () => void;
  onJoinCluster: () => void;
  onManageCluster: () => void;
  onOpenSharing: () => void;
  /** Jump to the full model section in Settings (the cluster card's picker
   *  covers the common case; Settings still owns weights paths and the rest). */
  onOpenModelSetting: () => void;
  onSwitchModel: (modelId: string, quant: string) => void;
  weights?: WeightsInfo;
}) {
  const { t, tErr } = useI18n();
  const s = props.snap;
  const nNodes = props.pair && props.pair.peers.length > 0 ? props.pair.peers.length : 3;
  const open = isLocalGguf(props.model.id);
  // Single-node-first: does the selected model+precision fit THIS machine alone
  // (N=1)? If so the empty-state leads with "serve locally" instead of pairing.
  // The shortfall goes down with it — the local row reports how far off it is
  // rather than just going quiet. An open GGUF has no client-side estimate:
  // the button stays live and the ENGINE rules on fit at start (worded
  // refusal on the local-engine card), per hard invariant "no client guess
  // overrides the scheduler".
  const standalone = open ? null : estimateClusterCapacity(props.model, s, props.tier.ctx, 1, props.quant);
  // The generic refusal surface (D2): whatever sentence the engine sent
  // through the JOIN_REFUSED / exit-3 channel, verbatim, where the user is
  // looking. WS-C's "upgrade machine X" (version mismatch) arrives through
  // the same channel and needs no client change. The local-engine card
  // carries its own copy, so this banner covers the cluster/legacy paths.
  const refused = !props.localEngine ? props.engStatus?.refusedReason ?? null : null;
  return (
    <main className="main main--wide">
      {s.source === "dev-fixture" ? <FixtureBanner /> : null}
      <div className="dash-grid">
        <div className="dash-col dash-col--cluster">
          {refused ? (
            <div className="hw-blocked" role="alert">
              <strong>{t("cluster.refusedTitle")}</strong>
              <span className="hw-blocked__detail">{tErr(refused)}</span>
            </div>
          ) : null}
          {props.localEngine ? (
            <LocalEngineCard
              label={props.localEngine.label}
              api={props.localApi}
              stats={props.localStats}
              engStatus={props.engStatus}
              settingModelId={props.model.id}
              settingQuant={props.quant}
              customName={props.customName}
              onSwitchModel={props.onSwitchModel}
              onPickCustom={props.onPickCustom}
              onStop={props.onStopLocal}
              weights={props.weights}
            />
          ) : (
          <ClusterCard
            pair={props.pair}
            fitsStandalone={standalone === null ? true : standalone.gapBytes === 0}
            openModel={open}
            onServeStandalone={props.onServeStandalone}
            weights={props.weights}
            onCreate={props.onCreateCluster}
            onJoin={props.onJoinCluster}
            onManage={props.onManageCluster}
            onOpenSharing={props.onOpenSharing}
            settingModelId={props.model.id}
            settingModelLabel={props.model.label}
            settingQuant={props.quant}
            onOpenModelSetting={props.onOpenModelSetting}
            onSwitchModel={props.onSwitchModel}
            onPickCustom={props.onPickCustom}
            customName={props.customName}
          />
          )}
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
  const [session, setSession] = useState<Session | null>(() => getAuthProvider().currentSession());
  // The signed-in person's public identity (display name + avatar colour), so
  // the chat can show the user as the account they configured on the platform
  // rather than a generic silhouette. Cloud sessions only: a local identity has
  // no profile to read, and a failed fetch simply leaves the neutral glyph —
  // never block or nag over an avatar.
  const [identity, setIdentity] = useState<UserIdentity | null>(null);
  useEffect(() => {
    if (!session || session.provider !== "cloud") { setIdentity(null); return; }
    let live = true;
    void getMe()
      .then((me) => { if (live) setIdentity(identityFrom(me)); })
      .catch(() => { if (live) setIdentity(null); });
    return () => { live = false; };
  }, [session]);
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

  // ---- open model intake + local llama.cpp engine (v2 WS-D1) ---------------
  // The selection may be a user-supplied GGUF (LOCAL_GGUF_ID sentinel): then
  // there is no manifest, the display spec is synthesized from the file name,
  // and serving goes through the coordinator's llamacpp single-machine mode
  // instead of the pairing path.
  const customSel: CustomModelSource | null = isLocalGguf(settings.modelId)
    ? {
        source: settings.customSource,
        path: settings.customGgufPath,
        repo: settings.customHfRepo,
        file: settings.customHfFile,
      }
    : null;
  const customName = customSel ? customGgufName(customSel) : "";
  const model = customSel ? localGgufSpec(customName) : getModel(settings.modelId);

  // The local llama.cpp engine this client started (llamacpp_serve). Not
  // persisted: the sidecar dies with the client, so a fresh launch starts
  // clean. `label` is what the user picked; the AUTO id the coordinator read
  // from the GGUF header arrives later via /idletoken/v1/stats.
  const [localEngine, setLocalEngine] = useState<{ gguf: string; label: string } | null>(null);

  // Aggregate engine-supervisor status, held app-level: the local-engine card,
  // the chat gating and the generic refusal banner all read it. (EngineCard
  // keeps its own subscription — it also wants the log tail.)
  const [engStatus, setEngStatus] = useState<EngineStatus | null>(null);
  useEffect(() => {
    const eng = getEngineProvider();
    let live = true;
    eng.status().then((s) => live && setEngStatus(s)).catch(() => {});
    const un = eng.onStatus(setEngStatus);
    return () => {
      live = false;
      un();
    };
  }, []);

  // --- Getting the weights in place (B1/B2) --------------------------------
  // The client used to pass an empty string to the engine in every case, and on
  // the engine side an empty string takes the mock branch, which no longer falls
  // back automatically -- so a freshly installed client was certain to fail
  // loading once a model was picked, with nobody telling the user where to get
  // the weights. Resolution now has three real paths; see resolveLocalWeights in
  // weights.ts.
  const [weightsPath, setWeightsPath] = useState("");
  const [needsWeights, setNeedsWeights] = useState(false);
  // Bytes of an unfinished copy on disk. The download resumes from it, so this
  // is the difference between "4.7 GB to fetch" and "600 MB to go".
  const [partialBytes, setPartialBytes] = useState(0);
  const [dl, setDl] = useState<{
    have: number; total: number; note?: string;
  } | null>(null);
  /**
   * Why the last attempt stopped — a footnote, never a state.
   *
   * A download that did not finish leaves the machine in exactly the state it
   * was in before: no weights. That is what the row now says, with the button
   * that fixes it, because "Download failed" as a state of its own is a dead
   * end the user has to clear before the normal affordance comes back — and it
   * is not even true after a resume, which starts from what is already there.
   * The reason still ships, quietly, so a download that keeps dying is not a
   * button that silently does nothing.
   */
  const [lastError, setLastError] = useState<string | null>(null);

  const refreshWeights = useCallback(async () => {
    try {
      const r = isLocalGguf(settings.modelId)
        ? await resolveCustomWeights({
            modelDir: settings.modelDir,
            custom: {
              source: settings.customSource,
              path: settings.customGgufPath,
              repo: settings.customHfRepo,
              file: settings.customHfFile,
            },
          })
        : await resolveLocalWeights({
            modelDir: settings.modelDir,
            manifest: getManifest(settings.modelId),
            quant: settings.quant,
          });
      setWeightsPath(r.path);
      setNeedsWeights(r.needsDownload);
      setPartialBytes(r.haveBytes);
    } catch {
      // A failed probe must not block the UI: treat it as "needs downloading", and
      // the user gets the real error when they press download.
      setWeightsPath("");
      setNeedsWeights(true);
      setPartialBytes(0);
    }
  }, [
    settings.modelDir,
    settings.modelId,
    settings.quant,
    settings.customSource,
    settings.customGgufPath,
    settings.customHfRepo,
    settings.customHfFile,
  ]);

  useEffect(() => { void refreshWeights(); }, [refreshWeights]);

  // Ids the user has cancelled. The engine only notices the cancel flag between
  // reads, and a read blocks until the next chunk arrives — on a slow endpoint
  // that is seconds away. Waiting for it meant the row sat there showing
  // progress after the user pressed Cancel, then flashed a message. The click
  // is the decision; the UI acts on it now and drops whatever that download
  // says afterwards.
  const cancelled = useRef<Set<string>>(new Set());

  /** The in-flight `primary` fetch, if any. There is one download slot, and
   *  the engine refuses a second writer on the same id (two writers on one
   *  .part once blessed a corrupt file as complete — see weights_fetch), so
   *  everything that starts or abandons a download coordinates through this. */
  const fetchInFlight = useRef<Promise<void> | null>(null);

  /** What a download "belongs to": one selection, one key. `dl` is a single
   *  app-wide value and the row renders wherever the selection is — so without
   *  an owner, whatever download happens to be running paints the row of
   *  whatever model happens to be selected (this is exactly how "I picked B
   *  and it showed A's progress" looked). */
  const ownerKeyOf = useCallback(
    (modelId: string, quant: string, custom?: CustomModelSource) =>
      isLocalGguf(modelId) && custom
        ? `custom:${custom.source}:${custom.source === "file" ? custom.path : `${custom.repo}/${custom.file}`}`
        : `${modelId}:${quant}`,
    []
  );
  /** Owner of the current/last `primary` download. */
  const dlOwner = useRef<string | null>(null);
  /** Owner key of the selection on screen, readable from event callbacks. */
  const curSelKey = ownerKeyOf(settings.modelId, settings.quant, {
    source: settings.customSource,
    path: settings.customGgufPath,
    repo: settings.customHfRepo,
    file: settings.customHfFile,
  });
  const curSelRef = useRef(curSelKey);
  useEffect(() => {
    curSelRef.current = curSelKey;
  });

  const cancelDownload = useCallback(() => {
    cancelled.current.add("primary");
    setDl(null);
    void cancelFetch("primary");
    void refreshWeights();
  }, [refreshWeights]);

  /**
   * Show a failed download — or say nothing, when the "failure" is the user's
   * own Cancel.
   *
   * `weights_fetch` rejects for a cancel too (correctly: there are no usable
   * weights and the caller must not continue), and every catch here used to
   * turn that rejection into a red row. It appeared SECONDS after the click,
   * because the download only notices the flag when its blocked read returns —
   * long enough that it read as an unrelated failure. And since `dl` is one
   * app-wide value with no model attached, the stale error then followed the
   * user from model to model, which is what "all of them say download failed"
   * was.
   */
  const reportWeightsError = useCallback((e: unknown, kind: "download" | "cluster" = "download") => {
    setDl(null);
    // A cancel leaves no footnote at all: the user knows why it stopped, they
    // stopped it.
    if (isWeightsCancelled(e)) {
      setLastError(null);
    } else {
      const msg = String(e);
      setLastError(msg);
      // The footnote is cleared by the next attempt or a model switch; the
      // problem log is what remains for "this keeps failing, here is what it
      // says" (Settings -> Data & about).
      recordProblem({
        at: new Date().toISOString(),
        kind,
        message: msg,
        detail: { model: settings.modelId, quant: settings.quant },
      });
    }
    void refreshWeights();
  }, [refreshWeights, settings.modelId, settings.quant]);

  // What the download row is showing right now, readable from the UI-test
  // directives — those run in a mount-time effect and would otherwise see the
  // first render's value forever.
  const errRef = useRef(lastError);
  useEffect(() => {
    errRef.current = lastError;
  });

  // A download report belongs to the model that was being downloaded. Switching
  // model (or precision) makes it answer a question nobody asked — clear it, and
  // let refreshWeights say what THIS model's state is.
  //
  // The download itself is abandoned too, not just its report. Clearing alone
  // was not enough: the old model's fetch kept running, and its next progress
  // event resurrected the cleared row — so a freshly selected model whose
  // weights were already complete on disk still said "downloading" until the
  // abandoned download ended. The cancel keeps the `.part`, so switching back
  // resumes from where it stopped.
  useEffect(() => {
    if (fetchInFlight.current) {
      cancelled.current.add("primary");
      void cancelFetch("primary");
    }
    setDl(null);
    setLastError(null);
  }, [
    settings.modelId,
    settings.quant,
    settings.customSource,
    settings.customGgufPath,
    settings.customHfRepo,
    settings.customHfFile,
  ]);

  // Progress event subscription. Subscribed once for the whole application;
  // only the `primary` slot reaches the row (see below).
  useEffect(() => {
    let un: (() => void) | null = null;
    let dead = false;
    onFetchProgress((p) => {
      const id = p.id || "primary";
      // Only the `primary` slot drives the row. Other ids run real downloads
      // too (the UI-test oracles), and their progress must not paint the row
      // of whatever model happens to be selected.
      if (id !== "primary") return;
      if (cancelled.current.has(id)) {
        // The download is winding down. Only its final word clears the mark —
        // anything before that (a last progress tick) is stale by definition.
        if (p.kind === "done" || p.kind === "error" || p.kind === "cancelled") {
          cancelled.current.delete(id);
          void refreshWeights();
        }
        return;
      }
      // Ownership: the download knows which selection started it, and its
      // progress may only paint the row when that selection is still the one
      // on screen. Cancel-on-switch already stops the abandoned download; this
      // is the second lock, for any event that arrives inside that window (or
      // any path that forgot to cancel).
      const owned = dlOwner.current === curSelRef.current;
      if (p.kind === "progress" || p.kind === "probe") {
        if (!owned) return;
        // The event still says which endpoint is serving the bytes, but that
        // stays out of the row on purpose: where the download comes from is an
        // implementation detail the user is not asked to think about.
        setDl((prev) => ({
          have: p.have ?? prev?.have ?? 0,
          total: p.total ?? prev?.total ?? 0,
          note: p.note ?? prev?.note,
        }));
      } else if (p.kind === "done") {
        setDl(null);
        void refreshWeights();
      } else if (p.kind === "cancelled") {
        // Cancelled without going through our button (older shell, or a second
        // window): still not an error — back to "not downloaded", resume ready.
        setDl(null);
        void refreshWeights();
      } else if (p.kind === "error") {
        setDl(null);
        // The footnote goes on the row the failure belongs to; a model the
        // user already left keeps its story out of the new row. The problem
        // log still gets it through the command-rejection path.
        if (owned) setLastError(p.message ?? "download failed");
        void refreshWeights();
      }
    }).then((f) => (dead ? f() : (un = f)));
    return () => { dead = true; un?.(); };
  }, [refreshWeights]);

  /**
   * Start (or take over) the one `primary` download. An older fetch still in
   * flight — the user abandoned it by switching models, or pressed Download
   * again — is cancelled and awaited out first, so the new one does not bounce
   * off the engine's one-writer-per-id guard with "a download is already
   * running". Its `.part` survives the cancel; a later return to that model
   * resumes from it.
   */
  const runPrimaryFetch = useCallback(async (target: DownloadTarget, destDir: string, owner: string) => {
    dlOwner.current = owner;
    setLastError(null); // this attempt speaks for itself
    // Optimistic row, but only on the row it belongs to — the caller may be a
    // switch flow whose selection has already moved on again.
    if (curSelRef.current === owner) setDl({ have: 0, total: target.expectBytes });
    const prev = fetchInFlight.current;
    if (prev) {
      cancelled.current.add("primary");
      void cancelFetch("primary");
      await prev.catch(() => {});
      // The wound-down fetch's final event normally clears the mark; delete it
      // again in case that event was already processed before the mark landed,
      // or the new download's own progress would be swallowed with it.
      cancelled.current.delete("primary");
    }
    const p = fetchWeights({ id: "primary", target, destDir });
    fetchInFlight.current = p;
    try {
      await p;
    } catch (e) {
      // A failure of a download the user has already walked away from (the
      // selection changed while it ran, which also requested its cancel) is
      // not the on-screen model's story — classify it with the cancels so no
      // caller paints it as that model's failure.
      if (!isWeightsCancelled(e) && dlOwner.current === owner && owner !== curSelRef.current) {
        throw new WeightsCancelled(String(e));
      }
      throw e;
    } finally {
      if (fetchInFlight.current === p) fetchInFlight.current = null;
    }
  }, []);

  /** Ensure usable weights are present locally, downloading them when needed, and
   *  return the final path (an empty string means the cluster feeds the shards). */
  // Preset caps derive from the machine's totals (learned from the first probe);
  // "custom" uses the precise sliders directly. Declared HERE, above the launch
  // paths, because every engine this client starts has to be told the cap —
  // see engineTuning(). It used to sit below them, which is why it could only
  // ever reach the probe.
  const caps = useMemo(
    () => effectiveCaps(settings, totals),
    [settings.resourcePreset, settings.maxVramMb, settings.maxRamMb, totals]
  );

  /**
   * Resolve (downloading if necessary) the weights for the selected model.
   *
   * `over` overrides the model/precision for THIS call. switchModel needs it:
   * it saves the new selection and immediately acts on it, and `settings` in
   * this closure is still the old value until React re-renders — without the
   * override the switch would faithfully download and start the model the user
   * just moved away from.
   */
  const ensureWeights = useCallback(async (over?: { modelId: string; quant: string }): Promise<string> => {
    const modelId = over?.modelId ?? settings.modelId;
    const quant = over?.quant ?? settings.quant;
    const r = await resolveLocalWeights({
      modelDir: settings.modelDir,
      manifest: getManifest(modelId),
      quant,
    });
    if (!r.needsDownload || !r.target) return r.path;
    const dir = settings.modelDir || (await defaultModelDir());
    await runPrimaryFetch(r.target, dir, ownerKeyOf(modelId, quant));
    const after = await resolveLocalWeights({
      modelDir: dir,
      manifest: getManifest(modelId),
      quant,
    });
    return after.path;
  }, [settings.modelDir, settings.modelId, settings.quant, runPrimaryFetch, ownerKeyOf]);

  /** Resolve (downloading if necessary) a user-supplied GGUF, returning the
   *  local path the engine will load. HF picks reuse the exact weights
   *  machinery curated models use (weights_fetch is manifest-agnostic);
   *  local-file picks are only re-verified — a moved file is a one-sentence
   *  answer here instead of a load failure later. */
  const ensureCustomWeights = useCallback(
    async (c: CustomModelSource): Promise<string> => {
      const r = await resolveCustomWeights({ modelDir: settings.modelDir, custom: c });
      if (r.path) return r.path;
      if (!r.needsDownload || !r.target) {
        throw new Error(`GGUF file not found: ${c.source === "file" ? c.path : `${c.repo}/${c.file}`}`);
      }
      const dir = settings.modelDir || (await defaultModelDir());
      await runPrimaryFetch(r.target, dir, ownerKeyOf(LOCAL_GGUF_ID, "", c));
      const after = await resolveCustomWeights({ modelDir: dir, custom: c });
      if (!after.path) throw new Error("download finished but the file did not verify — try again");
      return after.path;
    },
    [settings.modelDir, runPrimaryFetch, ownerKeyOf]
  );

  /**
   * Serve a user-supplied GGUF on this machine: coordinator in llamacpp
   * single-machine mode (WS-D1). The scheduler verdict — fits / refuses with a
   * worded reason, exit 3 — comes back through the engine supervisor's
   * log/refusal channels and renders on the local-engine card.
   *
   * ctxSize 0 on purpose: the coordinator sizes the context from measured
   * memory (32K ask, 16K floor, worded refusal below it). Passing the legacy
   * tier here would turn "it fits at 16K" into a refusal at 128K nobody asked
   * for.
   *
   * The "this machine's usage" caps ride along as --max-vram-mb/--max-ram-mb
   * (0 = uncapped): the coordinator probes the machine itself, then caps what
   * the probe reports before planning — same contract as the worker flag.
   */
  const serveOpenModel = useCallback(
    async (over?: CustomModelSource): Promise<void> => {
      const sel = over ?? customSel;
      if (!sel) return;
      const label = customGgufName(sel);
      try {
        if (!inTauri()) {
          // Browser dev build: no engine to start; show the card in fixture mode.
          setLocalEngine({ gguf: sel.path || sel.file, label });
          setView("cluster");
          return;
        }
        const path = await ensureCustomWeights(sel);
        const { invoke } = await import("@tauri-apps/api/core");
        await invoke("llamacpp_serve", {
          ggufPath: path,
          apiHost: settings.apiHost || "0.0.0.0",
          apiPort: settings.apiPort || 8000,
          apiToken: settings.apiToken,
          ctxSize: 0,
          maxDecode: settings.maxTokens,
          maxVramMb: caps.maxVramMb,
          maxRamMb: caps.maxRamMb,
        });
        setLocalEngine({ gguf: path, label });
        setView("cluster");
      } catch (e) {
        if (!isWeightsCancelled(e)) console.error("serve-open-model:", e);
        reportWeightsError(e, "cluster");
      }
    },
    [customSel, ensureCustomWeights, settings.apiHost, settings.apiPort, settings.apiToken, settings.maxTokens, caps, reportWeightsError]
  );

  const stopLocalEngine = useCallback(async () => {
    try {
      await getEngineProvider().stop();
    } catch {
      /* stopping a dead engine is still stopped */
    }
    setLocalEngine(null);
  }, []);

  /**
   * Rebuilds are serialized, and stale ones are skipped. "Selecting a model IS
   * the switch" (2026-08-15) made quick successive picks easy — the Settings
   * radios are a list of one-click switches — and two rebuilds running
   * concurrently race each other through leave/create/start on the pairing
   * provider. Each pick bumps the sequence; a queued rebuild that is no longer
   * the newest does not run. The last pick wins, in one line of teardown.
   */
  const switchSeq = useRef(0);
  const switchChain = useRef<Promise<void>>(Promise.resolve());
  const queueRebuild = useCallback((job: () => Promise<void>): Promise<void> => {
    const seq = ++switchSeq.current;
    const chained = switchChain.current.then(async () => {
      if (seq !== switchSeq.current) return;
      await job();
    });
    // The chain must survive a failed job, or one refused start would wedge
    // every switch after it.
    switchChain.current = chained.catch(() => {});
    return chained;
  }, []);

  /**
   * Switch to a user-supplied GGUF from wherever the picker is open. Same
   * restart semantics as switchModel: save the choice; when something is
   * running, tear it down and serve the new file through the local engine
   * (llamacpp_serve stops any engine sidecars itself — a model switch is a
   * rebuild, never a hot swap).
   */
  const switchToCustom = useCallback(
    async (c: CustomModelSource) => {
      updateSettings({
        ...settings,
        modelId: LOCAL_GGUF_ID,
        quant: "",
        customSource: c.source,
        customGgufPath: c.path,
        customHfRepo: c.repo,
        customHfFile: c.file,
      });
      const clusterRunning = pairSnap && pairSnap.peers.length > 0;
      if (!clusterRunning && !localEngine) return; // nothing running: the pick is the whole operation
      setView("cluster");
      return queueRebuild(async () => {
        try {
          if (clusterRunning) await getPairingProvider().leave();
          await serveOpenModel(c);
        } catch (e) {
          if (!isWeightsCancelled(e)) console.error("switch-to-custom:", e);
          reportWeightsError(e, "cluster");
        }
      });
    },
    [settings, pairSnap, localEngine, serveOpenModel, reportWeightsError, queueRebuild]
  );

  // One object, two renderers: the model row in Settings → Quick, and the
  // "run on this machine" row on Cluster. Both start the same download, so
  // both must show the same progress — building it once is what keeps them
  // from drifting into two half-truths.
  const weightsInfo = useMemo<WeightsInfo>(
    () => ({
      needs: needsWeights,
      path: weightsPath,
      dl,
      partialBytes,
      lastError,
      onDownload: () =>
        void (customSel && customSel.source === "hf"
          ? ensureCustomWeights(customSel).then(() => refreshWeights())
          : ensureWeights()
        ).catch((e) => reportWeightsError(e)),
      onCancel: cancelDownload,
    }),
    // customSel is derived from settings each render; its fields (not its
    // identity) are what matter here, and they are covered by refreshWeights'
    // own deps — spelling the primitives out again keeps the memo honest.
    [
      needsWeights,
      weightsPath,
      dl,
      partialBytes,
      lastError,
      ensureWeights,
      ensureCustomWeights,
      refreshWeights,
      cancelDownload,
      reportWeightsError,
      settings.modelId,
      settings.customSource,
      settings.customHfRepo,
      settings.customHfFile,
    ]
  );

  /**
   * "Run it here": download the weights if needed, then serve from this one
   * machine. Single-node-first (small-model-design §4) — a solo create
   * (num_workers=1, co-located worker) plus start IS a 1-node serving cluster;
   * more machines can join later via Manage.
   *
   * Standalone means this machine is the coordinator and nobody else can feed
   * it shards, so a complete local copy is **required** — hence ensureWeights
   * first, which blocks for the whole download. That step used to be absent:
   * "auto" passed an empty path, the engine took the mock branch, mock no
   * longer falls back, and all the user saw was "failed to start".
   *
   * Named (not inline) so the UI-test channel can drive the REAL handler
   * instead of a lookalike copy — headless_pair's "create" is not equivalent,
   * it waits for a second peer before starting.
   */
  const serveStandalone = useCallback(async (over?: { modelId: string; quant: string }): Promise<boolean> => {
    // Returns false when the hardware probe has not landed yet — there is no
    // hostname/GPU to register. A user cannot hit this (the button only exists
    // once the dashboard has a probe), but the UI-test channel fires on mount,
    // and a silent no-op there reads as "serving is broken".
    if (!snap) return false;
    try {
      const path = await ensureWeights(over);
      await getPairingProvider().create({
        hostname: snap.hostname,
        gpu: snap.gpu_name,
        modelPath: path,
        tuning: engineTuning(over ? { ...settings, ...over } : settings, caps),
      });
      // allowSolo: this IS the one-machine flow. Without it the engine's
      // 2-machine pairing floor rejects the start and the button dies after
      // downloading the weights.
      await getPairingProvider().start(true);
      return true;
    } catch (e) {
      // Cancelling the weights download cancels serving too — that is the same
      // decision, not a second failure to report.
      if (!isWeightsCancelled(e)) console.error("serve-standalone:", e);
      reportWeightsError(e, "cluster");
      return true; // it ran; it failed. Distinct from "could not run yet".
    }
  }, [snap, ensureWeights, settings, caps, reportWeightsError]);

  /**
   * Switch the model from wherever it is displayed (chat header, cluster card).
   *
   * The engine has no hot swap — a model is fixed when the coordinator loads it
   * — so this is "save the choice, and rebuild whatever is running around it":
   *
   *   nothing running  save only. The next start picks it up.
   *   one machine      leave (stops the engine) -> serveStandalone with the new
   *                    selection, downloading its weights first if needed.
   *   several machines leave, then re-create the roster **under the same join
   *                    code** and stop there. The other machines rejoin (their
   *                    client re-registers when it finds itself missing from
   *                    the roster) and the user presses Start — the same
   *                    forming flow as any other cluster, so nothing new has to
   *                    be explained. Minting a fresh code here would strand
   *                    every machine still holding the old one.
   *
   * It also jumps to the Cluster page whenever it restarts something: the
   * download bar, the per-machine stages and the failure messages all live
   * there, and a switch started from chat would otherwise be a page that goes
   * quiet for however long an 80 GB model takes.
   */
  const switchModel = useCallback(
    async (modelId: string, quant: string) => {
      // The setting is written immediately — the radio/picker must reflect the
      // choice now, not after whatever rebuild is currently winding down.
      updateSettings({ ...settings, modelId, quant });
      return queueRebuild(async () => {
      // Switching AWAY from a running local llama.cpp engine: stop it, then
      // serve the curated pick the single-machine way (same restart promise
      // the picker made — nothing keeps running under the old model).
      if (localEngine) {
        await stopLocalEngine();
        if (snap) await serveStandalone({ modelId, quant });
        return;
      }
      const running = pairSnap && pairSnap.peers.length > 0;
      if (!running || !snap) return;
      const machines = pairSnap.peers.length;
      const code = pairSnap.code;
      const account = !!pairSnap.accountMode;
      setView("cluster");
      try {
        await getPairingProvider().leave();
        if (machines <= 1) {
          await serveStandalone({ modelId, quant });
          return;
        }
        const path = await ensureWeights({ modelId, quant });
        const self = {
          hostname: snap.hostname,
          gpu: snap.gpu_name,
          modelPath: path,
          tuning: engineTuning({ ...settings, modelId, quant }, caps),
        };
        // Account mode has no typed code: the secret is derived from the
        // account, so every machine re-derives the same one and finds us again.
        if (account) {
          const g = platformGate();
          const secret = g.ok && g.session.userId ? await accountPairSecret(g.session.userId, g.url, settings.clusterName.trim() || "home") : null;
          if (secret) await getPairingProvider().createAccount(self, secret);
        } else {
          await getPairingProvider().create(self, code ?? undefined);
        }
      } catch (e) {
        if (!isWeightsCancelled(e)) console.error("switch-model:", e);
        reportWeightsError(e, "cluster");
      }
      });
    },
    [settings, pairSnap, snap, localEngine, stopLocalEngine, serveStandalone, ensureWeights, caps, reportWeightsError, queueRebuild]
  );

  // The UI-test effect runs once on mount, so it captures the FIRST render's
  // closure — where snap is still null. A ref keeps the directive pointed at
  // the current handler instead of a stale one.
  const serveStandaloneRef = useRef(serveStandalone);
  useEffect(() => {
    serveStandaloneRef.current = serveStandalone;
  });
  const serveOpenRef = useRef(serveOpenModel);
  useEffect(() => {
    serveOpenRef.current = serveOpenModel;
  });

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
      // Echo the list to the launcher's log first: whether this loop ran at
      // all (and with what) must be observable from a headless run.
      if (list.length) reportTest("directives", list);
      for (const d of list) {
        const m = d.match(/^engine-start:(worker|coordinator)$/);
        if (m) getEngineProvider().start(m[1] as EngineRole).catch(() => {});
        // Quit after N ms (graceful-exit tests). It goes through app_quit, not
        // through the window's close button: with "close to tray" on — the
        // default — closing the window HIDES it, and every gate that ends with
        // this directive would hang waiting for a process that is still
        // happily serving in the background.
        const q = d.match(/^quit:(\d+)$/);
        if (q) setTimeout(() => void quitApp(), Number(q[1]));

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
            ...engineTuning(settings, caps),
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
            }).catch((e) => {
              // Surface to the launcher's log too: a directive failure that
              // only reaches the webview console is invisible to every
              // headless run (cost a whole debugging round, 2026-08-15).
              console.error("uiTest pairing:", e);
              reportTest("pairing-invoke-error", { op, error: String(e) });
            })
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
                tuning: engineTuning(settings, caps),
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
        // ---- update (gate G-UPDATE) ------------------------------------
        // `update-check` asks the real feed through the real provider; the
        // gate points IDLETOKEN_UPDATE_URL/PUBKEY at a manifest it signed
        // itself, so what runs here is the production path, not a copy of it.
        if (d === "update-check") {
          const provider = getUpdateProvider();
          provider
            .check(loadSettings().updateChannel)
            .then(async (info) => {
              const st = await provider.state(loadSettings().updateChannel);
              reportTest("update", {
                found: info !== null,
                version: info?.version ?? null,
                current: st.currentVersion,
                feed: st.feed,
                feedOverridden: st.feedOverridden,
                error: null,
              });
            })
            // The distinction the gate exists to protect: a check that could
            // not run reports an error, never "up to date".
            .catch((e) => reportTest("update", { found: null, error: String(e) }));
        }
        // Download + verify WITHOUT installing (installing would replace the
        // binary the harness is running). The signature check happens inside
        // the download, so this is the assertion that a tampered artifact is
        // refused — the gate runs it both ways.
        if (d === "update-download") {
          const provider = getUpdateProvider();
          provider
            .check(loadSettings().updateChannel)
            .then(async (info) => {
              if (!info) {
                reportTest("update-download", { ok: false, reason: "no update offered by the feed" });
                return;
              }
              const bytes = await provider.download();
              reportTest("update-download", { ok: true, version: info.version, bytes });
            })
            .catch((e) => reportTest("update-download", { ok: false, reason: String(e) }));
        }
        // ---- tray / background residency (gate G-TRAY) -------------------
        // What the shell believes about the tray and the window, straight from
        // Rust: whether an icon actually exists (it can fail to on Linux) and
        // therefore whether hiding the window is allowed at all.
        if (d === "report-shell-window") {
          windowState()
            .then((st) => reportTest("shell-window", st ?? { error: "not running in the shell" }))
            .catch((e) => reportTest("shell-window", { error: String(e) }));
        }
        // Close the window the way the X button does and report what happened.
        // The report arriving at all is half the assertion: it is printed by a
        // process that closing the window did not kill.
        const ct = d.match(/^close-to-tray:(\d+)$/);
        if (ct) {
          (async () => {
            // Directives run before the settings-sync effect below, so the
            // shell is told explicitly rather than tested against whatever
            // window.json happened to be left behind by an earlier run.
            await syncWindowPrefs(loadSettings());
            const before = await windowState();
            const { getCurrentWindow } = await import("@tauri-apps/api/window");
            await getCurrentWindow().close();
            await new Promise((r) => setTimeout(r, Number(ct[1])));
            const after = await windowState();
            reportTest("close-to-tray", {
              hideAllowed: before?.hideAllowed ?? null,
              trayAlive: before?.trayAlive ?? null,
              visibleBefore: before?.visible ?? null,
              visibleAfter: after?.visible ?? null,
              stillRunning: after !== null,
            });
          })().catch((e) => reportTest("close-to-tray", { error: String(e) }));
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
        // Cancel oracle (2026-08-10). "Press cancel, then see what the client
        // says" is a timing question, and timing is exactly what reading the
        // code cannot settle: the engine notices the cancel flag only between
        // reads, so the wrong version stayed on screen for seconds and then
        // announced a failure. This starts a real download, cancels it through
        // the SAME handler the button calls, and reports what arrived after.
        //   weights-cancel:<modelId>:<ms-before-cancel>
        const wc = d.match(/^weights-cancel:([a-z0-9.\-]+):(\d+)$/);
        if (wc) {
          (async () => {
            const seen: string[] = [];
            const un = await onFetchProgress((p) => {
              if (p.kind !== "progress") seen.push(p.kind);
            });
            try {
              const man = getManifest(wc[1]);
              const target = resolveDownload(man, defaultQuant(wc[1]));
              if (!target) { reportTest("weightsCancel", { error: "manifest has no repo/gguf" }); return; }
              const dir = await defaultModelDir();
              // Route the rejection the way a user's click does. Swallowing it
              // here (`.catch(() => {})`) is what let the bug hide: the command
              // rejects on a cancel too, and in the app that rejection landed
              // in a catch that painted "Download failed — cancelled …".
              void fetchWeights({ id: "primary", target, destDir: dir }).catch(reportWeightsError);
              await new Promise((r) => setTimeout(r, Number(wc[2])));
              const tCancel = Date.now();
              cancelDownload();
              const uiClearedMs = Date.now() - tCancel; // the click-to-UI latency
              await new Promise((r) => setTimeout(r, 15000)); // let the engine wind down
              reportTest("weightsCancel", {
                uiClearedMs,
                kindsAfterStart: seen,
                sawError: seen.includes("error"),
                sawCancelled: seen.includes("cancelled"),
                // What the user is left looking at. The event stream had been
                // right about this for a while and the screen still said
                // "Download failed — cancelled (...)", because the failure came
                // in on the command's rejection instead. An oracle that only
                // watches the events cannot see that; this one asks the UI.
                errorShownAfterCancel: errRef.current,
              });
            } catch (e) {
              reportTest("weightsCancel", { error: String(e) });
            } finally {
              un();
            }
          })();
        }
        // Concurrency oracle (2026-08-10). Two fetches for one id used to both
        // run, appending to the same .part until it was longer than the source
        // — which the resume path then renamed and called a finished download.
        // The second call must be refused, and the file must never exceed total.
        //   weights-double:<modelId>
        const wd = d.match(/^weights-double:([a-z0-9.\-]+)$/);
        if (wd) {
          (async () => {
            try {
              const man = getManifest(wd[1]);
              const target = resolveDownload(man, defaultQuant(wd[1]));
              if (!target) { reportTest("weightsDouble", { error: "manifest has no repo/gguf" }); return; }
              const dir = await defaultModelDir();
              const first = fetchWeights({ id: "primary", target, destDir: dir }).catch((e) => `ERR:${e}`);
              await new Promise((r) => setTimeout(r, 3000)); // let it get past register()
              let secondRefused = false;
              let secondMsg = "";
              try {
                await fetchWeights({ id: "primary", target, destDir: dir });
              } catch (e) {
                secondRefused = true;
                secondMsg = String(e);
              }
              cancelDownload();
              await new Promise((r) => setTimeout(r, 8000));
              const st = await weightsState(dir, target.file, target.expectBytes);
              reportTest("weightsDouble", {
                secondRefused, secondMsg,
                haveBytes: st.have_bytes, expectBytes: target.expectBytes,
                overshoot: st.have_bytes > target.expectBytes,
              });
              void first;
            } catch (e) {
              reportTest("weightsDouble", { error: String(e) });
            }
          })();
        }
        // "Does finishing the download actually start the service?" — driven
        // through the SAME callback the button uses, then it reports what the
        // cluster and its API did. headless_pair's create() is not a substitute:
        // it waits for a second peer before starting, which is the opposite of
        // what this path is for.
        if (d === "serve-standalone") {
          (async () => {
            const t0 = Date.now();
            try {
              // There is no snapshot() on the provider — state arrives by
              // subscription, so latch the latest and watch that. Held in an
              // object because a plain `let` assigned only inside the callback
              // gets narrowed to `never` by control-flow analysis.
              const box: { s: PairingSnapshot | null } = { s: null };
              const un = getPairingProvider().subscribe((v) => { box.s = v; });
              // Wait for the probe before pressing: on mount there is no snap.
              let pressed = false;
              for (let i = 0; i < 30 && !pressed; i++) {
                pressed = await serveStandaloneRef.current();
                if (!pressed) await new Promise((r) => setTimeout(r, 1000));
              }
              if (!pressed) { reportTest("serveStandalone", { error: "probe never landed; never pressed" }); un(); return; }
              for (let i = 0; i < 120; i++) {
                await new Promise((r) => setTimeout(r, 2000));
                if (box.s && (box.s.phase === "ready" || box.s.peers.some((p) => p.stage === "error"))) break;
              }
              un();
              const s = box.s;
              reportTest("serveStandalone", {
                seconds: Math.round((Date.now() - t0) / 1000),
                phase: s?.phase ?? null,
                peers: s?.peers.map((p) => ({ stage: p.stage, layers: `${p.layerLo}-${p.layerHi}` })) ?? [],
                api: s?.api ?? null,
              });
            } catch (e) {
              reportTest("serveStandalone", { error: String(e), seconds: Math.round((Date.now() - t0) / 1000) });
            }
          })();
        }
        // Open-model oracle (v2 WS-D1): serve an arbitrary local GGUF through
        // the REAL handler (coordinator llamacpp mode), then report what the
        // coordinator said — auto-manifest id + engine state on success, the
        // worded refusal on refusal. Both outcomes are reportable on purpose:
        // the refusal path is a feature, not a failure of the test.
        //   serve-open-gguf:<absolute-path.gguf>
        const sg = d.match(/^serve-open-gguf:(\S+)$/);
        if (sg) {
          (async () => {
            const t0 = Date.now();
            try {
              await serveOpenRef.current({ source: "file", path: sg[1], repo: "", file: "" });
              const { invoke } = await import("@tauri-apps/api/core");
              const base = `http://127.0.0.1:${loadSettings().apiPort || 8000}`;
              let stats: Record<string, unknown> | null = null;
              let refused: string | null = null;
              for (let i = 0; i < 60; i++) {
                await new Promise((r) => setTimeout(r, 2000));
                const st = await getEngineProvider().status();
                if (st.refusedReason) {
                  refused = st.refusedReason;
                  break;
                }
                try {
                  stats = await invoke<Record<string, unknown>>("api_stats", { baseUrl: base });
                } catch {
                  /* API not up yet */
                }
                if (stats && (stats.engine_state === "ready" || stats.engine_state === "failed")) break;
              }
              reportTest("serveOpenGguf", {
                ctx: uiTestCtx,
                seconds: Math.round((Date.now() - t0) / 1000),
                refused,
                engine: (stats?.engine as string) ?? null,
                engineState: (stats?.engine_state as string) ?? null,
                model: (stats?.model as string) ?? null,
              });
            } catch (e) {
              reportTest("serveOpenGguf", { error: String(e) });
            }
          })();
        }
        // Stop-generation oracle (2026-08-11). Two claims worth proving on real
        // hardware: deltas stop arriving when the user presses Stop, and the
        // partial text survives. Timing again — unprovable by reading code.
        //   chat-stop:<ms-before-stop>
        const cs = d.match(/^chat-stop:(\d+)$/);
        if (cs) {
          (async () => {
            try {
              const { listen } = await import("@tauri-apps/api/event");
              const { invoke } = await import("@tauri-apps/api/core");
              // Wait for the cluster this test needs.
              let api: ClusterApi | null = null;
              const box: { s: PairingSnapshot | null } = { s: null };
              const un = getPairingProvider().subscribe((v) => { box.s = v; });
              for (let i = 0; i < 120; i++) {
                await new Promise((r) => setTimeout(r, 2000));
                if (box.s?.api?.status === "online") { api = box.s.api; break; }
              }
              un();
              if (!api) { reportTest("chatStop", { error: "cluster never came online" }); return; }

              const reqId = `stoptest-${Date.now()}`;
              let deltas = 0;
              let chars = 0;
              let deltasAfterStop = 0;
              let stopped = false;
              const unl = await listen<{ id: string; kind: string; text?: string }>("api-chat", (ev) => {
                if (ev.payload.id !== reqId) return;
                if (ev.payload.kind === "delta") {
                  deltas++;
                  chars += ev.payload.text?.length ?? 0;
                  if (stopped) deltasAfterStop++;
                }
              });
              void invoke("api_chat_stream", {
                id: reqId,
                baseUrl: api.baseUrl,
                messages: [{ role: "user", content: "Write a very long, detailed 2000-word essay about distributed systems." }],
                token: "",
                model: settings.modelId,
                maxTokens: 4096,
              }).catch(() => {});
              await new Promise((r) => setTimeout(r, Number(cs[1])));
              const atStop = deltas;
              const tStop = Date.now();
              stopped = true;
              const wasRunning = await invoke<boolean>("api_chat_cancel", { id: reqId });
              await new Promise((r) => setTimeout(r, 8000)); // watch for stragglers
              unl();
              reportTest("chatStop", {
                wasRunning,
                deltasBeforeStop: atStop,
                charsKept: chars,
                deltasAfterStop,
                msWatchedAfterStop: Date.now() - tStop,
              });
            } catch (e) {
              reportTest("chatStop", { error: String(e) });
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
  }, [settings.accent, settings.density, settings.reduceMotion]);

  // uiScale 0 = auto: track the window and pick a band (see autoUiScale). The
  // listener exists only in auto mode, and the band + hysteresis mean a resize
  // drag crosses at most one boundary — a zoom change re-lays-out the whole
  // document, so it has to be rare, not per-pixel.
  const [autoScale, setAutoScale] = useState(() => autoUiScale(window.innerWidth, window.innerHeight));
  useEffect(() => {
    if (settings.uiScale !== 0) return;
    let raf = 0;
    const onResize = () => {
      cancelAnimationFrame(raf);
      // body zoom does not change window.innerWidth, so this cannot feed back.
      raf = requestAnimationFrame(() => setAutoScale((p) => autoUiScale(window.innerWidth, window.innerHeight, p)));
    };
    onResize();
    window.addEventListener("resize", onResize);
    return () => { window.removeEventListener("resize", onResize); cancelAnimationFrame(raf); };
  }, [settings.uiScale]);
  useEffect(() => {
    (document.body.style as { zoom?: string }).zoom = String(settings.uiScale || autoScale);
  }, [settings.uiScale, autoScale]);

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

  // The local llama.cpp engine's API, shaped like the cluster's so chat and
  // stats consume either without caring which. "online" = the coordinator
  // process is up and answering its API (it 503s honestly while the model
  // loads — that message reaches the user through chat and the engine card).
  const localApi: ClusterApi | null = localEngine
    ? {
        baseUrl: `http://127.0.0.1:${settings.apiPort || 8000}`,
        status:
          engStatus === null || engStatus.state === "starting"
            ? "starting"
            : engStatus.state === "running" || engStatus.state === "restarting"
              ? "online"
              : "offline",
      }
    : null;

  // One poll for the local engine's counters/health, shared by the topbar pill
  // and the local-engine card (the coordinator serves requests serially — a
  // second poller would compete with generation for nothing).
  const localStats = useClusterStats(localApi, "engine", {});

  const cluster: ClusterState =
    pairSnap === null || pairSnap.phase === "idle"
      ? pairSnap && pairSnap.peers.length > 0
        ? "joining"
        : localEngine
          ? // "ready" only once the engine reports ready — the API 503s while
            // the model loads, and a green pill over a 503 is a fake green.
            localStats?.engine_state === "ready"
            ? "ready"
            : "joining"
          : "standalone"
      : pairSnap.phase === "ready"
        ? "ready"
        : "joining";

  // ---- shell integration: tray, window behaviour, launch at login ----------
  // These four settings are the only ones the Rust side needs a copy of: it
  // acts on them before the webview exists (start minimized) and after it may
  // be gone (the window's X). Pushing them on every change is what keeps its
  // mirror from drifting; see system.ts.
  useEffect(() => {
    void syncWindowPrefs(settings);
  }, [settings.trayIcon, settings.closeToTray, settings.startMinimized, settings.rememberWindow]);

  // Settings are the source of truth for "launch at login", and this enforces
  // them on the OS every launch — self-healing when something else (an
  // uninstall/reinstall, a cleanup tool) removed the entry behind our back.
  useEffect(() => {
    void setAutostart(settings.autostart).catch(() => {
      /* the OS refused; the setting stays as the user's intent */
    });
  }, [settings.autostart]);

  // Tray menu text + the one line of status it shows. Rebuilt on language and
  // on cluster changes rather than translated in Rust — the language and the
  // cluster state both live here, and a second copy of either is a second
  // thing that can go stale.
  const trayMachines = pairSnap?.peers.length ?? 0;
  useEffect(() => {
    const status =
      cluster === "ready"
        ? t2("tray.statusServing", lang).replace("{n}", String(Math.max(1, trayMachines)))
        : cluster === "joining"
          ? t2("tray.statusStarting", lang)
          : t2("tray.statusIdle", lang);
    void syncTray({
      open: t2("tray.open", lang),
      status,
      check_update: t2("tray.checkUpdate", lang),
      quit: t2("tray.quit", lang),
    });
  }, [lang, cluster, trayMachines, settings.trayIcon]);

  // ---- one-time "still running in the tray" notice -------------------------
  // The first close-to-tray is otherwise indistinguishable from the app having
  // quit. The shell emits `tray:hidden` after hiding the window (main.rs); the
  // webview stays alive while hidden, so an in-app toast is waiting when the
  // window comes back — and the flag in settings keeps it to exactly once.
  const [trayToast, setTrayToast] = useState(false);
  useEffect(() => {
    if (!inTauri()) return;
    let un: (() => void) | null = null;
    let dead = false;
    void import("@tauri-apps/api/event").then(({ listen }) =>
      listen("tray:hidden", () => {
        setSettings((prev) => {
          if (prev.trayHintShown) return prev;
          const next = { ...prev, trayHintShown: true };
          saveSettings(next);
          setTrayToast(true);
          return next;
        });
      }).then((f) => (dead ? f() : (un = f)))
    );
    return () => {
      dead = true;
      un?.();
    };
  }, []);
  // Dismiss only after the window has been VISIBLE for a while — a timer that
  // runs out while the window is hidden would remove the notice before anyone
  // could have read it.
  useEffect(() => {
    if (!trayToast) return;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const arm = () => {
      if (timer) clearTimeout(timer);
      timer = null;
      if (document.visibilityState === "visible") timer = setTimeout(() => setTrayToast(false), 8000);
    };
    arm();
    document.addEventListener("visibilitychange", arm);
    return () => {
      if (timer) clearTimeout(timer);
      document.removeEventListener("visibilitychange", arm);
    };
  }, [trayToast]);

  // ---- update ------------------------------------------------------------
  const [update, setUpdate] = useState<UpdateResult | null>(null);

  /**
   * One check, two ways of reporting it.
   *
   * `announce: "always"` is a check the user asked for (Settings button, tray
   * menu): every outcome opens the dialog, including "you are current" and
   * "could not reach the feed" — a button that answers nothing reads as
   * broken, and the two answers are not the same fact.
   *
   * `announce: "found"` is the automatic check at startup: it interrupts only
   * when there is something to install. A failed check there is deliberately
   * silent — an unreachable feed four seconds after launch is not worth a
   * modal — which is exactly why the manual path must never be.
   */
  const checkUpdate = useCallback(
    async (announce: "always" | "found") => {
      try {
        const info = await getUpdateProvider().check(settings.updateChannel);
        if (info) setUpdate({ kind: "found", info });
        else if (announce === "always") {
          const st = await getUpdateProvider().state(settings.updateChannel);
          setUpdate({ kind: "upToDate", current: st.currentVersion });
        }
      } catch (e) {
        if (announce === "always") setUpdate({ kind: "error", message: String(e) });
      }
    },
    [settings.updateChannel]
  );

  // Automatic check: once per launch, a few seconds in so it never competes
  // with the probe and the cluster rejoin.
  useEffect(() => {
    if (!settings.autoUpdate) return;
    const timer = setTimeout(() => void checkUpdate("found"), 4000);
    return () => clearTimeout(timer);
    // Once per launch: re-running it on every settings edit would nag.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // "Check for updates…" from the tray menu, through the same path as the
  // Settings button.
  useEffect(() => {
    if (!inTauri()) return;
    const un = import("@tauri-apps/api/event").then(({ listen }) =>
      listen("tray-check-update", () => void checkUpdate("always"))
    );
    return () => {
      un.then((f) => f());
    };
  }, [checkUpdate]);

  return (
    <LangContext.Provider value={{ lang, setLang }}>
      <div className="app">
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
                api={pairSnap?.api ?? localApi}
                source={pairSnap?.source ?? "engine"}
                apiToken={settings.apiToken}
                modelId={settings.modelId}
                modelLabel={model.label}
                quant={settings.quant}
                maxTokens={settings.maxTokens}
                identity={identity}
                onGoCluster={() => setView("cluster")}
                machines={pairSnap?.peers.length ?? (localEngine ? 1 : 0)}
                onSwitchModel={(id, q) => void switchModel(id, q)}
                onPickCustomModel={(c) => void switchToCustom(c)}
                customName={customName}
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
                weights={weightsInfo}
                onSwitchModel={(id, q) => void switchModel(id, q)}
                onWeightsChanged={() => void refreshWeights()}
                onCheckUpdate={() => checkUpdate("always")}
                onClose={() => setView("cluster")}
              />
            ) : (
              <Dashboard
                snap={snap}
                model={model}
                quant={settings.quant}
                tier={TIERS.find((x) => x.id === settings.tier) ?? TIERS[1]}
                pair={pairSnap}
                localEngine={localEngine}
                localApi={localApi}
                localStats={localStats}
                engStatus={engStatus}
                customName={customName}
                onStopLocal={() => void stopLocalEngine()}
                onPickCustom={(c) => void switchToCustom(c)}
                // Wrapped, not passed through: serveStandalone now takes an
                // optional {modelId,quant} override, and a bare handler would
                // hand it the click event as that argument. An open GGUF's
                // "run it here" is the local llama.cpp engine, not pairing.
                onServeStandalone={() => void (customSel ? serveOpenModel() : serveStandalone())}
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
                onOpenModelSetting={() => openSettings("quick")}
                onSwitchModel={(id, q) => void switchModel(id, q)}
                weights={weightsInfo}
              />
            )}
          </div>
        </div>
      </div>
      {/* No floating download bar at all (2026-08-10). Every weight state —
          missing, downloading, failed, ready — is rendered on the model row it
          belongs to, in Settings → Quick. A box that covers the page to narrate
          a file transfer the user already asked for is pure interruption. */}
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
            tuning: engineTuning(settings, caps),
          }}
          session={session}
          modelId={settings.modelId}
          initialView={pairingView}
          onSignIn={() => {
            setShowPairing(false);
            setShowAuth(true);
          }}
          onClose={() => setShowPairing(false)}
        />
      ) : null}
      {update ? (
        <UpdateDialog
          result={update}
          onClose={() => setUpdate(null)}
          onOutcome={(o) => reportTest("update-install", o)}
        />
      ) : null}
      {trayToast ? (
        <div className="toast" role="status" onClick={() => setTrayToast(false)}>
          {t2("tray.hidden", lang)}
        </div>
      ) : null}
    </LangContext.Provider>
  );
}

// Loading/error strings render in App itself, ABOVE the LangContext provider
// — a tiny direct lookup keeps them translated without hook gymnastics.
function t2(key: keyof (typeof STRINGS)["en"], lang: Lang): string {
  return (STRINGS[lang] as Record<string, string>)[key] ?? key;
}
