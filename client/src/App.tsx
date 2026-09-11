import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { LangContext, STRINGS, useI18n, type Lang } from "./i18n";
import { getResourceProvider } from "./provider";
import { getEngineProvider, type EngineLogLine, type EngineStatus } from "./provider/engine";
import type { NodeSnapshot, ClusterState } from "./types";
import { HW_OK, HW_NO_GPU, HW_CC_TOO_LOW, HW_DRIVER_TOO_OLD, HW_VRAM_TOO_SMALL, HW_GPU_UNSUPPORTED, HW_MACOS_SEALED } from "./types";
import { getModel, getManifest, estimateClusterCapacity, poolVram, poolRam, clusterCapacityVerdict, hybridRequirements, isMoeModel, moeRamExpertBudget, pickBestFittingModel, backendOfOs, type ModelSpec, type MoeLayoutBudget } from "./models";
import { resolveLocalWeights, fetchWeights, onFetchProgress, defaultModelDir, cancelFetch, resolveDownload, verifyWeights, isWeightsCancelled, type DownloadTarget } from "./weights";
import { loadSettings, saveSettings, settingsWerePersisted, effectiveCaps, effectiveCtx, engineTuning, overflowTuning, autoUiScale, contextTiersFor, type AppSettings, type ContextTier, type Tier } from "./settings";
import { getAuthProvider, type Session } from "./auth";
import SettingsPanel from "./SettingsPanel";
import { OverflowToggleButton, ShareToggleButton } from "./PlatformPanel";
import SparkBalancePill from "./SparkBalancePill";
import AuthScreen from "./AuthScreen";
import PairingPanel from "./PairingPanel";
import Chat from "./Chat";
import ModelPicker from "./ModelPicker";
import StartupProgress from "./StartupProgress";
import WeightsRow, { type WeightsInfo } from "./WeightsRow";
import { inTauri, getMe, resumeSharingAgent } from "./platform";
import { identityFrom, type UserIdentity } from "./Avatar";
import { getPairingProvider, type PairingSnapshot, type ClusterApi, type PeerNode } from "./pairing";
import { recordProblem } from "./problems";
import { useClusterStats, servedModelOf, type ClusterStats } from "./clusterStats";
import { compactCount, ctxLabel, fmtGiB, fmtQuant, pct } from "./format";
import { setAutostart, syncTray, syncWindowPrefs } from "./system";

type Theme = "dark" | "light";

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
// machine remains in the client; commercial controls live in the signed-in portal.
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
  // Same "cluster.ready" wording as the cluster page's phase label, so the two
  // surfaces never disagree about whether the machine is ready to answer.
  const clusterKey =
    props.cluster === "ready" ? "cluster.ready" : props.cluster === "joining" ? "cluster.joining" : "cluster.standalone";
  return (
    <header className="topbar">
      <div className="topbar__layout">
        <div className="topbar__primary">
          <div className="brand">
            <span className="brand__mark"><span className="brand__name">IdleToken</span></span>
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
        </div>
        <div className="topbar__actions">
          {/* The two marketplace directions are independent global controls. They
              stay together in the top-right chrome instead of being mixed into a
              model or deployment decision. */}
          <div className="topbar__market-actions">
            <ShareToggleButton serviceReady={props.cluster === "ready"} onNeedLogin={props.onSignIn} />
            <OverflowToggleButton serviceReady={props.cluster === "ready"} onNeedLogin={props.onSignIn} />
          </div>
          <button className={`pill pill--${props.cluster}`} onClick={props.onGoCluster} title={t("nav.cluster")}>
            <span className="pill__dot" />
            <span className="pill__label">{t(clusterKey)}</span>
          </button>
          {/* Account balance is global chrome, not cluster-page content. The
              component hides itself when signed out and never persists its value. */}
          <SparkBalancePill />
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
        </div>
      </div>
    </header>
  );
}

// ---- this machine: one dense card = probe strip + capacity guidance --------
// Replaces the old GPU card + stat tiles + layer spine trio (2026-07 UX pass):
// the probe is supporting detail, so it gets one row; the pixels go to the
// question that actually matters before pairing — "is my hardware enough,
// and how far off am I?" (estimateClusterCapacity, engine-estimate mirror).
interface ModelLayoutJson {
  expert_bytes_total: number;
  n_expert: number;
  n_expert_used: number;
  expert_bytes_complete: boolean;
}

const modelLayoutCache = new Map<string, Promise<MoeLayoutBudget>>();

function loadMoeLayout(modelId: string, quant: string, ggufPath: string): Promise<MoeLayoutBudget> {
  const key = `${modelId}\n${quant}\n${ggufPath}`;
  const cached = modelLayoutCache.get(key);
  if (cached) return cached;
  const pending = import("@tauri-apps/api/core")
    .then(({ invoke }) => invoke<ModelLayoutJson>("inspect_model_layout", {
      modelId,
      quant,
      ggufPath,
    }))
    .then((r) => ({
      expertBytesTotal: r.expert_bytes_total,
      nExpert: r.n_expert,
      nExpertUsed: r.n_expert_used,
      complete: r.expert_bytes_complete,
    }))
    .catch((error) => {
      modelLayoutCache.delete(key);
      throw error;
    });
  modelLayoutCache.set(key, pending);
  return pending;
}

function NodeCapacityCard(props: {
  snap: NodeSnapshot;
  model: ModelSpec;
  quant: string; // selected precision → sizes the weight bytes in the estimate
  tier: { id: number; ctx: number };
  nNodes: number; // actual roster size when paired; one real machine otherwise
  /** The paired machines, each carrying the memory IT measured (roster). More
   *  than one = the verdict is about the pool, not just this machine. */
  peers?: PeerNode[];
  weights?: WeightsInfo;
}) {
  const { t } = useI18n();
  const s = props.snap;
  // Availability and the UI estimate use the uncapped memory the GPU reports
  // as free. A user-selected runtime usage cap answers a different question
  // ("how much may IdleToken consume?") and must not make a 15.7 GiB-free card
  // claim that only 13.1 GiB is available.
  const freeMem = { vram_usable: s.vram_usable };
  // Paired: total what every machine reported and answer for the CLUSTER
  // (2026-08-15). Each machine measures its own memory and sends it with its
  // join, so this question was always answerable before pressing Start — until
  // now the only answer came from the coordinator refusing afterwards.
  const peers = props.peers ?? [];
  const pool = useMemo(() => poolVram(peers), [peers]);
  const clustered = peers.length > 1;
  const isMoe = isMoeModel(props.model.id);
  // A MoE model may keep routed experts in node-local RAM. Each member first
  // applies its own hard gates (including Windows' WDDM page-lock ceiling),
  // then the UI adds those already-capped contributions. Zero for dense
  // models and unified memory — see moeRamExpertBudget. The deploy buttons
  // read the same budget through the same function.
  const ramPool = useMemo(() => poolRam(peers), [peers]);
  /* A pre-fix Windows sidecar does not report the WDDM-aware field. Do not
   * silently fall back to all free RAM there: that is precisely the optimistic
   * estimate this fix removes. Non-Windows builds have no WDDM ceiling and may
   * retain their historical usable-RAM value during a rolling upgrade. */
  const localRamExpert = s.ram_expert_usable ?? (s.os === "windows" ? 0 : s.ram_usable);
  const ramExpert = clustered
    ? moeRamExpertBudget(props.model.id, ramPool.bytes, false)
    : moeRamExpertBudget(props.model.id, localRamExpert, s.unified_memory);
  // Backend matters: GLM-5.2's measured workspace is 1.50 GiB on CUDA and
  // 33.25 GiB on Metal. Passing the machine's own OS keeps this card and the
  // deploy buttons reading the same number.
  const gpuAvailable = clustered ? pool.bytes : freeMem.vram_usable;
  const estimateMem = { vram_usable: gpuAvailable };
  const gpuCap = estimateClusterCapacity(props.model, estimateMem, props.tier.ctx,
                                         props.nNodes, props.quant,
                                         backendOfOs(s.os), 0);
  const gpuOnly = gpuAvailable >= gpuCap.needBytes;
  // Mode follows the selected resource path, not the architecture label. An
  // MoE that fits wholly in VRAM is GPU_ONLY and therefore has no RAM row.
  // Unified-memory machines also have no second pool to display.
  const hybridMode = isMoe && !gpuOnly && ramExpert > 0 && (clustered || !s.unified_memory);
  const [cpuName, setCpuName] = useState("");
  useEffect(() => {
    let live = true;
    setCpuName("");
    if (!hybridMode || !inTauri()) return () => { live = false; };
    import("@tauri-apps/api/core")
      .then(({ invoke }) => invoke<string>("cpu_name"))
      .then((name) => {
        if (live) setCpuName(name.trim());
      })
      .catch(() => {
        if (live) setCpuName("");
      });
    return () => { live = false; };
  }, [hybridMode]);
  const [moeLayout, setMoeLayout] = useState<MoeLayoutBudget | null>(null);
  const [layoutState, setLayoutState] = useState<"waiting" | "loading" | "ready" | "error">("waiting");
  useEffect(() => {
    let live = true;
    setMoeLayout(null);
    if (!hybridMode || props.weights?.needs || !props.weights?.path || !inTauri()) {
      setLayoutState("waiting");
      return () => { live = false; };
    }
    setLayoutState("loading");
    loadMoeLayout(props.model.id, props.quant, props.weights.path)
      .then((layout) => {
        if (!live) return;
        setMoeLayout(layout.complete ? layout : null);
        setLayoutState(layout.complete ? "ready" : "error");
      })
      .catch(() => {
        if (live) setLayoutState("error");
      });
    return () => { live = false; };
  }, [hybridMode, props.model.id, props.quant, props.weights?.needs, props.weights?.path]);
  const hybridNeed = hybridMode ? hybridRequirements(gpuCap.needBytes, moeLayout) : null;
  const pooledVerdict = !hybridMode
    ? clustered
      ? clusterCapacityVerdict(gpuCap.needBytes, gpuAvailable, 0, false,
                               pool.complete, true)
      : gpuOnly ? "fits" : "short"
    : !hybridNeed || (clustered && (!pool.complete || !ramPool.complete))
      ? "unknown"
      : gpuAvailable < hybridNeed.vramNeedBytes || ramExpert < hybridNeed.ramNeedBytes
        ? "short"
        : clustered ? "hybrid-check" : "fits";
  const short = pooledVerdict === "short";
  // "Cannot tell" beats a wrong "not enough": a member that reported nothing
  // makes the total a lower bound, and only a SHORTFALL can be wrong that way
  // (a total that already covers the model cannot be talked down by adding
  // more memory to it). With RAM in the sum, an unreported RAM figure counts
  // the same way for a MoE model.
  const missingReport = pooledVerdict === "unknown";
  /* An aggregate MoE sum is only a necessary pre-flight check. Runtime still
   * performs the exact owner-local admission check at launch; the card keeps
   * its verdict focused on whether both displayed estimates meet their floors. */
  // Never round available memory upward or required memory downward. Hybrid
  // uses two decimals because the measured Windows pair has only ~76 MiB of
  // expert-RAM headroom: one decimal would print 77.3 beside 77.3 and hide the
  // very boundary this card is meant to explain. GPU-only keeps the quieter
  // one-decimal readout used elsewhere.
  const budgetDigits = hybridMode ? 2 : 1;
  const budgetScale = 10 ** budgetDigits;
  const GBHave = (b: number) => (
    Math.floor((b / 1024 ** 3) * budgetScale) / budgetScale
  ).toFixed(budgetDigits);
  const GBNeed = (b: number) => (
    Math.ceil((b / 1024 ** 3) * budgetScale) / budgetScale
  ).toFixed(budgetDigits);
  // ONE source for "what is left". This used to recompute
  // `vram_total - vram_used_other` while the capacity line below read
  // `vram_usable`; the two happen to be equal on the machines tested, but they
  // are two expressions for one quantity and nothing kept them equal. The probe
  // now reports remaining memory directly (NVML's own `free`), so read that.
  const vFree = fmtGiB(s.vram_usable);
  const vTotal = fmtGiB(s.vram_total);
  const rFree = fmtGiB(localRamExpert);
  const rTotal = fmtGiB(s.ram_total);
  const total = props.model.totalLayers;
  const ticks = useMemo(() => Array.from({ length: total }), [total]);
  // Each physical pool gets its own coverage bar. Combining both ratios into
  // one bar hid which resource was tight and made RAM look interchangeable
  // with VRAM; Hybrid admission deliberately treats them as separate gates.
  const gpuNeedBytes = hybridMode ? hybridNeed?.vramNeedBytes : gpuCap.needBytes;
  const ramNeedBytes = hybridMode ? hybridNeed?.ramNeedBytes : undefined;
  const coverageLayers = (have: number, need?: number) => need && need > 0
    ? Math.floor(total * Math.min(1, have / need))
    : 0;
  const gpuVisibleLayers = coverageLayers(gpuAvailable, gpuNeedBytes);
  const ramVisibleLayers = coverageLayers(ramExpert, ramNeedBytes);
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
            {vFree.value}
            <span className="unit">/ {vTotal.value} {vTotal.unit}</span>
          </span>
          <div className="track track--mini">
            <span className="track__usable" style={{ width: `${pct(s.vram_total - s.vram_used_other, s.vram_total)}%` }} />
            <span className="track__used" style={{ width: `${pct(s.vram_used_other, s.vram_total)}%` }} />
          </div>
        </div>
        {/* RAM appears only when this selection actually needs Hybrid. A MoE
            that fits in VRAM is GPU_ONLY and reads exactly like a dense model.
            The value is the page-lock-aware expert budget, but the label stays
            ordinary "memory" for people who should not need WDDM vocabulary. */}
        {hybridMode ? (
          <div className="nstat nstat--cpu">
            <span className="nstat__k">{t("node.cpu")}</span>
            <span className="nstat__v">{cpuName || "—"}</span>
            <span className="nstat__sub">{s.cpu_count > 0 ? t("node.threads", { n: s.cpu_count }) : ""}</span>
          </div>
        ) : null}
        {hybridMode ? (
          <div className="nstat nstat--bar">
            <span className="nstat__k">{t("capacity.availableRam")}</span>
            <span className="nstat__v">
              {rFree.value}
              <span className="unit">/ {rTotal.value} {rTotal.unit}</span>
            </span>
            <div className="track track--mini">
              <span className="track__usable" style={{ width: `${pct(localRamExpert, s.ram_total)}%` }} />
              <span className="track__used" style={{ width: `${pct(s.ram_total - localRamExpert, s.ram_total)}%` }} />
            </div>
          </div>
        ) : null}
      </div>

      {/* GPU_ONLY has one physical pool. Hybrid has two, so each pool owns its
          numbers and its own coverage bar; neither is added to the other. */}
      <div className="capacity">
        {!hybridMode ? (
          <div className="capacity__head">
            <span className="capacity__mode">{t("capacity.modeGpu")}</span>
          </div>
        ) : null}
        <div className={`capacity__resources${hybridMode ? " capacity__resources--headless" : ""}`}>
          <div className="capacity__resource">
            <span className="capacity__resource-kind">{t("node.vram")}</span>
            <div className="capacity__resource-stats">
              <span className="capacity__resource-stat">
                <span>{t("capacity.available")}</span>
                <strong>{GBHave(gpuAvailable)} GB</strong>
              </span>
              <span className="capacity__resource-stat">
                <span>{t(hybridMode ? "capacity.minimum" : "capacity.required")}</span>
                <strong className={short && (!hybridNeed || gpuAvailable < hybridNeed.vramNeedBytes) ? "capacity__gap" : ""}>
                  {hybridMode
                    ? hybridNeed ? `${GBNeed(hybridNeed.vramNeedBytes)} GB` : "—"
                    : `${GBNeed(gpuCap.needBytes)} GB`}
                </strong>
              </span>
            </div>
            <div className="spine capacity__spine" role="img" aria-label={`${t("node.vram")} · ${t("capacity.available")} ${GBHave(gpuAvailable)} GB · ${t(hybridMode ? "capacity.minimum" : "capacity.required")} ${gpuNeedBytes ? `${GBNeed(gpuNeedBytes)} GB` : "—"}`}>
              {ticks.map((_, i) => (
                <span key={i} className={`tick${i < gpuVisibleLayers ? " tick--on" : ""}`} />
              ))}
            </div>
          </div>
          {hybridMode ? (
            <div className="capacity__resource">
              <span className="capacity__resource-kind">{t("node.ram")}</span>
              <div className="capacity__resource-stats">
                <span className="capacity__resource-stat">
                  <span>{t("capacity.available")}</span>
                  <strong>{GBHave(ramExpert)} GB</strong>
                </span>
                <span className="capacity__resource-stat">
                  <span>{t("capacity.expertStorage")}</span>
                  <strong className={short && (!hybridNeed || ramExpert < hybridNeed.ramNeedBytes) ? "capacity__gap" : ""}>
                    {hybridNeed ? `${GBNeed(hybridNeed.ramNeedBytes)} GB` : "—"}
                  </strong>
                </span>
              </div>
              <div className="spine capacity__spine" role="img" aria-label={`${t("node.ram")} · ${t("capacity.available")} ${GBHave(ramExpert)} GB · ${t("capacity.expertStorage")} ${ramNeedBytes ? `${GBNeed(ramNeedBytes)} GB` : "—"}`}>
                {ticks.map((_, i) => (
                  <span key={i} className={`tick${i < ramVisibleLayers ? " tick--on" : ""}`} />
                ))}
              </div>
            </div>
          ) : null}
        </div>
        {/* The bars are the quickest thing on this card to read; one short line
            below them names the combined outcome.
            The shortfall side no longer distinguishes one machine from a pool
            (2026-09-01): the head line above already quotes have/need for the
            selected context, and the caveats it used to carry (estimate vs
            runtime admission) said more than the moment needs. */}
        <p className={`capacity__verdict${short ? " capacity__verdict--no" : ""}`}>
          {hybridMode && !hybridNeed
            ? t(layoutState === "loading" ? "spine.hybridReading"
              : props.weights?.needs ? "spine.hybridDownload"
              : "spine.hybridUnavailable")
            : missingReport
            ? t("spine.unknown")
            : short
              ? t("spine.no")
              : hybridMode
                ? t("spine.hybridFits")
              : clustered
                ? t("spine.clusterFits", { n: peers.length })
                : t("spine.fits")}
        </p>
      </div>
    </section>
  );
}

// ---- engine diagnostics card (P1) ------------------------------------------
// The native engine is a separate process the client supervises (philosophy
// 17). This card is READ-ONLY: live state, crash/refusal reasons and the last
// log lines, for troubleshooting. Starting and stopping belong to the cluster
// flows (they attach a model path and tuning); crash → backoff restarts are
// the supervisor's job and surface here via events.
const LOG_TAIL = 6;

function EngineCard() {
  const { t, tErr } = useI18n();
  const [st, setSt] = useState<EngineStatus | null>(null);
  const [lines, setLines] = useState<EngineLogLine[]>([]);

  useEffect(() => {
    const eng = getEngineProvider();
    let live = true;
    eng.status().then((s) => live && setSt(s)).catch(() => {});
    eng.logs(LOG_TAIL).then((ls) => live && setLines(ls)).catch(() => {});
    const unStatus = eng.onStatus((s) => setSt(s));
    const unLog = eng.onLog((l) => setLines((prev) => [...prev.slice(-(LOG_TAIL - 1)), l]));
    return () => {
      live = false;
      unStatus();
      unLog();
    };
  }, []);

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
        {/* Read-only on purpose (2026-08-15): the engine is started and
            stopped by the cluster flows with a model path and tuning attached.
            The manual "start engine" button was a relic of the pre-llamacpp
            architecture — a bare worker with no model does nothing, and the
            button just looked broken. */}
        <div className="engine-card__head">
          <span className="engine-hint">{t("engine.advancedNote")}</span>
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
  const { t, lang } = useI18n();
  const stats = props.stats;
  if (!stats) return null;
  const totalTokens = stats.input_tokens + stats.output_tokens;
  const exact = (value: number) => value.toLocaleString(lang === "zh" ? "zh-CN" : "en-US");
  const up = stats.uptime_s;
  const uptimeLabel =
    up >= 86400
      ? t("stats.days", { n: Math.floor(up / 86400) })
      : up >= 3600
        ? t("stats.hours", { n: Math.floor(up / 3600) })
        : t("stats.mins", { n: Math.max(1, Math.floor(up / 60)) });
  return (
    <div className="activity">
      <span className="activity__item" title={exact(stats.requests)}>
        <b>{compactCount(stats.requests, lang)}</b> {t("stats.requests")}
      </span>
      <span className="activity__item" title={exact(totalTokens)}>
        <b>{compactCount(totalTokens, lang)}</b> {t("stats.tokens")}
      </span>
      {stats.last_tok_per_s > 0 ? (
        <span className="activity__item">
          <b>{stats.last_tok_per_s.toFixed(1)}</b> tok/s
        </span>
      ) : null}
      {(stats.cached_tokens ?? 0) > 0 ? (
        <span
          className="activity__item"
          title={`${t("stats.cacheTitle")}: ${exact(stats.cached_tokens!)}`}
        >
          {t("stats.cache")} <b>{compactCount(stats.cached_tokens!, lang)}</b> tok
        </span>
      ) : null}
      {/* The granted window + KV dtype, straight from the engine: context and
          KV precision are automatic (ctx-kv-simplification), so this readback
          is where the user sees what the machine decided. KV shown only when
          quantized — "f16" is the wordless default. */}
      {(stats.ctx_size ?? 0) > 0 ? (
        <span className="activity__item" title={t("stats.ctxTitle")}>
          {t("stats.ctx")}{" "}
          <b>
            {stats.ctx_size! >= 1048576
              ? `${Math.round(stats.ctx_size! / 1048576)}M`
              : `${Math.round(stats.ctx_size! / 1024)}K`}
          </b>
          {(() => {
            // Disclose BOTH halves when an explicit measurement override makes
            // them differ. Automatic tiers keep K/V uniform, but the readback
            // must still describe what actually runs rather than assume that.
            const k = stats.kv_cache_k;
            const v = stats.kv_cache_v || k;
            const quantized = (x?: string) => !!x && x !== "f16" && x !== "bf16";
            if (!quantized(k) && !quantized(v)) return "";
            return k === v ? ` · KV ${k}` : ` · KV ${k}/${v}`;
          })()}
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
  /** Does the selected model+precision fit THIS machine alone, by the MEASURED
   *  budget? Drives which deployment gets the primary button — nothing else.
   *  Both entries stay on screen and both stay clickable either way; only the
   *  visual weight follows the fact.
   *
   *  This was pinned to "cluster" from 2026-09-01 to 2026-09-02, because the
   *  local-fit verdict was then a closed-form ESTIMATE that measured 8.2x low
   *  on GLM-5.2 — not something to steer a user with. The need side is measured
   *  now (results/memory-need-measured-20260901.md), so the verdict is worth
   *  following. Undefined = unknown = leave cluster primary. */
  fitsStandalone?: boolean;
  // A hardware/backend fact, not a capacity estimate. Unsupported compute
  // hardware remains a hard gate; an estimated memory shortfall does not.
  canServeStandalone?: boolean;
  onServeStandalone?: () => void;
  // Weight presence and download state belong to the selected model, above the
  // two deployment choices. Rendering this inside each choice made one transfer
  // look like two independent jobs.
  weights?: WeightsInfo;
  onCreate: () => void;
  onNeedLogin: () => void;
  onManage: () => void;
  onLeave: () => Promise<void>;
  // The LOCAL setting, used only to detect disagreement with what the cluster
  // reports it is serving. Never used as the displayed value.
  settingModelId: string;
  settingModelLabel: string;
  settingQuant: string;
  ctxTokens: ContextTier;
  onCtxTokensChange: (ctx: ContextTier) => void;
  // Save a pick before any cluster is running.
  onSwitchModel: (modelId: string, quant: string) => void;
  engineStatus?: EngineStatus | null;
}) {
  const { t, tErr } = useI18n();
  const [copiedApi, setCopiedApi] = useState(false);
  const [copiedCode, setCopiedCode] = useState(false);
  const [pickOpen, setPickOpen] = useState(false);
  // Start/leave failures used to be unhandled rejections: the button did
  // nothing on screen. One operation strip keeps both actions honest.
  const [opErr, setOpErr] = useState<string | null>(null);
  const [leaving, setLeaving] = useState(false);
  const [startRequested, setStartRequested] = useState(false);
  const snap = props.pair;
  useEffect(() => {
    if (!snap?.canStart || snap.phase !== "idle") setStartRequested(false);
  }, [snap?.canStart, snap?.phase]);
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
  // `phase=ready` is stronger and fresher than a previously polled stats
  // snapshot: the native pairing layer only publishes it after
  // /cluster/status itself reports engine_state=ready. A stats request can be
  // delayed behind a long generation (the coordinator deliberately serves a
  // bounded number of requests), so retaining its earlier "starting" value
  // here used to leave a loading warning on a cluster that was already
  // answering chat. Never let the weaker observation contradict the stronger
  // one.
  const engineState = snap?.phase === "ready" ? "ready" : stats?.engine_state;
  const startupActive = !!snap && snap.phase !== "idle" && snap.phase !== "ready";
  const startupLabel = snap?.phase === "probing"
    ? t("pairing.phase.probing")
    : snap?.phase === "splitting"
      ? t("pairing.phase.splitting")
      : props.engineStatus?.state === "restarting" || engineState === "restarting"
        ? t("startup.restarting")
        : props.engineStatus?.state === "starting"
          ? t("startup.launching")
          : t("startup.loadingModel");
  const anyError = active && snap.peers.some((p) => p.stage === "error");
  const canServe = props.canServeStandalone !== false;
  // One readiness gate for both deployment paths. `path` matters as well as
  // `needs`: those states start as ""/false while the first disk probe is in
  // flight, and treating false as ready briefly enabled cluster creation on a
  // fresh install. A missing `weights` prop is kept compatible with fixtures
  // that do not exercise the download surface.
  const weightsReady = !props.weights
    || (!!props.weights.path && !props.weights.needs && !props.weights.dl);
  // Single machine leads when it can actually hold the model: it is faster (no
  // RPC hop), simpler, and strictly better for privacy since nothing leaves the
  // machine. Hard constraint #1 says single-machine users are the majority and
  // should not pay the clustering tax; making them press the secondary button
  // to get the simpler path was exactly that tax.
  const localLeads = props.fitsStandalone === true && canServe;
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
          {props.settingQuant ? <span className="cluster-model__quant">{fmtQuant(props.settingQuant)}</span> : null}
          {/* Nothing is running yet, so this pick is free: it writes the
              setting and the two options below re-read it. Paired, the switch
              handler ignores picks (a cluster is rebuilt around one model), so
              the link is replaced by the reason instead of a pick that goes
              nowhere (2026-09-06). */}
          {active ? (
            <span className="cluster-model__paired">{t("model.change.paired")}</span>
          ) : (
            <button className="linkbtn cluster-model__change" onClick={() => setPickOpen((v) => !v)}>
              {t("model.change")}
            </button>
          )}
          {pickOpen ? (
            <ModelPicker
              modelId={props.settingModelId}
              quant={props.settingQuant}
              running={null}
              onPick={props.onSwitchModel}
              onClose={() => setPickOpen(false)}
            />
          ) : null}
          {/* Downloading is a property of the selected model+precision, not of
              either deployment path. Keep one control and one progress bar
              here so local and cluster never narrate the same transfer twice. */}
          {props.weights && (props.weights.needs || props.weights.dl) ? (
            <WeightsRow w={props.weights} idle="show" />
          ) : null}
        </div>

        {/* Context window (2026-09-02, docs/ctx-tiers-2026-09.md). Its own row
            under the model because it is the third thing that defines what will
            be served, and because it is the one with a cost the user cannot see
            otherwise: llama.cpp allocates the whole window at load, so this is
            pre-paid VRAM whether a session uses it or not. It replaced a "1M"
            checkbox — a checkbox could only say "long or not" and could not show
            that the default had moved from 256K to 128K.

            Tiers above the model's ceiling are not rendered at all rather than
            rendered-and-clamped: an option that silently becomes a different
            number is worse than an absent one. */}
        <div className="cluster-model cluster-model--ctx" role="radiogroup"
             aria-label={t("model.contextWindow")}>
          <span className="cluster-model__label">{t("model.contextWindow")}</span>
          {contextTiersFor(props.settingModelId).map((tier) => (
            <label
              key={tier}
              className={`ctxtier${props.ctxTokens === tier ? " is-on" : ""}`}
            >
              <input
                type="radio"
                name="ctx-tier"
                checked={props.ctxTokens === tier}
                onChange={() => props.onCtxTokensChange(tier)}
              />
              <span>{ctxLabel(tier)}</span>
            </label>
          ))}
        </div>

        {/* Two ways to deploy, always both on screen. They used to be one
            either/or row driven by fitsStandalone, which meant a machine big
            enough to go solo was never offered "join someone else's cluster",
            and a machine too small never saw the local option at all — the
            path you can't take should say why, not disappear. */}
        <div className="deploy-opt">
          <div className="deploy-opt__text">
            <h3 className="deploy-opt__title">{t("deploy.local")}</h3>
          </div>
          <button
            className={localLeads ? "btn-primary" : "btn-secondary"}
            // Serving needs the weights already here. The selected-model row
            // above owns downloading, so this button waits rather than
            // duplicating that action or its progress. Capacity is deliberately
            // NOT a client-side disable condition: the coordinator performs the
            // authoritative GPU admission for the exact selected context.
            disabled={!canServe || !weightsReady}
            onClick={props.onServeStandalone}
          >
            {t("cluster.serveLocal")}
          </button>
          {/* No shortfall sentence here (2026-09-01): the capacity card beside
              this one already says whether the VRAM is short, and runtime
              admission is the authoritative answer either way. Repeating it
              under the button only made the choice look forbidden. */}
        </div>

        <div className="deploy-opt">
          <div className="deploy-opt__text">
            <h3 className="deploy-opt__title">{t("deploy.cluster")}</h3>
          </div>
          <div className="deploy-opt__actions">
            {/* One button: "Create a cluster" opens the pairing dialog, which
                already offers joining with a code — a second button for the
                same dialog's other tab was noise (removed 2026-08-15). */}
            <button
              className={localLeads ? "btn-secondary" : "btn-primary"}
              disabled={!weightsReady}
              title={!weightsReady ? t("pairing.needsModel") : undefined}
              onClick={props.onCreate}
            >
              {t("cluster.create")}
            </button>
          </div>
          {/* Still no "recommended" badge and no explanatory copy: the emphasis
              swap is the whole signal. Both paths remain one click away, and
              the coordinator's runtime admission is still the authority — this
              only stops the UI from pointing at the slower path when the
              measured budget says the simpler one works. */}
        </div>

        {/* The capability table (A-P1-3) was here until 2026-08-21 (Settings →
            Models already carries it), and the platform usage ranking
            (PopularModels) until 2026-08-25 — removed on the user's call: the
            ranking lives on the portal home page, not in the client. */}
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
        {snap.peers.length === 1 ? (
          <button
            className="linkbtn cluster-head__manage"
            disabled={leaving}
            onClick={() => {
              setOpErr(null);
              setLeaving(true);
              void props.onLeave()
                .catch((e) => setOpErr(tErr(String(e))))
                .finally(() => setLeaving(false));
            }}
          >
            {t("pairing.leave")}
          </button>
        ) : (
          <button className="linkbtn cluster-head__manage" onClick={props.onManage}>
            {t("cluster.manage")}
          </button>
        )}
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
          {/* Engine-reported precision first; when it is blank (older
              coordinator, or a file name outside the variant table) fall back
              to the precision THIS client launched with — first-hand
              knowledge, but only while the served id matches the setting. */}
          {(served.quant || (served.id === props.settingModelId ? props.settingQuant : "")) ? (
            <span className="cluster-model__quant">{fmtQuant(served.quant || props.settingQuant)}</span>
          ) : null}
          {/* Running deployments are read-only. Changing model requires an
              explicit cluster exit followed by a fresh start, so no restart
              shortcut is offered here. */}
        </div>
      ) : null}
      {startupActive ? (
        <StartupProgress
          label={startupLabel}
          detail={snap.peers.length > 1 ? t("startup.clusterDetail") : t("startup.localDetail")}
        />
      ) : null}
      {/* Inference-engine health (v2, llamacpp mode): mirrored from the
          coordinator's /health via stats. Chat answers 503 until "ready", so a
          quiet panel over a 503ing API would be a lie; absent on the legacy
          path, where per-peer stages carry the same news. */}
      {engineState && engineState !== "ready" && !startupActive ? (
        <p
          className={`cluster-hint ${
            engineState === "failed" ? "cluster-hint--bad" : "cluster-hint--warn"
          }`}
        >
          {t(`cluster.engine.${engineState}` as const)}
          {(stats?.engine_restarts ?? 0) > 0
            ? ` · ${t("cluster.engine.restarts", { n: stats?.engine_restarts ?? 0 })}`
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
                : p.modelReady === false
                  ? t("pairing.model.preparing")
                  : t(`pairing.stage.${p.stage}` as const)}
            </span>
          </div>
        ))}
      </div>

      {/* On a joining machine, preparing the creator-selected full GGUF is an
          automatic background step, but it is not invisible. The same row
          shows progress, a concrete failure, and a retry; once complete the
          native roster publishes modelReady and the creator unlocks Start. */}
      {snap.phase === "idle" && snap.peers.some((p) => p.self && p.modelReady === false) && props.weights ? (
        <WeightsRow w={props.weights} idle={props.weights.needs ? "show" : "hide"} />
      ) : null}

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

      {snap.phase === "idle" && snap.peers.some((p) => p.modelReady === false) ? (
        <p className="cluster-hint">{t("pairing.model.waiting")}</p>
      ) : null}

      {snap.canStart ? (
        <div className="cluster-start-actions">
          <button
            className="btn-primary btn-block cluster-start"
            disabled={startRequested}
            onClick={() => {
              setOpErr(null);
              setStartRequested(true);
              void getPairingProvider()
                // The roster may have been created before the adjacent switch
                // changed. Refresh only overflow at the moment of launch.
                .start(false, props.weights?.path, overflowTuning(loadSettings()))
                .catch((e) => {
                  setStartRequested(false);
                  setOpErr(tErr(String(e)));
                });
            }}
          >
            {startRequested
              ? t("pairing.phase.starting")
              : `${t("pairing.startCluster", { n: snap.peers.length })} →`}
          </button>
        </div>
      ) : null}
      {opErr ? <p className="cluster-hint cluster-hint--bad">{opErr}</p> : null}

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
          <ActivityRow stats={stats} />
        </div>
      ) : null}

      {/* No "share this cluster" entry (2026-08-15): it deep-linked into the
          platform settings, which are hidden until the platform side is live
          for users. It returns with them. */}
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
// NOTE (2026-08-15): with the open intake removed nothing sets `localEngine`
// yet, so this card is dormant. It stays because it is the llamacpp
// single-machine surface (WS-B1: verdict/manifest latching, engine health) —
// the curated single-machine path rewires onto it when it moves off the
// legacy pairing flow.
function LocalEngineCard(props: {
  label: string; // the file being served
  api: ClusterApi | null;
  engStatus: EngineStatus | null;
  settingModelId: string;
  settingQuant: string;
  onSwitchModel: (modelId: string, quant: string) => void;
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
        {/* Same precision rule as the cluster card's serving row: engine-
            reported first, launch setting as the fallback only when the ids
            agree — this card can serve an arbitrary local GGUF, where the
            setting's precision would be a guess about someone else's file. */}
        {(stats?.quant || (stats?.model === props.settingModelId ? props.settingQuant : "")) ? (
          <span className="cluster-model__quant">{fmtQuant(stats?.quant || props.settingQuant)}</span>
        ) : null}
        <button className="linkbtn cluster-model__change" onClick={() => setPickOpen((v) => !v)}>
          {t("model.change")}
        </button>
        {pickOpen ? (
          <ModelPicker
            modelId={props.settingModelId}
            quant={props.settingQuant}
            running={{ modelId: props.settingModelId, quant: "", machines: 1 }}
            onPick={props.onSwitchModel}
            onClose={() => setPickOpen(false)}
          />
        ) : null}
      </div>

      {/* Download progress for an HF-sourced GGUF lives next to the card that
          started it. Only live progress renders here — the resting "not
          downloaded" states belong to the download manager in Settings. */}
      {props.weights?.dl ? <WeightsRow w={props.weights} idle="hide" /> : null}

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
  ctxTokens: ContextTier;
  onCtxTokensChange: (ctx: ContextTier) => void;
  pair: PairingSnapshot | null;
  /** Local llama.cpp engine (open-GGUF serving) — replaces the cluster card
   *  while it runs; this machine IS the whole deployment. */
  localEngine: { gguf: string; label: string } | null;
  localApi: ClusterApi | null;
  localStats: ClusterStats | null;
  engStatus: EngineStatus | null;
  onStopLocal: () => void;
  onServeStandalone: () => void;
  onCreateCluster: () => void;
  onNeedLogin: () => void;
  onJoinCluster: () => void;
  onManageCluster: () => void;
  onLeaveCluster: () => Promise<void>;
  /** Jump to the full model section in Settings (the cluster card's picker
   *  covers the common case; Settings still owns weights paths and the rest). */
  onSwitchModel: (modelId: string, quant: string) => void;
  weights?: WeightsInfo;
}) {
  const { t, tErr } = useI18n();
  const s = props.snap;
  // No hypothetical cluster: before pairing this card compares this machine
  // with a one-node requirement. Once paired, use the real roster count. The
  // old fallback of three added two imaginary engine-overhead allocations and
  // made a 256K estimate look larger without any machines to justify it.
  const nNodes = props.pair && props.pair.peers.length > 0 ? props.pair.peers.length : 1;
  // Does the model fit THIS machine alone (N=1)? Only the primary-button
  // emphasis reads it; both deployment entries stay clickable regardless, and
  // the coordinator still performs the authoritative admission. Same function,
  // same backend and same measured workspace the capacity card renders, so the
  // card cannot say "fits" while the buttons point the other way.
  const standaloneRamExpert = s.ram_expert_usable ?? (s.os === "windows" ? 0 : s.ram_usable);
  const standalone = estimateClusterCapacity(props.model, s, props.tier.ctx, 1,
                                             props.quant, backendOfOs(s.os),
                                             moeRamExpertBudget(props.model.id, standaloneRamExpert,
                                                                s.unified_memory));
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
              onSwitchModel={props.onSwitchModel}
              onStop={props.onStopLocal}
              weights={props.weights}
            />
          ) : (
          <ClusterCard
            pair={props.pair}
            fitsStandalone={standalone.gapBytes === 0}
            canServeStandalone={(s.hw_status ?? HW_OK) === HW_OK}
            onServeStandalone={props.onServeStandalone}
            weights={props.weights}
            onCreate={props.onCreateCluster}
            onNeedLogin={props.onNeedLogin}
            onManage={props.onManageCluster}
            onLeave={props.onLeaveCluster}
            settingModelId={props.model.id}
            settingModelLabel={props.model.label}
            settingQuant={props.quant}
            ctxTokens={props.ctxTokens}
            onCtxTokensChange={props.onCtxTokensChange}
            onSwitchModel={props.onSwitchModel}
            engineStatus={props.engStatus}
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
            peers={props.pair?.peers}
            weights={props.weights}
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
  // (The settings deep-link state left with its last caller, 2026-08-26 —
  // the "download in Settings" hand-off is gone; SettingsPanel still accepts
  // initialCategory for any future deep link.)
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
  // platform.ts drops the stored session when the gateway answers 401 (token
  // expired). React state does not follow localStorage on its own — without
  // this listener the top bar keeps showing a signed-in account whose every
  // request fails, which is exactly the trap the 401 handling exists to break.
  useEffect(() => {
    const onExpired = () => {
      setSession(null);
      setShowAuth(true);
    };
    window.addEventListener("idletoken:session-expired", onExpired);
    return () => window.removeEventListener("idletoken:session-expired", onExpired);
  }, []);
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

  // The local llama.cpp engine this client started (llamacpp_serve). Not
  // persisted: the sidecar dies with the client, so a fresh launch starts
  // clean. Dormant since the open intake was removed (2026-08-15) — nothing
  // sets it until the curated single-machine path moves onto llamacpp mode
  // (see LocalEngineCard's note); the plumbing stays because that move is the
  // architecture's direction (coordinator drives llama-server, no RPC solo).
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

  // The provider switch is a STANDING choice: turned on once, it holds across
  // launches. Until 0.1.10 nothing restarted the agent after a client restart,
  // so the panel showed "on" over a machine that had quietly stopped earning.
  // Errors go to the console only: the resume has no owner watching it, and
  // the sharing panel's own status line is where a broken agent is explained.
  useEffect(() => {
    resumeSharingAgent()
      .then((r) => { if (r === "started") console.info("sharing agent resumed"); })
      .catch((e) => console.error("sharing agent resume:", e));
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
  // The gguf file the current selection resolves to — the key that ties the
  // selection to its entry in the download maps below.
  const [selFile, setSelFile] = useState("");
  /**
   * Downloads, keyed by gguf file name (2026-08-15 redesign). Settings is a
   * download manager: every model row downloads, cancels and reports
   * independently, and none of it is tied to which model is *selected*. The
   * previous design had ONE app-wide download slot bound to the selection,
   * which is how "download A, then switch and download B" killed both.
   * The file name is the natural key: unique per model+precision, and the same
   * identity the engine's one-writer-per-file guard uses (weights.rs).
   */
  const [dls, setDls] = useState<Record<string, NonNullable<WeightsInfo["dl"]>>>({});
  /**
   * Why the last attempt for a file stopped — a footnote on its row, never a
   * state. A download that did not finish leaves the machine exactly where it
   * was (without the weights), and the next attempt resumes from the .part.
   */
  const [dlErrors, setDlErrors] = useState<Record<string, string>>({});
  const refreshWeights = useCallback(async () => {
    try {
      const r = await resolveLocalWeights({
        modelDir: settings.modelDir,
        manifest: getManifest(settings.modelId),
        quant: settings.quant,
      });
      setWeightsPath(r.path);
      setNeedsWeights(r.needsDownload);
      setPartialBytes(r.haveBytes);
      setSelFile(resolveDownload(getManifest(settings.modelId), settings.quant)?.file ?? "");
    } catch {
      // A failed probe must not block the UI: treat it as "needs downloading", and
      // the user gets the real error when they press download.
      setWeightsPath("");
      setNeedsWeights(true);
      setPartialBytes(0);
    }
  }, [settings.modelDir, settings.modelId, settings.quant]);

  useEffect(() => { void refreshWeights(); }, [refreshWeights]);

  // Ids the user has cancelled. The engine only notices the cancel flag between
  // reads, and a read blocks until the next chunk arrives — on a slow endpoint
  // that is seconds away. Waiting for it meant the row sat there showing
  // progress after the user pressed Cancel, then flashed a message. The click
  // is the decision; the UI acts on it now and drops whatever that download
  // says afterwards.
  const cancelled = useRef<Set<string>>(new Set());

  /** Files with a fetch currently in flight — the double-click guard readable
   *  from callbacks (state would be stale there). */
  const activeFetches = useRef<Set<string>>(new Set());
  // Every caller awaiting the same file shares one promise. A joiner's
  // automatic cluster preparation can begin while the user-visible download
  // row is already fetching that file; returning false to the second caller
  // used to make it report a failure even though the transfer was healthy.
  const fetchTasks = useRef<Map<string, Promise<boolean>>>(new Map());

  /** Everything on disk may have changed: re-probe the selected model.
   *  (The per-row Settings re-probe went with the download manager,
   *  2026-08-26 — the cluster card is the one weights surface now.) */
  const bumpWeights = useCallback(() => {
    void refreshWeights();
  }, [refreshWeights]);

  /**
   * Start one file's download and see it through. Returns true when the file
   * is complete, false when it stopped (cancel or error — the row's footnote
   * says which). Failures are reported on the row and in the problem log here,
   * so callers do not each invent their own reporting.
   */
  const startDownload = useCallback(
    (file: string, target: DownloadTarget): Promise<boolean> => {
      if (!file) return Promise.resolve(false);
      const existing = fetchTasks.current.get(file);
      if (existing) return existing;
      const task = (async () => {
        activeFetches.current.add(file);
        const dir = settings.modelDir || (await defaultModelDir());
        setDlErrors((m) => {
          if (!(file in m)) return m;
          const n = { ...m };
          delete n[file];
          return n;
        });
        setDls((m) => ({ ...m, [file]: { have: 0, total: target.expectBytes } }));
        try {
          await fetchWeights({ id: file, target, destDir: dir });
          return true;
        } catch (e) {
          if (!isWeightsCancelled(e)) {
            const msg = String(e);
            setDlErrors((m) => ({ ...m, [file]: msg }));
            recordProblem({
              at: new Date().toISOString(),
              kind: "download",
              message: msg,
              detail: { file },
            });
          }
          return false;
        } finally {
          activeFetches.current.delete(file);
          fetchTasks.current.delete(file);
          setDls((m) => {
            const n = { ...m };
            delete n[file];
            return n;
          });
          bumpWeights();
        }
      })();
      fetchTasks.current.set(file, task);
      return task;
    },
    [settings.modelDir, bumpWeights]
  );

  /** Cancel one file's download. Its `.part` is kept; the next attempt resumes. */
  const cancelDownloadFor = useCallback((file: string) => {
    cancelled.current.add(file);
    setDls((m) => {
      const n = { ...m };
      delete n[file];
      return n;
    });
    void cancelFetch(file);
  }, []);

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
    // A cancel is the user's own decision — nothing to report. Everything else
    // goes to the problem log (Settings → Data & about); download failures
    // additionally carry their own per-row footnote via dlErrors.
    if (isWeightsCancelled(e)) return;
    recordProblem({
      at: new Date().toISOString(),
      kind,
      message: String(e),
      detail: { model: settings.modelId, quant: settings.quant },
    });
  }, [settings.modelId, settings.quant]);

  // What the selected model's download row is showing right now, readable from
  // the UI-test directives — those run in a mount-time effect and would
  // otherwise see the first render's value forever.
  const errRef = useRef<string | null>(null);
  useEffect(() => {
    errRef.current = dlErrors[selFile] ?? null;
  });

  // Progress event subscription — one for the whole application; each event
  // lands on its own file's entry. Downloads run independently of the model
  // selection, so nothing here cares which model is on screen.
  useEffect(() => {
    let un: (() => void) | null = null;
    let dead = false;
    onFetchProgress((p) => {
      const id = p.id;
      // The UI-test oracles run real downloads under this id; their progress
      // must not appear in the download manager.
      if (!id || id === "uitest") return;
      if (cancelled.current.has(id)) {
        // The download is winding down. Only its final word clears the mark —
        // anything before that (a last progress tick) is stale by definition.
        if (p.kind === "done" || p.kind === "error" || p.kind === "cancelled") {
          cancelled.current.delete(id);
          bumpWeights();
        }
        return;
      }
      if (p.kind === "progress" || p.kind === "probe") {
        // The event still says which endpoint is serving the bytes, but that
        // stays out of the row on purpose: where the download comes from is an
        // implementation detail the user is not asked to think about.
        setDls((m) => {
          const previous = m[id];
          const phaseChanged = p.phase !== undefined && p.phase !== previous?.phase;
          return {
            ...m,
            [id]: {
              have: p.have ?? previous?.have ?? 0,
              total: p.total ?? previous?.total ?? 0,
              // A note belongs to one event phase. Keeping the previous note
              // made "verifying" stick after the next GGUF part had resumed
              // downloading, which was both misleading and impossible for the
              // user to distinguish from a second hash pass. Other notices (for
              // example "server cannot resume") stay visible within their phase.
              note: p.note ?? (phaseChanged ? undefined : previous?.note),
              phase: p.phase ?? previous?.phase,
              phaseHave: p.phaseHave,
              phaseTotal: p.phaseTotal,
              part: p.part,
              parts: p.parts,
            },
          };
        });
      } else if (p.kind === "done" || p.kind === "cancelled" || p.kind === "error") {
        setDls((m) => {
          if (!(id in m)) return m;
          const n = { ...m };
          delete n[id];
          return n;
        });
        if (p.kind === "error") {
          setDlErrors((m) => ({ ...m, [id]: p.message ?? "download failed" }));
        }
        bumpWeights();
      }
    }).then((f) => (dead ? f() : (un = f)));
    return () => { dead = true; un?.(); };
  }, [bumpWeights]);

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
  // (The memory-shape argument engineTuning used for resolving the KV "auto"
  // dtype left with the KV selector, 2026-08-25: the coordinator decides the
  // dtype where the memory plan is made — see ctx-kv-simplification.)

  /** Resolve and integrity-check one exact curated model. Joining a cluster
   * may additionally fetch it: the creator chose the identity, so the joiner
   * prepares that same full GGUF automatically before it can report ready. */
  const prepareWeights = useCallback(async (
    over?: { modelId: string; quant: string },
    fetchMissing = false,
  ): Promise<string> => {
    const modelId = over?.modelId ?? settings.modelId;
    const quant = over?.quant ?? settings.quant;
    const manifest = getManifest(modelId);
    let r = await resolveLocalWeights({
      modelDir: settings.modelDir,
      manifest,
      quant,
    });
    if ((r.needsDownload || !r.path) && fetchMissing) {
      const target = r.target ?? resolveDownload(manifest, quant);
      if (!target || !(await startDownload(target.file, target))) {
        throw new Error(`[WEIGHTS_NOT_DOWNLOADED] ${modelId}${quant ? ` ${quant}` : ""}`);
      }
      // The download command verifies every part before returning. Resolve
      // again instead of manufacturing a path so split GGUF completeness and
      // the marker files remain one source of truth.
      r = await resolveLocalWeights({ modelDir: settings.modelDir, manifest, quant });
    }
    if (r.needsDownload || !r.path) {
      // "[CODE] detail" — localized by tErr (ERROR_KEYS in i18n.ts).
      throw new Error(`[WEIGHTS_NOT_DOWNLOADED] ${modelId}${quant ? ` ${quant}` : ""}`);
    }
    // Complete on disk but never hash-checked (script download, or a client
    // from before the integrity gate): verify NOW, before any engine sees the
    // file. On mismatch the engine deletes it and rejects — re-probe so the
    // UI flips back to "needs downloading" instead of offering a file that no
    // longer exists.
    if (r.needsVerify && r.target) {
      try {
        const dir = settings.modelDir || (await defaultModelDir());
        // A split model verifies file by file: each part carries its own hash
        // and marker, and a part already verified is a fast marker hit.
        const files = [
          { file: r.target.file, sha256: r.target.sha256 },
          ...r.target.parts.map((p) => ({ file: p.file, sha256: p.sha256 ?? "" })),
        ];
        for (const f of files) {
          await verifyWeights({ id: f.file, destDir: dir, file: f.file, sha256: f.sha256 });
        }
      } catch (e) {
        bumpWeights();
        throw e;
      }
    }
    return r.path;
  }, [settings.modelDir, settings.modelId, settings.quant, bumpWeights, startDownload]);

  /** Creator/standalone preflight: creation never starts a surprise transfer.
   * The cluster card owns the explicit download control and progress. */
  const ensureWeights = useCallback(
    (over?: { modelId: string; quant: string }) => prepareWeights(over, false),
    [prepareWeights]
  );

  /** Joiner preparation: exact creator-selected model, automatic download. */
  const ensureClusterWeights = useCallback(
    (over: { modelId: string; quant: string }) => prepareWeights(over, true),
    [prepareWeights]
  );

  /**
   * Fetch the exact model a cluster demanded when it refused this machine, and
   * hand back everything the retrying join needs to describe itself.
   *
   * This replaces the effect that used to do the same work AFTER joining
   * (removed 2026-09-01, when admission started requiring the weights). That
   * effect can no longer fire — a machine without the weights never reaches the
   * roster, so `peers.find(p => p.self).modelReady` is true for every member
   * there is — and leaving it in would be a dormant path that starts a
   * multi-gigabyte transfer nobody asked for if a member ever appeared unready
   * again.
   *
   * The cluster's identity overwrites this machine's own selection: the two
   * disagreeing is precisely what earned the refusal, and the settings have to
   * be saved before the join so every other surface names the same model.
   *
   * `tuning` is built from `next`, not from React state, because the join goes
   * out in the same turn: state has not committed, and a stale tuning would
   * send the OLD model id and earn a second refusal for the model we just
   * finished downloading.
   */
  const prepareClusterModel = useCallback(
    async (modelId: string, quant: string) => {
      const next = { ...loadSettings(), modelId, quant };
      setSettings(next);
      saveSettings(next);
      const modelPath = await ensureClusterWeights({ modelId, quant });
      return { modelPath, tuning: engineTuning(next, caps) };
    },
    [ensureClusterWeights, caps]
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
   * Rebuilds are serialized, and stale ones are skipped: two rebuilds running
   * concurrently race each other through leave/create/start on the pairing
   * provider. Each pick from the Cluster page's picker bumps the sequence; a
   * queued rebuild that is no longer the newest does not run — the last pick
   * wins, with one line of teardown.
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

  // The selected model's one weight state. The Cluster page renders it once,
  // directly under the model choice; deployment buttons consume readiness but
  // do not each grow their own copy of the download control.
  const weightsInfo = useMemo<WeightsInfo>(
    () => ({
      needs: needsWeights,
      path: weightsPath,
      dl: dls[selFile] ?? null,
      partialBytes,
      lastError: dlErrors[selFile] ?? null,
      onDownload: () => {
        const target = resolveDownload(getManifest(settings.modelId), settings.quant);
        if (target) void startDownload(target.file, target);
      },
      onCancel: () => cancelDownloadFor(selFile),
    }),
    [needsWeights, weightsPath, dls, dlErrors, selFile, partialBytes, cancelDownloadFor, startDownload, settings.modelId, settings.quant]
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
      // Tuning comes from STORAGE, not from the `settings` React state: the
      // The top-right Request help button saves through saveSettings() without going through this
      // component's state, so the state can be minutes stale by the time the
      // engine starts. Read at spawn time, or "turn Request help on, then start
      // the cluster" launches a coordinator with no overflow flags — seen
      // live on a Windows compute node (2026-08-21): the panel promised the borrow settings
      // would apply on the next engine start, and the next start ignored them.
      await getPairingProvider().create({
        hostname: snap.hostname,
        gpu: snap.gpu_name,
        modelPath: path,
        tuning: engineTuning(over ? { ...loadSettings(), ...over } : loadSettings(), caps),
      });
      // allowSolo: this IS the one-machine flow. Without it the engine's
      // 2-machine pairing floor rejects the start and the button dies after
      // downloading the weights.
      await getPairingProvider().start(true, path, overflowTuning(loadSettings()));
      return true;
    } catch (e) {
      // Cancelling the weights download cancels serving too — that is the same
      // decision, not a second failure to report.
      if (!isWeightsCancelled(e)) console.error("serve-standalone:", e);
      reportWeightsError(e, "cluster");
      return true; // it ran; it failed. Distinct from "could not run yet".
    }
  }, [snap, ensureWeights, settings, caps, reportWeightsError]);

  /** Save a model choice only when no pairing roster is active. A running or
   * forming cluster is deliberately immutable: the user exits it first, then
   * chooses a model and starts a fresh deployment. `localEngine` is the
   * dormant open-GGUF path rather than a pairing cluster, so it keeps its own
   * explicit stop/restart behaviour until that legacy surface is retired. */
  const switchModel = useCallback(
    async (modelId: string, quant: string) => {
      if (pairSnap && pairSnap.peers.length > 0) {
        // A paired cluster is rebuilt around one model; the page says so where
        // the change link would be (2026-09-06: this used to return silently,
        // and the pick looked like it did nothing).
        console.warn("model switch ignored: paired with", pairSnap.peers.length, "peer(s)");
        return;
      }
      // The setting is written immediately — the radio/picker must reflect the
      // choice now, not after whatever rebuild is currently winding down.
      updateSettings({
        ...settings,
        modelId,
        quant,
        // A model with a lower ceiling cannot honour the stored tier; snap the
        // selection down to the largest window this model really offers rather
        // than sending one the planner has no measurement for.
        ctxTokens: contextTiersFor(modelId).includes(settings.ctxTokens)
          ? settings.ctxTokens
          : contextTiersFor(modelId)[contextTiersFor(modelId).length - 1],
      });
      return queueRebuild(async () => {
        // Switching away from a running local llama.cpp engine: stop it, then
        // serve the curated pick the single-machine way. Pairing clusters never
        // reach this branch because of the guard above.
        if (localEngine) {
          await stopLocalEngine();
          await serveStandalone({ modelId, quant });
        }
      });
    },
    [settings, pairSnap, localEngine, stopLocalEngine, serveStandalone, queueRebuild]
  );

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);

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

  // The dashboard probe is deliberately uncapped: "available VRAM" means the
  // GPU's real current free memory, not the user's IdleToken usage limit. The
  // limit is still passed to every engine launch through engineTuning(); it
  // simply no longer falsifies the hardware reading or the reference estimate.
  // We keep the old snapshot visible during a manual re-probe so only the first
  // load shows a spinner.
  useEffect(() => {
    let live = true;
    setError(null);
    getResourceProvider()
      .probe()
      .then((s) => {
        if (!live) return;
        setSnap(s);
        setTotals({ vram_total: s.vram_total, ram_total: s.ram_total });
        // Hand this machine's real free GPU memory to the pairing layer, which
        // sends it with every join. Each machine measures its own, so the
        // roster can total the pool and answer "does the model fit on all of
        // us" BEFORE anyone presses Start — that used to be answerable only by
        // the coordinator refusing after the fact. Reported in every mode: the
        // numbers have to be in place before a join is sent.
        if (inTauri()) {
          void import("@tauri-apps/api/core")
            .then(({ invoke }) =>
              invoke("pairing_report_memory", {
                vramFree: s.vram_usable,
                ramFree: s.ram_usable,
                ramExpertFree: s.ram_expert_usable ?? (s.os === "windows" ? 0 : s.ram_usable),
                unifiedMemory: s.unified_memory,
              })
            )
            .catch(() => {
              /* pre-pairing or an older backend: the pool total just stays
                 incomplete, which the card reports as "cannot tell" */
            });
        }
        // First-start selection (B3): the default model used to be hardcoded to
        // DSv4-Flash (80.76 GiB), so the first thing a new user with an 8 GB card
        // saw was "does not fit on one machine, please build a cluster" -- with one
        // machine to their name. Given a real probe, recommend **the largest model
        // this machine can actually run** instead.
        // Only when settings have never been saved: a choice the user made
        // themselves must not be overwritten.
        if (firstRun.current && !settingsWerePersisted()) {
          firstRun.current = false;
          const ctx = effectiveCtx(settings);
          const pick = pickBestFittingModel(s, ctx);
          if (pick.modelId !== settings.modelId || pick.quant !== settings.quant) {
            const next = { ...settings, modelId: pick.modelId, quant: pick.quant };
            setSettings(next);
            saveSettings(next);
          }
        }
      })
      .catch((e) => live && setError(String(e?.message ?? e)));
    return () => {
      live = false;
    };
  }, [nonce]);

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
            {/* Chat owns live stream listeners and request ids. It must remain
                mounted while the user visits Cluster or Settings; conditionally
                rendering it orphaned the visible transcript while the native
                request kept its engine slot and GPU work alive. */}
            <Chat
              visible={!error && !!snap && view === "chat"}
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
            />
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
            ) : view === "settings" ? (
              <SettingsPanel
                asPage
                settings={settings}
                onChange={updateSettings}
                snap={snap}
                theme={theme}
                onTheme={setTheme}
                lang={lang}
                onLang={setLang}
                session={session}
                onSignIn={() => setShowAuth(true)}
                onWeightsChanged={bumpWeights}
                onClose={() => setView("cluster")}
              />
            ) : view === "cluster" ? (
              <Dashboard
                snap={snap}
                model={model}
                quant={settings.quant}
                tier={{ id: settings.tier || 2, ctx: effectiveCtx(settings) } as Tier}
                ctxTokens={settings.ctxTokens}
                onCtxTokensChange={(ctx) => updateSettings({ ...settings, ctxTokens: ctx })}
                pair={pairSnap}
                localEngine={localEngine}
                localApi={localApi}
                localStats={localStats}
                engStatus={engStatus}
                onStopLocal={() => void stopLocalEngine()}
                // Wrapped, not passed through: serveStandalone takes an
                // optional {modelId,quant} override, and a bare handler would
                // hand it the click event as that argument.
                onServeStandalone={() => void serveStandalone()}
                onCreateCluster={() => {
                  setPairingView("choose");
                  setShowPairing(true);
                }}
                onNeedLogin={() => setShowAuth(true)}
                onJoinCluster={() => {
                  setPairingView("join");
                  setShowPairing(true);
                }}
                onManageCluster={() => {
                  setPairingView("choose");
                  setShowPairing(true);
                }}
                onLeaveCluster={() => getPairingProvider().leave()}
                onSwitchModel={(id, q) => void switchModel(id, q)}
                weights={weightsInfo}
              />
            ) : null}
          </div>
        </div>
      </div>
      {/* No floating download bar. The selected model row owns its download;
          deployment choices only consume the resulting ready state. */}
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
            // From storage, not state — same staleness as serveStandalone.
            tuning: engineTuning(loadSettings(), caps),
          }}
          session={session}
          initialView={pairingView}
          prepareSelectedModel={() => ensureWeights()}
          prepareClusterModel={prepareClusterModel}
          onSignIn={() => {
            setShowPairing(false);
            setShowAuth(true);
          }}
          onClose={() => setShowPairing(false)}
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
