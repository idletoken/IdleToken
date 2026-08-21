import { useEffect, useMemo, useRef, useState } from "react";
import { useI18n, type Lang } from "./i18n";
import { EndpointsPanel } from "./EndpointsPanel";
import { MODEL_BRANDS, shortModelLabel, hasQuantChoice, quantOptions, defaultQuant, getManifest } from "./models";
import { resolveDownload, weightsState, defaultModelDir, type DownloadTarget } from "./weights";
import { inTauri } from "./platform";
import Capability from "./Capability";
import { fmtBytes } from "./format";
import {
  APP_VERSION,
  DEFAULT_SETTINGS,
  TIERS,
  effectiveCaps,
  loadSettings,
  saveSettings,
  type AppSettings,
  type ResourcePreset,
} from "./settings";
import type { NodeSnapshot } from "./types";
import type { Session } from "./auth";
import WeightsRow from "./WeightsRow";
import StoredModels from "./StoredModels";
import { getEngineProvider } from "./provider/engine";
import { exportableSettings } from "./diagnostics";
import PlatformPanel from "./PlatformPanel";

type Theme = "dark" | "light";
const MiB = 1024 ** 2;
const GiB = 1024 ** 3;

// Bilingual label, co-located with the schema (avoids ~200 i18n keys at this scale).
type Bi = { en: string; zh: string };
const L = (b: Bi, lang: Lang) => b[lang];

// "problems" / "links" / "legal" were block types here until 2026-08-21; the
// pages that used them (Problems & diagnostics, the About link list, the legal
// documents) were removed, so the block types went with them rather than stay
// as three renderer branches nothing can reach.
type FieldType =
  | "toggle" | "select" | "text" | "password" | "number" | "slider" | "time" | "note" | "action"
  /** Full-width block: the model folder's contents, with a Delete per file. */
  | "stored-models";
interface Field {
  key?: keyof AppSettings;
  type: FieldType;
  label?: Bi;
  hint?: Bi;
  reserved?: boolean;
  options?: { value: string; label: Bi }[];
  /** select whose stored value is a number (option values still written as strings). */
  numeric?: boolean;
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  placeholder?: string;
  action?: "export" | "import" | "clearData" | "clearKv" | "checkUpdate";
  showIf?: (s: AppSettings) => boolean;
}
interface Section {
  label?: Bi;
  fields: Field[];
}
interface Category {
  id: string;
  label: Bi;
  note?: Bi;
  bespoke?: "quick" | "platform" | "endpoints" | "resources";
  /** Only render this category when the predicate holds (the platform console
   *  needs a build that has a platform to talk to). */
  visible?: () => boolean;
  sections?: Section[];
}

// ---- small controls -------------------------------------------------------
function Toggle(props: { checked: boolean; onChange: (v: boolean) => void; label: string; disabled?: boolean }) {
  return (
    <button
      role="switch"
      aria-checked={props.checked}
      aria-label={props.label}
      disabled={props.disabled}
      className={`switch${props.checked && !props.disabled ? " is-on" : ""}${props.disabled ? " is-disabled" : ""}`}
      onClick={() => !props.disabled && props.onChange(!props.checked)}
    >
      <span className="switch__knob" />
    </button>
  );
}
function Segmented<T extends string>(props: { options: { value: T; label: string }[]; value: T; onChange: (v: T) => void }) {
  return (
    <div className="segmented" role="tablist">
      {props.options.map((o) => (
        <button key={o.value} role="tab" aria-selected={o.value === props.value} className={`segmented__opt${o.value === props.value ? " is-on" : ""}`} onClick={() => props.onChange(o.value)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}
function CapSlider(props: { label: string; noCap: string; totalBytes: number; valueMb: number; onChange: (mb: number) => void }) {
  const totalMb = Math.max(1, Math.round(props.totalBytes / MiB));
  const pos = props.valueMb === 0 ? totalMb : Math.min(props.valueMb, totalMb);
  const display = props.valueMb === 0 ? props.noCap : `${((props.valueMb * MiB) / GiB).toFixed(1)} GiB`;
  return (
    <div className="setting-row">
      <div className="setting-row__head">
        <span className="setting-row__k">{props.label}</span>
        <span className="setting-row__v">{display}</span>
      </div>
      <input className="slider" type="range" min={1024} max={totalMb} step={512} value={pos} onChange={(e) => { const v = Number(e.target.value); props.onChange(v >= totalMb ? 0 : v); }} />
    </div>
  );
}

/**
 * One model's weight state + download control — the unit of the download
 * manager (2026-08-15 split: Settings downloads, Cluster starts/switches,
 * Chat displays). The cell probes its own file on disk (re-probing when
 * `version` bumps) and renders the shared WeightsRow with THIS row's progress
 * and error, so every model downloads independently of every other row and of
 * whichever model is selected.
 */
function ModelWeightsCell(props: {
  file: string;
  target: DownloadTarget;
  modelDir: string;
  version: number;
  dl: { have: number; total: number; note?: string } | null;
  lastError: string | null;
  onStart: () => void;
  onCancel: () => void;
}) {
  const [st, setSt] = useState<{ needs: boolean; partial: number } | null>(null);
  useEffect(() => {
    let live = true;
    (async () => {
      // Browser dev build: no filesystem bridge; report "ready" so the page
      // stays usable (same policy as resolveLocalWeights).
      if (!inTauri()) {
        if (live) setSt({ needs: false, partial: 0 });
        return;
      }
      try {
        const dir = props.modelDir || (await defaultModelDir());
        const w = await weightsState(dir, props.file, props.target.expectBytes, props.target.sha256);
        if (live) setSt({ needs: !w.complete, partial: w.complete ? 0 : w.have_bytes });
      } catch {
        if (live) setSt({ needs: true, partial: 0 });
      }
    })();
    return () => {
      live = false;
    };
    // A settled download (or a deletion) bumps `version`, re-running the probe.
  }, [props.file, props.modelDir, props.version, props.target.expectBytes]);
  if (!st && !props.dl) return null;
  return (
    <WeightsRow
      w={{
        needs: st?.needs ?? true,
        path: "",
        dl: props.dl,
        partialBytes: st?.partial ?? 0,
        lastError: props.lastError,
        onDownload: props.onStart,
        onCancel: props.onCancel,
      }}
      idle="show"
    />
  );
}

// ---- schema (everything except the bespoke Models page) --------------------
// Reorganized 2026-08-15 along scenario lines (the same split as the pages:
// Settings downloads, Cluster runs, Chat talks):
//   Models        everything about model files on THIS machine — the download
//                 manager, precision, the capability table, the folder.
//   Cluster & API pairing, the API service, inference/cache knobs, and how to
//                 connect a client.
// "Advanced (coming soon)" is GONE, not moved: it did nothing a user could
// feel, and a settings page must not show controls that do nothing
// (principle 15).
//
// "Sharing & earnings" is BACK (2026-08-20, audit A-P1-2). It was removed on
// the same reasoning while the marketplace was not live. The marketplace IS
// live now — balance, ledger, providers and the lend/borrow switch are all real
// platform calls — and the client was the only place with no way in: you could
// not see your balance, list this cluster, or turn sharing on from the app that
// runs the cluster. The category renders `PlatformPanel`, which has been intact
// and unreachable the whole time.
const CATEGORIES: Category[] = [
  // The model page: download manager, capability table, storage, plus the
  // model-adjacent runtime knobs (context tier, resource caps). Language and
  // theme moved to Appearance (2026-08-15) — they were the one group here
  // that had nothing to do with models.
  { id: "quick", label: { en: "Models", zh: "模型" }, bespoke: "quick" },
  // Its own page (2026-08-15, was a group at the bottom of Models): how much
  // of this machine a cluster may use is a machine-level decision, not a
  // model-level one — it applies to whatever model runs.
  { id: "resources", label: { en: "Resource usage", zh: "资源占用" }, bespoke: "resources" },
  {
    id: "appearance",
    label: { en: "Appearance", zh: "外观" },
    sections: [{ fields: [
      { key: "accent", type: "select", label: { en: "Accent color", zh: "强调色" }, options: [
        { value: "amber", label: { en: "Electric", zh: "电光蓝" } }, { value: "teal", label: { en: "Teal", zh: "青绿" } },
        { value: "violet", label: { en: "Violet", zh: "紫" } }, { value: "rose", label: { en: "Rose", zh: "玫红" } } ] },
      // Was a 0.8–1.4 slider until 2026-08-11. Dragging it re-zoomed the whole
      // document on every 0.05 tick — the page strobed and the slider moved out
      // from under the cursor. Auto (follow the window) is the default now; the
      // manual escape hatch is four discrete steps, one relayout per choice.
      { key: "uiScale", type: "select", numeric: true, label: { en: "UI scale", zh: "界面缩放" }, options: [
        { value: "0", label: { en: "Auto", zh: "自动" } },
        { value: "0.9", label: { en: "Small", zh: "小" } },
        { value: "1", label: { en: "Standard", zh: "标准" } },
        { value: "1.15", label: { en: "Large", zh: "大" } },
        { value: "1.3", label: { en: "Extra large", zh: "特大" } } ] },
      { key: "density", type: "select", label: { en: "Density", zh: "密度" }, options: [
        { value: "comfortable", label: { en: "Comfortable", zh: "宽松" } }, { value: "compact", label: { en: "Compact", zh: "紧凑" } } ] },
      { key: "reduceMotion", type: "toggle", label: { en: "Reduce motion", zh: "减少动效" } },
    ] }],
  },
  {
    // Split from the old "Cluster & API" (2026-08-15): serving an API on this
    // machine and forming a cluster with other machines are different jobs
    // with different audiences, and one page mixing beacon ports with access
    // tokens read as noise. API = how a client on this machine connects;
    // Networking = how machines find each other and pair.
    id: "connect",
    bespoke: "endpoints",
    label: { en: "API", zh: "API" },
    sections: [
      { fields: [
        // No access-token field (2026-08-15): the API answers this machine
        // only (loopback, coord-enforced), so a token gates nothing a local
        // caller could not already do. The AppSettings key stays for engine
        // compatibility; the UI no longer offers it.
        // The "changes apply on restart" hint went with the rest of this page's
        // prose on 2026-08-21 — the page is the address and the port, nothing
        // else.
        { key: "apiPort", type: "number", label: { en: "Port", zh: "端口" } },
      ] },
    ],
  },
  // A-P1-2: the marketplace console. Hidden when this build has no platform
  // configured — an empty console is worse than no entry — but present the
  // moment there is one, signed in or not (the panel's own Guide explains what
  // to do next, which is what an entry that leads somewhere is for).
  {
    id: "platform",
    bespoke: "platform",
    label: { en: "Sharing & earnings", zh: "共享与收益" },
    visible: () => !!loadSettings().platformUrl.trim(),
  },
  {
    id: "network",
    label: { en: "Networking", zh: "联机组网" },
    sections: [
      // All consumed by src-tauri/src/pairing.rs. "LAN auto-discovery" is a
      // UDP broadcast beacon (it was never mDNS).
      //
      // "Only same subnet" and "Bind interface / IP" were removed from the UI
      // on 2026-08-15 (too complex for the audience — the user's call). The
      // AppSettings keys still exist and still reach the engine: subnet-only
      // stays at its default (off), bindNic at auto. ⚠ bindNic was the manual
      // escape for machines where a VPN adapter (Tailscale/Clash TUN) wins the
      // route — if that class of "machine online but unreachable" reports
      // returns, the fix is auto-detecting overlay adapters, not re-adding the
      // field. Settings import still applies both keys for hand-edited files.
      { label: { en: "Pairing & discovery", zh: "组网与发现" }, fields: [
        { key: "clusterName", type: "text", label: { en: "Cluster name", zh: "集群名" } },
        { key: "mdns", type: "toggle", label: { en: "LAN auto-discovery", zh: "局域网自动发现" } },
        { key: "discoveryPort", type: "number", label: { en: "Discovery port", zh: "发现端口" } },
        { key: "manualPeers", type: "text", label: { en: "Manual peer IPs", zh: "手动节点 IP" }, placeholder: "192.168.1.50, 192.168.1.51" },
        { key: "heartbeatSec", type: "number", label: { en: "Heartbeat (s)", zh: "心跳（秒）" } },
        { key: "preferCoordinator", type: "toggle", label: { en: "Prefer this machine as coordinator", zh: "优先本机作协调者" } },
        { key: "interStagePort", type: "number", label: { en: "Inter-stage port", zh: "节点间端口" } },
      ] },
    ],
  },
  {
    id: "privacy",
    label: { en: "Privacy", zh: "隐私保护" },
    // One line, twice over: cut to the mechanism on 2026-08-15, grown back to
    // four notes plus the legal documents on 2026-08-20 (audit A-P1-5), cut
    // back to the mechanism on 2026-08-21 (user's call). Prose assurances on a
    // privacy page read as marketing, and every claim beyond the mechanism
    // itself is another thing the page can be wrong about. The full story lives
    // in docs/privacy-design.md — layer 0 pinned to the coordinator, PSK-TLS
    // between machines with no plaintext fallback, loopback-only API — and each
    // of those is enforced in code whether or not this page recites it.
    // ⚠ The terms/privacy-policy buttons (`{ type: "legal" }`) went with them.
    // They were the only way into those documents from the app, and the
    // marketplace charges credits — if that turns out to be a requirement
    // rather than a preference, the LegalDocs component is still in this file.
    sections: [
      { fields: [
        { type: "note", label: {
          en: "Encryption: X25519 + AES-256-GCM envelope encryption.",
          zh: "加密方式：X25519 + AES-256-GCM 信封加密。" } },
        // "Share anonymous telemetry" was removed on 2026-08-13. There is no
        // telemetry client anywhere in the tree — not in the client, the engine
        // or the platform agent — so the switch never had a consumer. A privacy
        // page is the worst place to keep a placeholder: turning it off is what
        // the careful reader does, and it bought them nothing. When something
        // is actually collected, the switch comes back WITH the thing it gates.
      ] },
    ],
  },
  // Background residency + updates. Its own place rather than a corner of
  // "Data & about": on a machine that contributes to a cluster, "does closing
  // the window stop serving?" and "will it come back after a reboot?" are
  // among the first questions the product has to answer.
  {
    id: "system",
    label: { en: "Startup & updates", zh: "启动与更新" },
    sections: [
      { label: { en: "Window & tray", zh: "窗口与托盘" }, fields: [
        { key: "trayIcon", type: "toggle", label: { en: "Show tray icon", zh: "显示托盘图标" } },
        { key: "closeToTray", type: "toggle", label: { en: "Keep running when I close the window", zh: "关闭窗口后继续在后台运行" },
          showIf: (s) => s.trayIcon },
        { key: "startMinimized", type: "toggle", label: { en: "Start in the tray", zh: "启动后隐藏到托盘" },
          showIf: (s) => s.trayIcon },
        { key: "rememberWindow", type: "toggle", label: { en: "Remember window size and position", zh: "记住窗口大小与位置" } },
      ] },
      { label: { en: "Startup", zh: "开机启动" }, fields: [
        { key: "autostart", type: "toggle", label: { en: "Launch at login", zh: "开机自动启动" } },
      ] },
      { label: { en: "Updates", zh: "更新" }, fields: [
        { key: "autoUpdate", type: "toggle", label: { en: "Check for updates automatically", zh: "自动检查更新" } },
        { key: "updateChannel", type: "select", label: { en: "Update channel", zh: "更新通道" }, options: [
          { value: "stable", label: { en: "Stable", zh: "稳定版" } }, { value: "beta", label: { en: "Beta", zh: "测试版" } } ] },
        { type: "action", action: "checkUpdate", label: { en: "Check for updates now", zh: "立即检查更新" } },
      ] },
    ],
  },
  {
    id: "data",
    label: { en: "Data & about", zh: "数据与关于" },
    sections: [
      // "Log level" and "Enable experimental features" were removed on
      // 2026-08-13: nothing read either of them. The log level was the more
      // misleading of the two — it travelled INTO the diagnostics bundle, so a
      // bundle could say `logLevel: "debug"` about an engine that had never
      // been asked to run that way, and send whoever read it looking for logs
      // that were never produced.
      // "Data folder" removed with them: the client's data lives in this
      // machine's app-data directory (Tauri decides it) and in localStorage,
      // and no code ever read the box. Typing a path did not move anything.
      { label: { en: "Data & backup", zh: "数据与备份" }, fields: [
        { type: "action", action: "export", label: { en: "Export settings", zh: "导出设置" },
          hint: { en: "Keys and tokens are left out of the file.", zh: "导出文件不包含密钥与令牌。" } },
        { type: "action", action: "import", label: { en: "Import settings", zh: "导入设置" } },
        { type: "action", action: "clearData", label: { en: "Clear all local data", zh: "清除全部本地数据" } },
      ] },
      // "Problems & diagnostics" (the local failure log + the export button) is
      // gone as of 2026-08-21 (user's call). It was hidden on 2026-08-15, back
      // on 2026-08-20 for audit A-P1-5, and off again now. The machinery is
      // untouched — problems.ts still records failures, diagnostics.ts still
      // assembles the bundle, and the acceptance channel still exports one —
      // so this is a UI decision, not a capability removed. What it costs: a
      // user whose cluster will not start has no in-app way to hand over the
      // evidence, so a support thread starts from a description instead of a
      // file.
      { label: { en: "About", zh: "关于" }, fields: [
        { type: "note", label: { en: `IdleToken client ${APP_VERSION}`, zh: `IdleToken 客户端 ${APP_VERSION}` } },
        // Said wrong until 2026-08-13 ("MIT-licensed"). IdleToken is
        // **Apache-2.0** — LICENSE, NOTICE and the README have always said so.
        // The third-party inventory (llama.cpp, ds4, both MIT) was cut from
        // this line on 2026-08-15: the NOTICE file that ships with the build is
        // the attribution Apache-2.0 §4(d) actually requires, and an About box
        // that re-lists it is a second copy to keep true.
        { type: "note", label: { en: "Apache-2.0.", zh: "Apache-2.0 许可。" } },
        // The outbound link list (guide / source / issues / security / notices)
        // was cut on 2026-08-21 with the rest of this page's extras. links.ts
        // still holds the addresses; "request a model" now has no in-app route,
        // which is worth remembering because the curated list is the only way
        // to get a model and a GitHub issue is the only way into it.
      ] },
    ],
  },
];

// The "Advanced · coming soon" roadmap shelf is deleted outright (2026-08-15;
// it had been hidden from the nav since 2026-08-11). AppSettings keys and
// defaults for its reserved fields are unchanged, so nothing migrates.
function visibleCategories(): Category[] {
  return CATEGORIES.filter((c) => !c.visible || c.visible());
}

/** Where the engine keeps the KV cache when the field is left empty, per
 *  platform — matching worker_main.c's own choice (LOCALAPPDATA on Windows,
 *  XDG_CACHE_HOME/HOME elsewhere). Shown as a placeholder only; nothing here
 *  is written to settings. */
function defaultKvHint(): string {
  const ua = typeof navigator !== "undefined" ? navigator.userAgent : "";
  if (/Windows/i.test(ua)) return "%LOCALAPPDATA%\\IdleToken\\kv";
  if (/Mac OS X|Macintosh/i.test(ua)) return "~/Library/Caches/idletoken/kv";
  return "~/.cache/idletoken/kv";
}

// AboutLinks (guide / source / issues / security / notices) and LegalDocs (the
// platform's terms and privacy policy, fetched and rendered in a modal) both
// lived here until 2026-08-21, when the About and Privacy pages were cut back
// to a version line, a licence line and the encryption line. Both are in git
// history; links.ts still exports the addresses and the gateway still serves
// /legal/tos and /legal/privacy, so bringing either back is a paste, not a
// rewrite. The modal's styles are still in styles.css (.modal--legal).

// ---- panel -----------------------------------------------------------------
export default function SettingsPanel(props: {
  settings: AppSettings;
  onChange: (s: AppSettings) => void;
  snap: NodeSnapshot;
  theme: Theme;
  onTheme: (t: Theme) => void;
  lang: Lang;
  onLang: (l: Lang) => void;
  onClose: () => void;
  // Sharing & earnings needs the cloud session (and a way to get one).
  session?: Session | null;
  onSignIn?: () => void;
  // Deep link: "share this cluster →" must land on the sharing category, not
  // on whatever was open last. Applied on mount only — after that the nav owns
  // the selection, so navigating away inside settings isn't fought.
  initialCategory?: string | null;
  // ---- the download manager (2026-08-15 split) ----
  // Settings owns downloading and ONLY downloading: every model row shows its
  // own weight state with its own download/cancel, independent of which model
  // is selected. Selection and switching live on the Cluster page; chatting
  // shows the model read-only. App owns the progress-event subscription, so
  // the live pieces arrive as props keyed by gguf file name.
  downloads?: Record<string, { have: number; total: number; note?: string }>;
  downloadErrors?: Record<string, string>;
  /** Bumped by App whenever a download settles or weights are deleted — every
   *  row re-probes the disk. */
  weightsVersion?: number;
  onStartDownload?: (file: string, target: DownloadTarget) => void;
  onCancelDownload?: (file: string) => void;
  /** Something changed the model folder (the stored-models list calls it after
   *  a deletion) — App re-probes and bumps weightsVersion. */
  onWeightsChanged?: () => void;
  /** Run an update check the user asked for. App owns it because every outcome
   *  — found / already current / could not check — is shown in the same
   *  dialog, and that dialog belongs above this panel. */
  onCheckUpdate?: () => Promise<void> | void;
  // Sidebar IA: settings is a PLACE now, rendered inline in the content area.
  // (The modal wrapper remains only for potential embedded reuse.)
  asPage?: boolean;
  // The diagnostics bundle asks for the current cluster state. When an address is
  // given, use the one the cluster actually advertises; when it is not (this
  // machine is not the coordinator, or no cluster is running) fall back to the
  // local API port -- in that case "cannot connect" is itself a useful fact.
  apiBaseUrl?: string | null;
}) {
  const { t, lang } = useI18n();
  // Recomputed per render: the platform category appears the moment a
  // platform URL is set, without a reload.
  const cats = visibleCategories();
  const [active, setActive] = useState(
    props.initialCategory && cats.some((c) => c.id === props.initialCategory) ? props.initialCategory : "quick"
  );
  const [query, setQuery] = useState("");
  // "Clear my cache now" feedback (philosophy 15: every action has clear
  // loading/success/failure feedback).
  const [kvClear, setKvClear] = useState<"idle" | "busy" | "ok" | "err">("idle");
  // The button says "checking" while the feed is being asked; the ANSWER is
  // the dialog App opens, so there is nothing to report back here.
  const [upd, setUpd] = useState<"idle" | "busy">("idle");
  const fileRef = useRef<HTMLInputElement>(null);
  const s = props.settings;
  // Which model card is unfolded in the download manager. Clicking a card is
  // browsing — "show me this model's precisions" — and nothing more; what RUNS
  // is chosen on the Cluster page. Defaults to the model in use.
  const [openModelId, setOpenModelId] = useState<string>(s.modelId);
  // Per-model precision being VIEWED (and downloaded) — pure page state, never
  // written to settings.
  const [viewQuant, setViewQuant] = useState<Record<string, string>>({});
  const set = <K extends keyof AppSettings>(k: K, v: AppSettings[K]) => props.onChange({ ...s, [k]: v });
  const setCap = (k: "maxVramMb" | "maxRamMb", v: number) => props.onChange({ ...s, [k]: v, resourcePreset: "custom" });

  // Model-folder editing. The draft is committed on blur/Enter rather than per
  // keystroke — see the field for why. It re-syncs when the stored value
  // changes underneath us (the picker below, or an imported settings file).
  const [dirDraft, setDirDraft] = useState(s.modelDir);
  useEffect(() => setDirDraft(s.modelDir), [s.modelDir]);
  // What an empty setting resolves to, shown as the placeholder so the field
  // says where the weights actually go instead of looking unconfigured.
  const [defaultDir, setDefaultDir] = useState("");
  useEffect(() => {
    if (!inTauri()) return;
    let live = true;
    defaultModelDir().then((d) => live && setDefaultDir(d)).catch(() => {});
    return () => {
      live = false;
    };
  }, []);

  /** Native folder picker. Typing a path is still allowed (pasting one is
   *  normal), but on Windows — where this matters most, because the weights go
   *  on D: — hand-typed paths are the failure. */
  const pickModelDir = async () => {
    if (!inTauri()) return;
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const picked = await open({ directory: true, multiple: false, defaultPath: s.modelDir || defaultDir || undefined });
      if (typeof picked === "string" && picked) set("modelDir", picked);
    } catch {
      /* cancelled or unavailable: the text field remains the way in */
    }
  };
  // The caps actually in force — the same function App feeds to the probe and
  // to the engine, so the panel can never quote a limit the engine is not
  // given. Under a preset these are a fraction of this machine's totals; under
  // "custom" they are the slider values verbatim.
  const liveCaps = effectiveCaps(s, { vram_total: props.snap.vram_total, ram_total: props.snap.ram_total });

  const runAction = (a: Field["action"]) => {
    if (a === "export") {
      // A-P2-1: through the same allowlist the diagnostics bundle uses.
      // "Export settings" wrote the file the user then mails to someone, and
      // it used to carry `apiToken` and `overflowKey` in the clear with no
      // warning — the exact credentials diagnostics.ts was written to keep out
      // of a shared file. One allowlist, both exits.
      const blob = new Blob([JSON.stringify(exportableSettings(s), null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "idletoken-settings.json";
      link.click();
      URL.revokeObjectURL(url);
    } else if (a === "checkUpdate") {
      if (upd === "busy") return;
      setUpd("busy");
      void Promise.resolve(props.onCheckUpdate?.()).finally(() => setUpd("idle"));
    } else if (a === "import") {
      fileRef.current?.click();
    } else if (a === "clearKv") {
      if (kvClear === "busy") return;
      setKvClear("busy");
      getEngineProvider()
        .clearKvCache(s.kvDir)
        .then(() => setKvClear("ok"))
        .catch(() => setKvClear("err"))
        .finally(() => setTimeout(() => setKvClear("idle"), 2500));
    } else if (a === "clearData") {
      if (confirm(lang === "zh" ? "确定清除本机全部 IdleToken 数据？" : "Clear all IdleToken data on this machine?")) {
        localStorage.clear();
        props.onChange({ ...DEFAULT_SETTINGS });
        saveSettings(DEFAULT_SETTINGS);
        location.reload();
      }
    }
  };
  const onImportFile = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    f.text().then((txt) => {
      try {
        const parsed = JSON.parse(txt);
        props.onChange({ ...DEFAULT_SETTINGS, ...parsed });
      } catch {
        /* ignore malformed */
      }
    });
    e.target.value = "";
  };

  // search: flat list of matching fields across categories
  const q = query.trim().toLowerCase();
  const searchHits = useMemo(() => {
    if (!q) return null;
    const hits: { cat: Category; field: Field }[] = [];
    for (const c of cats) {
      for (const sec of c.sections ?? []) {
        for (const f of sec.fields) {
          if (f.label && L(f.label, lang).toLowerCase().includes(q)) hits.push({ cat: c, field: f });
        }
      }
    }
    return hits;
  }, [q, lang]);

  const cat = cats.find((c) => c.id === active) ?? cats[0];

  const renderField = (f: Field, i: number) => {
    if (f.showIf && !f.showIf(s)) return null;
    const label = f.label ? L(f.label, lang) : "";
    const hint = f.hint ? L(f.hint, lang) : undefined;
    const tag = f.reserved ? <span className="reserved-tag">{t("settings.reservedTag")}</span> : null;

    if (f.type === "note") return <p key={i} className="about-line">{label}</p>;
    if (f.type === "stored-models")
      return <StoredModels key={i} modelDir={s.modelDir} onChanged={props.onWeightsChanged} />;
    if (f.type === "action") {
      let btnLabel = label;
      if (f.action === "checkUpdate" && upd === "busy") btnLabel = t("update.checking");
      if (f.action === "clearKv" && kvClear !== "idle") {
        btnLabel =
          kvClear === "busy"
            ? L({ en: "Clearing…", zh: "清除中…" }, lang)
            : kvClear === "ok"
              ? L({ en: "Cleared ✓", zh: "已清除 ✓" }, lang)
              : L({ en: "Clear failed", zh: "清除失败" }, lang);
      }
      return (
        <div key={i} className="setting-row setting-row--inline">
          <div className="setting-row__label">
            <span className="setting-row__k">{label}</span>
            {hint ? <span className="setting-row__hint">{hint}</span> : null}
          </div>
          <button className={`btn-secondary${f.action === "clearData" ? " btn-danger" : ""}${f.action === "clearKv" && kvClear === "err" ? " btn-danger" : ""}`} disabled={(f.action === "clearKv" && kvClear === "busy") || (f.action === "checkUpdate" && upd === "busy")} onClick={() => runAction(f.action)}>{btnLabel}</button>
        </div>
      );
    }

    let control: React.ReactNode = null;
    const key = f.key!;
    const val = s[key];
    // Reserved = not wired engine/OS-side yet: keep visible (roadmap honesty)
    // but disabled — an editable control that does nothing is a broken promise.
    const dis = !!f.reserved;
    if (f.type === "toggle") control = <Toggle checked={val as boolean} disabled={dis} onChange={(v) => set(key, v as AppSettings[typeof key])} label={label} />;
    else if (f.type === "select")
      control = (
        <select className="select" value={String(val)} disabled={dis} onChange={(e) => set(key, (f.numeric ? Number(e.target.value) : e.target.value) as AppSettings[typeof key])}>
          {f.options!.map((o) => <option key={o.value} value={o.value}>{L(o.label, lang)}</option>)}
        </select>
      );
    else if (f.type === "text" || f.type === "password")
      control = <input className="field__input field__input--inline" type={f.type === "password" ? "password" : "text"} value={val as string} placeholder={f.placeholder} disabled={dis} onChange={(e) => set(key, e.target.value as AppSettings[typeof key])} />;
    else if (f.type === "number")
      control = <input className="field__input field__input--num" type="number" value={val as number} disabled={dis} onChange={(e) => set(key, (Number(e.target.value) || 0) as AppSettings[typeof key])} />;
    else if (f.type === "time")
      control = <input className="field__input field__input--num" type="time" value={val as string} disabled={dis} onChange={(e) => set(key, e.target.value as AppSettings[typeof key])} />;
    else if (f.type === "slider")
      control = (
        <div className="slider-inline">
          <input className="slider" type="range" min={f.min} max={f.max} step={f.step} value={val as number} disabled={dis} onChange={(e) => set(key, Number(e.target.value) as AppSettings[typeof key])} />
          <span className="slider-inline__v">{(val as number).toFixed(f.step && f.step < 1 ? 2 : 0)}{f.unit ?? ""}</span>
        </div>
      );

    return (
      <div key={i} className="setting-row setting-row--inline">
        <div className="setting-row__label">
          <span className="setting-row__k">{label}{tag}</span>
          {hint ? <span className="setting-row__hint">{hint}</span> : null}
        </div>
        <div className="setting-row__control">{control}</div>
      </div>
    );
  };

  const renderQuick = () => (
    <>
      <div className="setting-group">
        <div className="setting-group__label">{t("settings.model")}</div>
        {/* The download manager (2026-08-15 split). Clicking a card unfolds it
            — that is browsing, not choosing what runs (the Cluster page owns
            that; the tag on a card only reports it). The unfolded card shows
            ONE precision dropdown and ONE weights row for the precision being
            viewed — every precision was listed flat here once, and a page of
            near-identical rows drowned the two facts that matter. Downloads
            keep running when the card folds or the dropdown moves; the header
            badge keeps them discoverable. */}
        <div className="model-list">
          {/* One card per BRAND (2026-08-15): a flat list of ~10 models, each
              with up to 26 precisions, ran off the screen. The choice is a
              funnel — whose model, then how big, then how precise — so the UI
              is one too: pick a brand card, pick a size inside it, pick the
              precision next to the size. */}
          {MODEL_BRANDS.map((brand) => {
            const inBrand = brand.models.some((m) => m.id === openModelId);
            const sel = inBrand
              ? brand.models.find((m) => m.id === openModelId)!
              : brand.models[0];
            const open = inBrand;
            const variants = hasQuantChoice(sel.id) ? quantOptions(sel.id) : [];
            const q =
              viewQuant[sel.id] ??
              (s.modelId === sel.id ? s.quant || defaultQuant(sel.id) : defaultQuant(sel.id));
            const target = resolveDownload(getManifest(sel.id), q);
            // "Something in this brand is downloading" — computed over every
            // size AND precision, so a collapsed card still says so.
            const busy = brand.models.some((m) =>
              (hasQuantChoice(m.id) ? quantOptions(m.id) : []).some(
                (v) => props.downloads?.[resolveDownload(getManifest(m.id), v.quant)?.file ?? ""]
              ) || props.downloads?.[resolveDownload(getManifest(m.id))?.file ?? ""]
            );
            const current = brand.models.find((m) => m.id === s.modelId);
            return (
              <div key={brand.id} className={`model-opt model-opt--rows${open ? " is-on" : ""}`}>
                <div
                  className="model-opt__head"
                  role="button"
                  tabIndex={0}
                  onClick={() => setOpenModelId(open ? "" : brand.models[0].id)}
                  onKeyDown={(e) =>
                    (e.key === "Enter" || e.key === " ") && setOpenModelId(open ? "" : brand.models[0].id)
                  }
                >
                  <span className="model-opt__name">{brand.label}</span>
                  {/* Singular matters now (2026-08-21): splitting the Qwen card
                      by generation left Qwen3 and Qwen3.8 with exactly one
                      model each, and "1 sizes" was suddenly on screen. */}
                  <span className="model-opt__params">
                    {brand.models.length}
                    {L(
                      brand.models.length === 1
                        ? { en: " size", zh: " 个规格" }
                        : { en: " sizes", zh: " 个规格" },
                      lang
                    )}
                  </span>
                  {current ? (
                    <span className="model-opt__deploy">{t("settings.model.current")}</span>
                  ) : null}
                  {busy && !open ? (
                    <span className="model-opt__deploy model-opt__deploy--busy">
                      {L({ en: "Downloading", zh: "下载中" }, lang)}
                    </span>
                  ) : null}
                </div>
                {open ? (
                  <>
                    {/* The sizes in this brand. One click switches which size
                        the precision row below is about; nothing is applied to
                        the cluster from here (that is the Cluster page's job). */}
                    <div className="model-sizes">
                      {brand.models.map((m) => (
                        <button
                          key={m.id}
                          className={`model-size${m.id === sel.id ? " is-on" : ""}`}
                          onClick={() => setOpenModelId(m.id)}
                        >
                          {/* Brand stripped (the card title already says it)
                              and the params suppressed when the name already
                              carries them — "Qwen3.5 4B" beside "4B" said the
                              same thing twice. */}
                          <span className="model-size__name">{shortModelLabel(m.label, brand.label)}</span>
                          {shortModelLabel(m.label, brand.label)
                            .replace(/\s+/g, "")
                            .toUpperCase()
                            .includes(m.params.split("·")[0].replace(/\s+/g, "").toUpperCase()) ? null : (
                            <span className="model-size__params">{m.params}</span>
                          )}
                        </button>
                      ))}
                    </div>
                    <div className="model-variant">
                      {variants.length > 0 ? (
                        <>
                          <span className="model-variant__q">{t("settings.precision")}</span>
                          <select
                            className="select"
                            value={q}
                            onChange={(e) => setViewQuant((v) => ({ ...v, [sel.id]: e.target.value }))}
                          >
                            {variants.map((v) => (
                              <option key={v.quant} value={v.quant}>
                                {v.quant} · {fmtBytes(v.layer_weight_bytes + v.shared_weight_bytes)}
                              </option>
                            ))}
                          </select>
                        </>
                      ) : target ? (
                        <span className="model-variant__size">{fmtBytes(target.expectBytes)}</span>
                      ) : null}
                      {target ? (
                        <ModelWeightsCell
                          file={target.file}
                          target={target}
                          modelDir={s.modelDir}
                          version={props.weightsVersion ?? 0}
                          dl={props.downloads?.[target.file] ?? null}
                          lastError={props.downloadErrors?.[target.file] ?? null}
                          onStart={() => props.onStartDownload?.(target.file, target)}
                          onCancel={() => props.onCancelDownload?.(target.file)}
                        />
                      ) : null}
                    </div>
                  </>
                ) : null}
              </div>
            );
          })}
        </div>
      </div>
      {/* Page order (2026-08-15, user-specified): models → context/performance
          → storage → inference & cache → the capability table LAST. The table
          is a summary verdict over everything set above it — context tier
          changes its "max context" column — so it reads best at the bottom. */}
      <div className="setting-group">
        <div className="setting-group__label">{t("settings.tier")}</div>
        <div className="tier-list">
          {TIERS.map((tier) => {
            const ctxLabel = tier.ctx >= 1048576 ? "1M" : `${tier.ctx / 1024}K`;
            return (
              <button key={tier.id} className={`tier-opt${s.tier === tier.id ? " is-on" : ""}`} onClick={() => set("tier", tier.id)}>
                <span className="tier-opt__name">{t(`tier.${tier.id}.name` as const)}</span>
                <span className="tier-opt__ctx">{ctxLabel}</span>
              </button>
            );
          })}
        </div>
      </div>
      {/* Storage: the folder the download manager writes into, and what it
          holds. Moved here from "Cluster & API" (2026-08-15 reorg) — it
          belongs next to the downloads it serves, not under cluster plumbing.

          Models run to tens of GB, so the folder is routinely NOT on the
          system disk and users keep weights on several. Two consequences,
          both handled here (2026-08-15): a native folder picker, because
          typing a Windows path by hand is where this goes wrong; and every
          weights probe on this page is scoped to THIS folder, so pointing it
          at another disk re-scans and finds what is already there. The wording
          across the page says "the model folder", never "this machine". */}
      <div className="setting-group">
        <div className="setting-group__label">{L({ en: "Storage", zh: "存储" }, lang)}</div>
        <div className="setting-row setting-row--inline">
          <div className="setting-row__label">
            <span className="setting-row__k">{L({ en: "Model folder", zh: "模型目录" }, lang)}</span>
          </div>
          <div className="setting-row__control setting-row__control--dir">
            <input
              className="field__input field__input--inline"
              type="text"
              value={dirDraft}
              placeholder={defaultDir || "~/.idletoken/models"}
              // Committed on blur / Enter, not per keystroke: every commit
              // re-probes every model row and re-scans the folder, and a
              // half-typed path ("D:\mod") is a folder that holds nothing —
              // so typing used to flash "not in the model folder" down the
              // whole list before you reached the end of the path.
              onChange={(e) => setDirDraft(e.target.value)}
              onBlur={() => dirDraft !== s.modelDir && set("modelDir", dirDraft)}
              onKeyDown={(e) => {
                if (e.key === "Enter") (e.target as HTMLInputElement).blur();
                if (e.key === "Escape") setDirDraft(s.modelDir);
              }}
            />
            <button className="btn-secondary btn-secondary--sm" onClick={() => void pickModelDir()}>
              {L({ en: "Browse…", zh: "选择目录…" }, lang)}
            </button>
          </div>
        </div>
        <div className="setting-row">
          <span className="setting-row__k">{L({ en: "Weights in this folder", zh: "该目录下的权重" }, lang)}</span>
        </div>
        <StoredModels modelDir={s.modelDir} onChanged={props.onWeightsChanged} />
      </div>
      {/* Moved from "Cluster & API" (2026-08-15 reorg): reply length and the
          KV cache are properties of running models, not of cluster plumbing.
          Rendered through the same field renderer the schema pages use. */}
      <div className="setting-group">
        <div className="setting-group__label">{L({ en: "Inference & cache", zh: "推理与缓存" }, lang)}</div>
        {renderField({ key: "maxTokens", type: "number", label: { en: "Max tokens per reply", zh: "单次回复最大词元数" } }, 0)}
        {/* A-P2-6: the placeholder used to suggest /tmp/idletoken-kv on every
            platform, which on Windows is a path that does not exist and on
            macOS is one the OS empties without warning. It now shows where the
            engine puts the cache on THIS machine, which is also what an empty
            field means — so the "empty = the engine's own location" hint under
            it was saying the placeholder over again (removed 2026-08-21). */}
        {renderField({ key: "kvDir", type: "text", label: { en: "KV cache directory", zh: "KV 缓存目录" },
                       placeholder: defaultKvHint() }, 1)}
        {renderField({ type: "action", action: "clearKv", label: { en: "Clear KV cache", zh: "清除 KV 缓存" } }, 2)}
      </div>
      {/* "What can I run?" — the summary verdict, last on purpose (see the
          order note above). Moved out of the model group 2026-08-15. */}
      <div className="setting-group">
        <Capability apiBaseUrl={props.apiBaseUrl ?? null} />
      </div>
    </>
  );

  // "Resource usage" — its own page since 2026-08-15 (was the bottom of the
  // Models page): how much of this machine a cluster may use applies to
  // whatever model runs, so it is machine configuration, not model choice.
  const renderResources = () => (
    <div className="setting-group">
      <Segmented<ResourcePreset>
        value={s.resourcePreset}
        options={[
          { value: "conservative", label: t("preset.conservative") },
          { value: "balanced", label: t("preset.balanced") },
          { value: "max", label: t("preset.max") },
          ...(s.resourcePreset === "custom" ? [{ value: "custom" as ResourcePreset, label: t("preset.custom") }] : []),
        ]}
        onChange={(v) => set("resourcePreset", v)}
      />
      {/* The sliders show the limit that is IN EFFECT, not the stored custom
          number. They are not a second, separate cap — they are the precise
          form of the control above, and under a preset the value comes from
          that preset (a share of this machine's total). Showing `s.maxVramMb`
          here meant the default install said "Balanced" and "No limit" in the
          same box, one line apart, while 75% was what the engine got.
          Dragging either one takes over as a custom limit (setCap). */}
      <CapSlider label={t("settings.maxVram")} noCap={t("settings.noCap")} totalBytes={props.snap.vram_total} valueMb={liveCaps.maxVramMb} onChange={(mb) => setCap("maxVramMb", mb)} />
      <CapSlider label={t("settings.maxRam")} noCap={t("settings.noCap")} totalBytes={props.snap.ram_total} valueMb={liveCaps.maxRamMb} onChange={(mb) => setCap("maxRamMb", mb)} />
    </div>
  );

  const inner = (
    <div className={props.asPage ? "settings-view settings-view--page" : "settings-view"} role={props.asPage ? undefined : "dialog"} aria-label={t("settings.title")} onClick={(e) => e.stopPropagation()}>
        <input ref={fileRef} type="file" accept="application/json" hidden onChange={onImportFile} />
        <nav className="settings-nav">
          <div className="settings-nav__head">
            <h2>{t("settings.title")}</h2>
          </div>
          {/* A localized placeholder, not a bare 🔍 (A-P2-6): a magnifier
              glyph alone renders as an unlabelled box in the two font stacks
              this ships with, and it is the one control on the page that
              cannot be guessed from its position. */}
          <input className="settings-search" placeholder={t("settings.searchLabel")} value={query} onChange={(e) => setQuery(e.target.value)} aria-label={t("settings.searchLabel")} />
          <div className="settings-nav__list">
            {cats.map((c) => (
              <button key={c.id} className={`settings-nav__item${active === c.id && !q ? " is-on" : ""}`} onClick={() => { setActive(c.id); setQuery(""); }}>
                {L(c.label, lang)}
              </button>
            ))}
          </div>
        </nav>

        <div className="settings-content">
          {!props.asPage ? (
            <button className="iconbtn settings-close" onClick={props.onClose} aria-label={t("settings.done")}>✕</button>
          ) : null}
          {searchHits ? (
            <>
              <h3 className="settings-content__title">{L({ en: "Results", zh: "搜索结果" }, lang)}</h3>
              {searchHits.length === 0 ? <p className="setting-hint">{L({ en: "No matches.", zh: "没有匹配的设置。" }, lang)}</p> : null}
              <div className="setting-group">{searchHits.map((h, i) => renderField(h.field, i))}</div>
            </>
          ) : (
            <>
              <h3 className="settings-content__title">{L(cat.label, lang)}</h3>
              {cat.note ? <div className="cat-note">{L(cat.note, lang)}</div> : null}
              {cat.bespoke === "quick" ? renderQuick() : null}
              {cat.bespoke === "resources" ? renderResources() : null}
              {cat.bespoke === "endpoints" ? <EndpointsPanel settings={s} /> : null}
              {/* A-P1-2: the marketplace console, reachable again. */}
              {cat.bespoke === "platform" ? (
                <PlatformPanel
                  settings={s}
                  session={props.session ?? null}
                  onOpenSettings={() => setActive("connect")}
                  onSignIn={() => props.onSignIn?.()}
                />
              ) : null}
              {/* Language and theme live with the rest of Appearance (moved
                  from the model page 2026-08-15). They are not AppSettings
                  fields — the app shell owns them — so they render here rather
                  than through the schema. */}
              {cat.id === "appearance" ? (
                <div className="setting-group">
                  <div className="setting-row setting-row--inline">
                    <span className="setting-row__k">{t("settings.language")}</span>
                    <Segmented<Lang>
                      options={[{ value: "en", label: "EN" }, { value: "zh", label: "中文" }]}
                      value={props.lang}
                      onChange={props.onLang}
                    />
                  </div>
                  <div className="setting-row setting-row--inline">
                    <span className="setting-row__k">{t("settings.theme")}</span>
                    <Segmented<Theme>
                      options={[
                        { value: "dark", label: t("settings.themeDark") },
                        { value: "light", label: t("settings.themeLight") },
                      ]}
                      value={props.theme}
                      onChange={props.onTheme}
                    />
                  </div>
                </div>
              ) : null}
              {cat.sections?.map((sec, si) => (
                <div className="setting-group" key={si}>
                  {sec.label ? <div className="setting-group__label">{L(sec.label, lang)}</div> : null}
                  {sec.fields.map((f, i) => renderField(f, i))}
                </div>
              ))}
              <button className="linkbtn" onClick={() => props.onChange({ ...DEFAULT_SETTINGS })}>{t("settings.reset")}</button>
            </>
          )}
        </div>
    </div>
  );
  // Everything applies immediately — a "Done" button would wrongly imply an
  // unsaved state, so there is none (2026-07 audit #6).
  if (props.asPage) return <main className="main main--wide">{inner}</main>;
  return (
    <div className="modal-scrim" onClick={props.onClose}>
      {inner}
    </div>
  );
}
