import { useMemo, useRef, useState } from "react";
import { useI18n, type Lang } from "./i18n";
import { EndpointsPanel } from "./EndpointsPanel";
import { AVAILABLE_MODELS, hasQuantChoice, isLocalGguf, quantOptions, defaultQuant } from "./models";
import { customGgufName } from "./weights";
import Capability from "./Capability";
import { fmtBytes } from "./format";
import {
  APP_VERSION,
  DEFAULT_SETTINGS,
  TIERS,
  effectiveCaps,
  saveSettings,
  type AppSettings,
  type ResourcePreset,
} from "./settings";
import type { NodeSnapshot } from "./types";
import type { Session } from "./auth";
import PlatformPanel from "./PlatformPanel";
import WeightsRow, { type WeightsInfo } from "./WeightsRow";
import StoredModels from "./StoredModels";
import ProblemLog from "./ProblemLog";
import { getEngineProvider } from "./provider/engine";
import { buildDiagnosticsBundle, diagnosticsFileName } from "./diagnostics";

type Theme = "dark" | "light";
const MiB = 1024 ** 2;
const GiB = 1024 ** 3;

// Bilingual label, co-located with the schema (avoids ~200 i18n keys at this scale).
type Bi = { en: string; zh: string };
const L = (b: Bi, lang: Lang) => b[lang];

type FieldType =
  | "toggle" | "select" | "text" | "password" | "number" | "slider" | "time" | "note" | "action"
  /** Full-width block: the model folder's contents, with a Delete per file. */
  | "stored-models"
  /** Full-width block: the local problem log + its consent switch. */
  | "problems";
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
  action?: "export" | "import" | "clearData" | "clearKv" | "diagnostics" | "checkUpdate";
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
  bespoke?: "quick" | "platform" | "endpoints";
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

// ---- schema (everything except the bespoke Quick page) --------------------
// Six categories, down from fifteen (2026-07 UX audit): the handful of
// settings that actually work today must not drown in reserved placeholders.
// Everything not yet wired engine/OS-side lives under "Advanced (coming
// soon)" with disabled controls — honest, but out of the way.
const CATEGORIES: Category[] = [
  { id: "quick", label: { en: "Quick", zh: "常用" }, bespoke: "quick" },
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
    id: "connect",
    // The "how do I point Claude Code at this" block sits above the raw
    // host/port/token fields. Until now the client never told anyone its own
    // base URL: you had to go read the README to use the thing you had just
    // started (docs/api-surface.md §7).
    bespoke: "endpoints",
    label: { en: "Cluster & API", zh: "集群与 API" },
    note: { en: "Engine-side settings take effect when the cluster restarts.", zh: "引擎相关设置在集群重新启动后生效。" },
    sections: [
      // All nine are live as of 2026-08-13. The six pairing ones used to be
      // rendered as working controls that nothing read — the toggle labelled
      // "mDNS discovery" was the worst of them, since the discovery it did not
      // control is not mDNS either (it is a UDP broadcast beacon). They are
      // consumed by src-tauri/src/pairing.rs; each hint says WHICH of the two
      // discovery layers or which side of the handshake it governs, because
      // that is what nobody could tell from the label.
      { label: { en: "Pairing & discovery", zh: "组网与发现" }, fields: [
        { key: "clusterName", type: "text", label: { en: "Cluster name", zh: "集群名" } },
        { key: "mdns", type: "toggle", label: { en: "LAN auto-discovery", zh: "局域网自动发现" } },
        { key: "discoveryPort", type: "number", label: { en: "Discovery port", zh: "发现端口" } },
        { key: "manualPeers", type: "text", label: { en: "Manual peer IPs", zh: "手动节点 IP" }, placeholder: "192.168.1.50, 192.168.1.51" },
        { key: "heartbeatSec", type: "number", label: { en: "Heartbeat (s)", zh: "心跳（秒）" } },
        { key: "preferCoordinator", type: "toggle", label: { en: "Prefer this machine as coordinator", zh: "优先本机作协调者" } },
        { key: "sameSubnetOnly", type: "toggle", label: { en: "Only same subnet", zh: "仅限同子网" } },
        { key: "bindNic", type: "text", label: { en: "Bind interface / IP", zh: "绑定网卡 / IP" }, placeholder: "auto" },
        { key: "interStagePort", type: "number", label: { en: "Inter-stage port", zh: "节点间端口" } },
      ] },
      { label: { en: "API service", zh: "API 服务" }, fields: [
        { key: "apiHost", type: "text", label: { en: "Listen address", zh: "监听地址" } },
        { key: "apiPort", type: "number", label: { en: "Port", zh: "端口" } },
        { key: "apiToken", type: "password", label: { en: "Access token", zh: "访问令牌" } },
      ] },
      { label: { en: "Model & storage", zh: "模型与存储" }, fields: [
        // "Weights source" (auto | local file) was removed on 2026-08-13.
        //
        // The choice was not one: "auto" already covers every case — a complete
        // local copy is used as-is, a joiner streams only its own layers from
        // the coordinator, and a coordinator/standalone downloads. The label
        // ("Pull from cluster") described just the middle one, so on a single
        // machine it read as something for other people.
        //
        // "Local file" was worse than redundant: the path was handed to the
        // engine unchecked, so a typo — or the empty box you get the moment you
        // switch to it — showed "Weights ready on this machine" and failed at
        // load time instead. Pointing "Model download folder" at an existing
        // directory does the same job and is verified (weights_state).
        // reserved: nothing reads these yet. Joiners already fetch only their own
        // layers from the coordinator's shard repo (pairing.rs), so the download
        // folder / auto-download / checksum toggles have no consumer — and idle
        // unload is not implemented in the engine at all. Leaving them editable
        // would be exactly the "hollow setting" principle 15 forbids: the user changes one and
        // nothing happens, with no way to tell.
        { key: "modelDir", type: "text", label: { en: "Model download folder", zh: "模型下载目录" }, placeholder: "~/.idletoken/models" },
        // What that folder actually holds, and a way to get the space back. A
        // model is tens of gigabytes; trying two of them fills a laptop, and
        // until now the only way to clean up was to find the folder yourself.
        { type: "note", label: { en: "Downloaded weights on this machine", zh: "本机已下载的权重" } },
        { type: "stored-models" },
        { key: "autoDownload", type: "toggle", label: { en: "Auto-download on join", zh: "加入时自动下载" }, reserved: true },
        { key: "verifySha", type: "toggle", label: { en: "Verify checksum (sha256)", zh: "校验 sha256" }, reserved: true },
        { key: "idleUnload", type: "toggle", label: { en: "Unload model when idle", zh: "空闲时卸载模型" }, reserved: true },
        { key: "idleUnloadMin", type: "number", label: { en: "Idle timeout (min)", zh: "空闲超时（分钟）" }, reserved: true, showIf: (s) => s.idleUnload },
      ] },
      { label: { en: "Inference & cache", zh: "推理与缓存" }, fields: [
        // Real since 2026-08-11, in both directions: the chat sends it per
        // request, and it is also passed to the coordinator as --max-decode, so
        // it bounds third-party API clients too. (It used to be the worst shape
        // of a fake setting: the Rust chat path hardcoded 200/512 and the engine
        // hardcoded 4096, while the default here happened to be 512 — so it
        // looked wired right up until you changed it.)
        { key: "maxTokens", type: "number", label: { en: "Max tokens per reply", zh: "单次最大生成 tokens" } },
        { key: "kvDir", type: "text", label: { en: "KV cache directory", zh: "KV 缓存目录" }, placeholder: "/tmp/idletoken-kv" },
        { type: "action", action: "clearKv", label: { en: "Clear KV cache", zh: "清除 KV 缓存" } },
      ] },
      { label: { en: "Platform account", zh: "平台账号" }, fields: [
        { type: "note", label: { en: "Only your account identity goes to the platform; pairing and inference happen on your LAN.", zh: "只有账号身份经过平台；组网与推理在局域网内完成。" } },
        { key: "platformUrl", type: "text", label: { en: "Platform server URL", zh: "平台服务器地址" }, placeholder: "https://api.idletoken.ai" },
      ] },
    ],
  },
  // Was a top-level place ("Marketplace") until 2026-08-10. Browsing the
  // market is a browser job; what needs THIS machine — listing this cluster,
  // balance, ledger, API keys, rendezvous token — is a settings category.
  { id: "platform", label: { en: "Sharing & earnings", zh: "共享与收益" }, bespoke: "platform" },
  {
    id: "privacy",
    label: { en: "Privacy", zh: "隐私保护" },
    // Marketplace wording removed on 2026-08-13: this page is read on the
    // machine you own, by someone asking "what happens to what I type". Where
    // the platform can see plaintext is a property of selling/buying on the
    // market — it belongs on that screen, not here, where it only made the
    // encryption story harder to follow.
    note: {
      en: "Your prompt is encrypted end to end to the cluster serving it; the nodes running it do not see the original text. On your own machines, it never leaves them.",
      zh: "提示词经端到端加密送达提供服务的集群，执行节点接触不到原文。若集群全部由你自己的机器组成，内容不会离开这些机器。",
    },
    sections: [
      { fields: [
        { type: "note", label: { en: "Encryption: X25519 + AES-256-GCM envelope to the coordinator. Prompts are not written to logs.", zh: "加密方式：X25519 + AES-256-GCM 信封加密。提示词不写入日志。" } },
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
      { label: { en: "Problems & diagnostics", zh: "问题与诊断" }, fields: [
        // The improvement loop, and the honest bound on it: recorded here,
        // nothing is sent. What reaches us is what the user exports below.
        { type: "problems" },
        { type: "action", action: "diagnostics",
          label: { en: "Export diagnostics bundle", zh: "导出诊断包" } },
      ] },
      // "Data folder" removed with them: the client's data lives in this
      // machine's app-data directory (Tauri decides it) and in localStorage,
      // and no code ever read the box. Typing a path did not move anything.
      { label: { en: "Data & backup", zh: "数据与备份" }, fields: [
        { type: "action", action: "export", label: { en: "Export settings", zh: "导出设置" } },
        { type: "action", action: "import", label: { en: "Import settings", zh: "导入设置" } },
        { type: "action", action: "clearData", label: { en: "Clear all local data", zh: "清除全部本地数据" } },
      ] },
      { label: { en: "About", zh: "关于" }, fields: [
        { type: "note", label: { en: `IdleToken client ${APP_VERSION}`, zh: `IdleToken 客户端 ${APP_VERSION}` } },
        // Said wrong until 2026-08-13 ("MIT-licensed"). IdleToken is
        // **Apache-2.0** — LICENSE, NOTICE and the README have always said so;
        // the MIT belongs to vendored ds4, and the two got merged into one
        // sentence. A licence claim is the one line in an About box that
        // someone may act on, so it now names both, separately, and points at
        // the NOTICE that Apache-2.0 §4(d) requires to travel with the build.
        // Updated with the 2026-08-14 engine pivot: llama.cpp is the inference
        // engine now; ds4 remains only as the frozen DSv4-Flash legacy backend
        // (opt-in via IDLETOKEN_FORCE_BACKEND=ds4). NOTICE carries the full
        // third-party list, as Apache-2.0 §4(d) requires.
        { type: "note", label: {
          en: "Apache-2.0. Inference is powered by llama.cpp (MIT). Includes ds4 (MIT) as a legacy optional backend; other third-party components are listed in NOTICE.",
          zh: "Apache-2.0 许可。推理引擎为 llama.cpp（MIT）；内含 ds4（MIT）作为遗留可选组件；其余第三方组件见 NOTICE。" } },
      ] },
    ],
  },
  {
    id: "advanced",
    label: { en: "Advanced · coming soon", zh: "高级 · 即将推出" },
    note: {
      en: "Planned capabilities, shown so you know what's coming. Controls are disabled until the engine/OS integration lands — changing them would do nothing today.",
      zh: "这些能力已在规划中，提前展示让你知道方向；在引擎/系统集成落地前控件不可用——现在改了也不会生效。",
    },
    sections: [
      { label: { en: "Compute", zh: "计算" }, fields: [
        { key: "computeMode", type: "select", label: { en: "Compute mode", zh: "计算模式" }, reserved: true, options: [
          { value: "auto", label: { en: "Auto", zh: "自动" } }, { value: "gpu_only", label: { en: "GPU only", zh: "纯 GPU" } }, { value: "hybrid", label: { en: "Hybrid", zh: "混合" } } ] },
      ] },
      { label: { en: "Sampling defaults", zh: "默认采样" }, fields: [
        { key: "temperature", type: "slider", label: { en: "Temperature", zh: "温度" }, min: 0, max: 2, step: 0.05, reserved: true },
        { key: "topP", type: "slider", label: { en: "Top-p", zh: "Top-p" }, min: 0, max: 1, step: 0.01, reserved: true },
        { key: "topK", type: "number", label: { en: "Top-k", zh: "Top-k" }, reserved: true },
      ] },
      { label: { en: "KV cache", zh: "KV 缓存" }, fields: [
        { key: "kvMaxMb", type: "number", label: { en: "Max size (MiB)", zh: "尺寸上限（MiB）" }, reserved: true },
        { key: "kvTtlDays", type: "number", label: { en: "Idle TTL (days)", zh: "空闲保留（天）" }, reserved: true },
        { key: "kvEviction", type: "select", label: { en: "Eviction policy", zh: "驱逐策略" }, reserved: true, options: [
          { value: "lru", label: { en: "LRU (least recently used)", zh: "LRU（最久未用先出）" } },
          { value: "fifo", label: { en: "FIFO (oldest first)", zh: "FIFO（最早写入先出）" } } ] },
        { key: "kvOffload", type: "toggle", label: { en: "Offload live KV to disk when VRAM is tight", zh: "显存吃紧时把在用 KV 卸载到磁盘" }, reserved: true },
      ] },
      { label: { en: "API hardening", zh: "API 加固" }, fields: [
        { key: "apiOpenAI", type: "toggle", label: { en: "OpenAI API", zh: "OpenAI API" }, reserved: true },
        { key: "apiAnthropic", type: "toggle", label: { en: "Anthropic API", zh: "Anthropic API" }, reserved: true },
        { key: "apiStreaming", type: "toggle", label: { en: "Streaming (SSE)", zh: "流式 (SSE)" }, reserved: true },
        { key: "apiLocalOnly", type: "toggle", label: { en: "Localhost only", zh: "仅本地" }, reserved: true },
        { key: "apiCors", type: "text", label: { en: "CORS origins", zh: "CORS 来源" }, reserved: true },
        { key: "apiRateLimit", type: "number", label: { en: "Rate limit (req/min, 0=off)", zh: "限流（次/分，0=关）" }, reserved: true },
        { key: "apiTimeoutSec", type: "number", label: { en: "Request timeout (s)", zh: "请求超时（秒）" }, reserved: true },
        { key: "apiRequestLog", type: "toggle", label: { en: "Log requests", zh: "记录请求" }, reserved: true },
      ] },
      { label: { en: "Power & thermal", zh: "电源与散热" }, fields: [
        { key: "pauseOnGpuBusy", type: "toggle", label: { en: "Pause when I'm using the GPU (gaming)", zh: "我在用 GPU（游戏）时暂停" }, reserved: true },
        { key: "pauseOnBattery", type: "toggle", label: { en: "Pause on battery", zh: "用电池时暂停" }, reserved: true },
        { key: "scheduleEnabled", type: "toggle", label: { en: "Only run on a schedule", zh: "仅按时段运行" }, reserved: true },
        { key: "scheduleFrom", type: "time", label: { en: "From", zh: "从" }, reserved: true, showIf: (s) => s.scheduleEnabled },
        { key: "scheduleTo", type: "time", label: { en: "To", zh: "到" }, reserved: true, showIf: (s) => s.scheduleEnabled },
        { key: "powerLimitPct", type: "slider", label: { en: "GPU power limit", zh: "GPU 功耗上限" }, min: 30, max: 100, step: 5, unit: "%", reserved: true },
        { key: "tempLimitC", type: "number", label: { en: "Throttle above °C (0=off)", zh: "超过 °C 降频（0=关）" }, reserved: true },
      ] },
      { label: { en: "Notifications", zh: "通知" }, fields: [
        { key: "notifyEnabled", type: "toggle", label: { en: "Enable notifications", zh: "启用通知" }, reserved: true },
        { key: "notifyNodeChange", type: "toggle", label: { en: "Machine joins / leaves", zh: "机器加入 / 退出" }, reserved: true, showIf: (s) => s.notifyEnabled },
        { key: "notifyReady", type: "toggle", label: { en: "Cluster ready", zh: "集群就绪" }, reserved: true, showIf: (s) => s.notifyEnabled },
        { key: "notifyErrors", type: "toggle", label: { en: "Errors", zh: "错误" }, reserved: true, showIf: (s) => s.notifyEnabled },
        { key: "notifyDownload", type: "toggle", label: { en: "Download finished", zh: "下载完成" }, reserved: true, showIf: (s) => s.notifyEnabled },
        { key: "notifySound", type: "toggle", label: { en: "Play a sound", zh: "播放声音" }, reserved: true, showIf: (s) => s.notifyEnabled },
      ] },
      // Window / startup / updates moved OUT of this category on 2026-08-13,
      // into the visible "Startup & updates" one below: they are wired to the
      // shell now (src-tauri/src/window.rs, tray.rs, update.rs). Only the one
      // that still has no implementation stayed behind.
      { label: { en: "Startup", zh: "启动" }, fields: [
        { key: "autoRejoin", type: "toggle", label: { en: "Auto-rejoin last cluster", zh: "自动重连上次集群" }, reserved: true },
      ] },
      { label: { en: "Privacy hardening", zh: "隐私增强" }, fields: [
        { key: "privacyEncryptAtRest", type: "toggle", label: { en: "Encrypt data at rest (KV, snapshots)", zh: "落盘加密（KV、快照）" }, reserved: true },
        { key: "privacyLockMemory", type: "toggle", label: { en: "Lock plaintext in memory (mlock)", zh: "锁定明文内存 (mlock)" }, reserved: true },
        { key: "privacyPadding", type: "toggle", label: { en: "Pad request length", zh: "填充请求长度" }, reserved: true },
        { key: "privacyDpNoise", type: "toggle", label: { en: "Differential-privacy noise", zh: "差分隐私噪声" }, reserved: true },
        { key: "privacyDummyTokens", type: "toggle", label: { en: "Insert dummy tokens", zh: "插入 dummy token" }, reserved: true },
      ] },
    ],
  },
];

// Hidden for now (2026-08-11): "Advanced · coming soon" was a roadmap shelf —
// 19 disabled controls for things the engine/OS side hasn't landed. Until they
// are wired it is noise, so the category is kept in the schema (defaults and
// AppSettings keys are unchanged, nothing migrates) but taken out of the nav
// and out of search. Delete the id from this set to bring it back.
const HIDDEN_CATEGORIES: ReadonlySet<string> = new Set(["advanced"]);
const VISIBLE_CATEGORIES = CATEGORIES.filter((c) => !HIDDEN_CATEGORIES.has(c.id));

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
  // Weight presence + download control for the SELECTED model. Owned by App
  // (it holds the progress-event subscription); rendered on the model row.
  weights?: WeightsInfo;
  /** Selecting a model IS the switch (2026-08-15): one semantic everywhere.
   *  This routes the pick through the same flow as the model picker — save the
   *  choice; if something is running, stop it and rebuild with the new model.
   *  When absent (embedded reuse), the pick falls back to a plain setting
   *  write. */
  onSwitchModel?: (modelId: string, quant: string) => void;
  /** Re-probe the selected model's weights — the stored-models list calls it
   *  after a deletion, so the row above cannot keep claiming "ready" for a file
   *  that is no longer there. */
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
  const [active, setActive] = useState(
    props.initialCategory && VISIBLE_CATEGORIES.some((c) => c.id === props.initialCategory) ? props.initialCategory : "quick"
  );
  const [query, setQuery] = useState("");
  // "Clear my cache now" feedback (philosophy 15: every action has clear
  // loading/success/failure feedback).
  const [kvClear, setKvClear] = useState<"idle" | "busy" | "ok" | "err">("idle");
  const [diag, setDiag] = useState<"idle" | "busy" | "ok" | "err">("idle");
  // The button says "checking" while the feed is being asked; the ANSWER is
  // the dialog App opens, so there is nothing to report back here.
  const [upd, setUpd] = useState<"idle" | "busy">("idle");
  const fileRef = useRef<HTMLInputElement>(null);
  const s = props.settings;
  const set = <K extends keyof AppSettings>(k: K, v: AppSettings[K]) => props.onChange({ ...s, [k]: v });
  const setCap = (k: "maxVramMb" | "maxRamMb", v: number) => props.onChange({ ...s, [k]: v, resourcePreset: "custom" });
  // The caps actually in force — the same function App feeds to the probe and
  // to the engine, so the panel can never quote a limit the engine is not
  // given. Under a preset these are a fraction of this machine's totals; under
  // "custom" they are the slider values verbatim.
  const liveCaps = effectiveCaps(s, { vram_total: props.snap.vram_total, ram_total: props.snap.ram_total });

  const runAction = (a: Field["action"]) => {
    if (a === "export") {
      const blob = new Blob([JSON.stringify(s, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "idletoken-settings.json";
      link.click();
      URL.revokeObjectURL(url);
    } else if (a === "diagnostics") {
      if (diag === "busy") return;
      setDiag("busy");
      getEngineProvider()
        .diagnostics(props.apiBaseUrl || `http://127.0.0.1:${s.apiPort || 8000}`)
        .then((report) => {
          // Assembly goes through the same function in diagnostics.ts that the
          // acceptance channel uses -- the place that decides whether a token
          // goes out should not have two implementations.
          const bundle = buildDiagnosticsBundle(report, s);
          const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: "application/json" });
          const url = URL.createObjectURL(blob);
          const link = document.createElement("a");
          link.href = url;
          link.download = diagnosticsFileName();
          link.click();
          URL.revokeObjectURL(url);
          setDiag("ok");
        })
        .catch(() => setDiag("err"))
        .finally(() => setTimeout(() => setDiag("idle"), 2500));
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
    for (const c of VISIBLE_CATEGORIES) {
      for (const sec of c.sections ?? []) {
        for (const f of sec.fields) {
          if (f.label && L(f.label, lang).toLowerCase().includes(q)) hits.push({ cat: c, field: f });
        }
      }
    }
    return hits;
  }, [q, lang]);

  const cat = VISIBLE_CATEGORIES.find((c) => c.id === active) ?? VISIBLE_CATEGORIES[0];

  const renderField = (f: Field, i: number) => {
    if (f.showIf && !f.showIf(s)) return null;
    const label = f.label ? L(f.label, lang) : "";
    const hint = f.hint ? L(f.hint, lang) : undefined;
    const tag = f.reserved ? <span className="reserved-tag">{t("settings.reservedTag")}</span> : null;

    if (f.type === "note") return <p key={i} className="about-line">{label}</p>;
    if (f.type === "stored-models")
      return <StoredModels key={i} modelDir={s.modelDir} onChanged={props.onWeightsChanged} />;
    if (f.type === "problems") return <ProblemLog key={i} />;
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
        <div className="model-list">
          {AVAILABLE_MODELS.map((m) => (
            <label key={m.id} className={`model-opt${s.modelId === m.id ? " is-on" : ""}`}>
              <input
                type="radio"
                name="model"
                checked={s.modelId === m.id}
                // Switching model resets precision to that model's default so
                // we never carry a quant the new model doesn't offer. The pick
                // goes through the switch flow: if a cluster or local engine is
                // running, it is stopped and rebuilt with this model — there is
                // no "setting disagrees with what is running" state to explain.
                onChange={() =>
                  props.onSwitchModel
                    ? props.onSwitchModel(m.id, defaultQuant(m.id))
                    : props.onChange({ ...s, modelId: m.id, quant: defaultQuant(m.id) })
                }
              />
              <span className="model-opt__name">{m.label}</span>
              <span className="model-opt__params">{m.params}</span>
              {/* Whether a model can be pooled across machines changes what the
                  Cluster screen will let you do, and that is not guessable from
                  its size — say it on the row where the choice is made. */}
              <span className="model-opt__deploy">
                {t(m.singleNode ? "settings.model.singleNode" : "settings.model.cluster")}
              </span>
              {/* Weight status sits on the SELECTED row only. It used to be a
                  floating bar pinned to every screen, which is wrong twice:
                  "no weights yet" is the resting state of a fresh install (so
                  the bar never left), and the thing it is about — which model —
                  is chosen right here. One probe, on the model it describes. */}
              {s.modelId === m.id && props.weights ? <WeightsRow w={props.weights} idle="show" /> : null}
            </label>
          ))}
          {/* Open-intake selection (v2 WS-D1): when the setting points at a
              user-supplied GGUF, the curated radios above are all unchecked —
              this row says what IS selected instead of leaving the list
              looking broken. Changing/choosing an open model happens in the
              model picker (chat header / cluster card), which owns the file
              dialog and the HF input. */}
          {isLocalGguf(s.modelId) ? (
            <label className="model-opt is-on">
              <input type="radio" name="model" checked readOnly />
              <span className="model-opt__name">
                {customGgufName({
                  source: s.customSource,
                  path: s.customGgufPath,
                  repo: s.customHfRepo,
                  file: s.customHfFile,
                })}
              </span>
              <span className="model-opt__params">{t("model.open.badge")}</span>
              <span className="model-opt__deploy">{t("settings.model.singleNode")}</span>
              {props.weights ? <WeightsRow w={props.weights} idle="show" /> : null}
            </label>
          ) : null}
        </div>
        {!isLocalGguf(s.modelId) && hasQuantChoice(s.modelId) ? (
          <div className="setting-row setting-row--inline" style={{ marginTop: 10 }}>
            <div className="setting-row__label">
              <span className="setting-row__k">{t("settings.precision")}</span>
            </div>
            <div className="setting-row__control">
              <select
                className="select"
                value={s.quant || defaultQuant(s.modelId)}
                // A precision change is a model switch too — different weights,
                // same restart semantics as picking another model.
                onChange={(e) =>
                  props.onSwitchModel
                    ? props.onSwitchModel(s.modelId, e.target.value)
                    : set("quant", e.target.value)
                }
              >
                {quantOptions(s.modelId).map((v) => (
                  <option key={v.quant} value={v.quant}>
                    {v.quant} · {fmtBytes(v.layer_weight_bytes + v.shared_weight_bytes)}
                  </option>
                ))}
              </select>
            </div>
          </div>
        ) : null}
        {/* "What can I run?" moved here from the cluster card (2026-08-11).
            On the cluster card it was a table of models you could not choose —
            it listed what the hardware supports next to no way to act on it.
            Here it sits directly under the picker it is advice ABOUT. */}
        <Capability apiBaseUrl={props.apiBaseUrl ?? null} />
      </div>
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
      <div className="setting-group">
        <div className="setting-group__label">{t("settings.resource")}</div>
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
      <div className="setting-group">
        <div className="setting-group__label">{t("settings.appearance")}</div>
        <div className="setting-row setting-row--inline"><span className="setting-row__k">{t("settings.language")}</span>
          <Segmented<Lang> options={[{ value: "en", label: "EN" }, { value: "zh", label: "中文" }]} value={props.lang} onChange={props.onLang} /></div>
        <div className="setting-row setting-row--inline"><span className="setting-row__k">{t("settings.theme")}</span>
          <Segmented<Theme> options={[{ value: "dark", label: t("settings.themeDark") }, { value: "light", label: t("settings.themeLight") }]} value={props.theme} onChange={props.onTheme} /></div>
      </div>
    </>
  );

  const inner = (
    <div className={props.asPage ? "settings-view settings-view--page" : "settings-view"} role={props.asPage ? undefined : "dialog"} aria-label={t("settings.title")} onClick={(e) => e.stopPropagation()}>
        <input ref={fileRef} type="file" accept="application/json" hidden onChange={onImportFile} />
        <nav className="settings-nav">
          <div className="settings-nav__head">
            <h2>{t("settings.title")}</h2>
          </div>
          <input className="settings-search" placeholder="🔍" value={query} onChange={(e) => setQuery(e.target.value)} aria-label={t("settings.searchLabel")} />
          <div className="settings-nav__list">
            {VISIBLE_CATEGORIES.map((c) => (
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
              {cat.bespoke === "endpoints" ? <EndpointsPanel settings={s} /> : null}
              {cat.bespoke === "platform" ? (
                <PlatformPanel
                  settings={s}
                  session={props.session ?? null}
                  // "no platform URL" sends you to the field that fixes it,
                  // which lives one category over — not to a dead end.
                  onOpenSettings={() => setActive("connect")}
                  onSignIn={props.onSignIn ?? (() => {})}
                />
              ) : null}
              {cat.sections?.map((sec, si) => (
                <div className="setting-group" key={si}>
                  {sec.label ? <div className="setting-group__label">{L(sec.label, lang)}</div> : null}
                  {sec.fields.map((f, i) => renderField(f, i))}
                </div>
              ))}
              {/* Sharing & earnings holds no settings — a "reset to defaults"
                  under a credit ledger reads like it would reset the ledger. */}
              {cat.bespoke === "platform" ? null : (
                <button className="linkbtn" onClick={() => props.onChange({ ...DEFAULT_SETTINGS })}>{t("settings.reset")}</button>
              )}
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
