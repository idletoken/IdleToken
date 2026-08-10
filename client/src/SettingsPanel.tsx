import { useMemo, useRef, useState } from "react";
import { useI18n, type Lang } from "./i18n";
import { MODELS, hasQuantChoice, quantOptions, defaultQuant } from "./models";
import { fmtBytes } from "./format";
import {
  APP_VERSION,
  DEFAULT_SETTINGS,
  TIERS,
  saveSettings,
  type AppSettings,
  type ResourcePreset,
} from "./settings";
import type { NodeSnapshot } from "./types";
import type { Session } from "./auth";
import PlatformPanel from "./PlatformPanel";
import { getEngineProvider } from "./provider/engine";
import { buildDiagnosticsBundle, diagnosticsFileName } from "./diagnostics";

type Theme = "dark" | "light";
const MiB = 1024 ** 2;
const GiB = 1024 ** 3;

// Bilingual label, co-located with the schema (avoids ~200 i18n keys at this scale).
type Bi = { en: string; zh: string };
const L = (b: Bi, lang: Lang) => b[lang];

type FieldType = "toggle" | "select" | "text" | "password" | "number" | "slider" | "time" | "note" | "action";
interface Field {
  key?: keyof AppSettings;
  type: FieldType;
  label?: Bi;
  hint?: Bi;
  reserved?: boolean;
  options?: { value: string; label: Bi }[];
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  placeholder?: string;
  action?: "export" | "import" | "clearData" | "clearKv" | "diagnostics";
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
  bespoke?: "quick" | "platform";
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

// ---- weight status, on the model it belongs to ----------------------------
// Four states, and the button says what pressing it DOES in each:
//   downloading -> progress + Cancel      failed -> reason + Retry
//   missing     -> Download               present -> "ready", no button
export interface WeightsInfo {
  needs: boolean;
  path: string;
  dl: { have: number; total: number; endpoint?: string; note?: string; error?: string } | null;
  onDownload: () => void;
  onCancel: () => void;
}

function WeightsRow(props: { w: WeightsInfo }) {
  const { t } = useI18n();
  const d = props.w.dl;
  // Clicks land on a <label> that owns a radio input; without this, hitting
  // "Download" would also re-select the model (harmless here, but it makes the
  // button feel like it did something else).
  const stop = (e: React.MouseEvent) => e.preventDefault();

  if (d?.error) {
    return (
      <span className="model-opt__weights model-opt__weights--err" onClick={stop}>
        <span>{t("weights.failed")}</span>
        <button className="linkbtn" onClick={props.w.onDownload}>{t("weights.retry")}</button>
      </span>
    );
  }
  if (d) {
    const pct = d.total > 0 ? Math.min(100, Math.round((d.have / d.total) * 100)) : 0;
    return (
      <span className="model-opt__weights" onClick={stop}>
        <span className="model-opt__wbar"><i style={{ width: `${pct}%` }} /></span>
        <span>
          {fmtBytes(d.have)}{d.total > 0 ? ` / ${fmtBytes(d.total)}` : ""}
          {d.endpoint ? ` · ${t("weights.from")} ${new URL(d.endpoint).host}` : ""}
        </span>
        <button className="linkbtn" onClick={props.w.onCancel}>{t("weights.cancel")}</button>
      </span>
    );
  }
  if (props.w.needs) {
    return (
      <span className="model-opt__weights" onClick={stop}>
        <span>{t("weights.needed")}</span>
        <button className="linkbtn" onClick={props.w.onDownload}>{t("weights.download")}</button>
      </span>
    );
  }
  return <span className="model-opt__weights model-opt__weights--ok">{t("weights.ready")}</span>;
}

// ---- schema (everything except the bespoke Quick page) --------------------
// Six categories, down from fifteen (2026-07 UX audit): the handful of
// settings that actually work today must not drown in reserved placeholders.
// Everything not yet wired engine/OS-side lives under "Advanced (coming
// soon)" with disabled controls — honest, but out of the way.
const CATEGORIES: Category[] = [
  { id: "quick", label: { en: "Quick", zh: "快速" }, bespoke: "quick" },
  {
    id: "appearance",
    label: { en: "Appearance", zh: "外观" },
    sections: [{ fields: [
      { key: "accent", type: "select", label: { en: "Accent color", zh: "强调色" }, options: [
        { value: "amber", label: { en: "Electric", zh: "电光蓝" } }, { value: "teal", label: { en: "Teal", zh: "青绿" } },
        { value: "violet", label: { en: "Violet", zh: "紫" } }, { value: "rose", label: { en: "Rose", zh: "玫红" } } ] },
      { key: "uiScale", type: "slider", label: { en: "UI scale", zh: "界面缩放" }, min: 0.8, max: 1.4, step: 0.05, unit: "×" },
      { key: "density", type: "select", label: { en: "Density", zh: "密度" }, options: [
        { value: "comfortable", label: { en: "Comfortable", zh: "宽松" } }, { value: "compact", label: { en: "Compact", zh: "紧凑" } } ] },
      { key: "reduceMotion", type: "toggle", label: { en: "Reduce motion", zh: "减少动效" } },
    ] }],
  },
  {
    id: "connect",
    label: { en: "Cluster & API", zh: "集群与 API" },
    note: { en: "Engine-side values take effect when the cluster (re)starts.", zh: "引擎侧参数在集群（重新）启动时生效。" },
    sections: [
      { label: { en: "Pairing & discovery", zh: "组网与发现" }, fields: [
        { key: "clusterName", type: "text", label: { en: "Cluster name", zh: "集群名" } },
        { key: "mdns", type: "toggle", label: { en: "mDNS discovery", zh: "mDNS 发现" } },
        { key: "discoveryPort", type: "number", label: { en: "Discovery port", zh: "发现端口" } },
        { key: "manualPeers", type: "text", label: { en: "Manual peer IPs", zh: "手动 peer IP" }, placeholder: "192.168.1.50, 192.168.1.51" },
        { key: "heartbeatSec", type: "number", label: { en: "Heartbeat (s)", zh: "心跳（秒）" } },
        { key: "preferCoordinator", type: "toggle", label: { en: "Prefer this machine as coordinator", zh: "优先本机作协调者" } },
        { key: "sameSubnetOnly", type: "toggle", label: { en: "Only same subnet", zh: "仅限同子网" } },
        { key: "bindNic", type: "text", label: { en: "Bind interface / IP", zh: "绑定网卡 / IP" } },
        { key: "interStagePort", type: "number", label: { en: "Inter-stage port", zh: "节点间端口" } },
      ] },
      { label: { en: "API service", zh: "API 服务" }, fields: [
        { key: "apiHost", type: "text", label: { en: "Listen address", zh: "监听地址" }, hint: { en: "127.0.0.1 = this machine only.", zh: "127.0.0.1 = 仅本机可访问。" } },
        { key: "apiPort", type: "number", label: { en: "Port", zh: "端口" } },
        { key: "apiToken", type: "password", label: { en: "Access token", zh: "访问令牌" }, hint: { en: "Empty = open on the LAN.", zh: "留空 = 局域网内免鉴权。" } },
      ] },
      { label: { en: "Model & storage", zh: "模型与存储" }, fields: [
        { key: "weightsSource", type: "select", label: { en: "Weights source", zh: "权重来源" }, options: [
          { value: "auto", label: { en: "Pull from cluster", zh: "从集群拉取" } }, { value: "local", label: { en: "Local file", zh: "本地文件" } } ] },
        { key: "ggufPath", type: "text", label: { en: "GGUF file path", zh: "GGUF 文件路径" }, placeholder: "/path/to/ds4flash.gguf", showIf: (s) => s.weightsSource === "local" },
        // reserved: nothing reads these yet. Joiners already fetch only their own
        // layers from the coordinator's shard repo (pairing.rs), so the download
        // folder / auto-download / checksum toggles have no consumer — and idle
        // unload is not implemented in the engine at all. Leaving them editable
        // would be exactly the "hollow setting" principle 15 forbids: the user changes one and
        // nothing happens, with no way to tell.
        { key: "modelDir", type: "text", label: { en: "Model download folder", zh: "模型下载目录" }, placeholder: "~/.idletoken/models", hint: { en: "Empty = ~/.idletoken/models.", zh: "留空 = ~/.idletoken/models。" } },
        { key: "autoDownload", type: "toggle", label: { en: "Auto-download on join", zh: "加入时自动下载" }, reserved: true },
        { key: "verifySha", type: "toggle", label: { en: "Verify checksum (sha256)", zh: "校验 sha256" }, reserved: true },
        { key: "idleUnload", type: "toggle", label: { en: "Unload model when idle", zh: "空闲时卸载模型" }, reserved: true },
        { key: "idleUnloadMin", type: "number", label: { en: "Idle timeout (min)", zh: "空闲超时（分钟）" }, reserved: true, showIf: (s) => s.idleUnload },
      ] },
      { label: { en: "Inference & cache", zh: "推理与缓存" }, fields: [
        // reserved: the chat path hardcodes max_tokens in Rust (main.rs 200/512)
        // and never reads this. The default here happens to be 512 too, so it
        // LOOKS wired until you change it — the worst shape of a fake setting.
        { key: "maxTokens", type: "number", label: { en: "Max tokens per reply", zh: "单次最大生成 tokens" }, reserved: true },
        { key: "kvDir", type: "text", label: { en: "KV cache directory", zh: "KV 缓存目录" }, placeholder: "/tmp/idletoken-kv", hint: { en: "Empty = the platform cache directory.", zh: "留空 = 系统缓存目录。" } },
        { type: "action", action: "clearKv", label: { en: "Clear my cache now", zh: "立即清除我的缓存" }, hint: { en: "Wipes this machine's on-disk KV cache immediately.", zh: "立即清除本机落盘的 KV 缓存。" } },
      ] },
      { label: { en: "Platform account", zh: "平台账号" }, fields: [
        { type: "note", label: { en: "Only your identity crosses the wire — pairing and inference stay on your LAN.", zh: "只有账号身份走网络——组网与推理始终留在你的局域网内。" } },
        { key: "platformUrl", type: "text", label: { en: "Platform server URL", zh: "平台服务器地址" }, placeholder: "https://api.idletoken.ai",
          hint: { en: "Preset in release builds. Set: email sign-in goes through the platform and same-account machines can pair. Empty: identity stays on this machine (code pairing still works offline). Takes effect at the next sign-in.", zh: "发布版已预置。填写后邮箱登录走平台、同账号机器可互相组网；留空则身份仅存本机（验证码组网离线可用）。下次登录时生效。" } },
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
    note: {
      en: "Your prompt is encrypted end-to-end to the provider's cluster — the workers running your request only ever see hidden states, never your text. On the shared marketplace the platform sees plaintext to moderate content and meter usage (the trade-off of a shared service). On your own cluster, this is not needed.",
      zh: "你的 prompt 会端到端加密发到提供方集群——真正跑你请求的 workers 全程只见 hidden states，看不到你的文字。在共享市场里，平台会看到明文用于内容审核与计量（用共享服务的代价）。若是你自己的集群，则无需加密。",
    },
    sections: [
      { fields: [
        { type: "note", label: { en: "Prompts are envelope-encrypted (X25519 sealed box + AES-256-GCM) to the provider's coordinator; its workers never see your text. Prompts are never written to logs (enforced).", zh: "Prompt 信封加密（X25519 sealed box + AES-256-GCM）发到提供方 coordinator；它的 workers 看不到你的文字。Prompt 绝不写入日志（强制）。" } },
        { key: "telemetry", type: "toggle", label: { en: "Share anonymous telemetry", zh: "发送匿名遥测" } },
      ] },
    ],
  },
  {
    id: "data",
    label: { en: "Data & about", zh: "数据与关于" },
    sections: [
      { label: { en: "Diagnostics", zh: "诊断" }, fields: [
        { key: "logLevel", type: "select", label: { en: "Log level", zh: "日志级别" }, options: [
          { value: "error", label: { en: "error", zh: "error" } }, { value: "info", label: { en: "info", zh: "info" } }, { value: "debug", label: { en: "debug", zh: "debug" } } ] },
        { key: "experimental", type: "toggle", label: { en: "Enable experimental features", zh: "启用实验特性" } },
        { type: "action", action: "diagnostics",
          label: { en: "Export diagnostics bundle", zh: "导出诊断包" },
          hint: { en: "Hardware, driver, engine logs and cluster state in one file. No tokens, no prompts.",
                  zh: "把硬件、驱动、引擎日志、集群状态收成一个文件，便于求助。不含任何令牌与对话内容。" } },
      ] },
      { label: { en: "Data & backup", zh: "数据与备份" }, fields: [
        { key: "dataDir", type: "text", label: { en: "Data folder", zh: "数据目录" }, placeholder: "~/.idletoken" },
        { type: "action", action: "export", label: { en: "Export settings", zh: "导出设置" } },
        { type: "action", action: "import", label: { en: "Import settings", zh: "导入设置" } },
        { type: "action", action: "clearData", label: { en: "Clear all local data", zh: "清除全部本地数据" }, hint: { en: "Removes settings, identity and cluster state on this machine.", zh: "清除本机设置、身份与集群状态。" } },
      ] },
      { label: { en: "About", zh: "关于" }, fields: [
        { type: "note", label: { en: `IdleToken client ${APP_VERSION}`, zh: `IdleToken 客户端 ${APP_VERSION}` } },
        { type: "note", label: { en: "MIT-licensed. Reuses ds4 (MIT) for DSv4-Flash inference.", zh: "MIT 许可。复用 ds4 (MIT) 做 DSv4-Flash 推理。" } },
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
      { label: { en: "Window, startup & updates", zh: "窗口、启动与更新" }, fields: [
        { key: "closeToTray", type: "toggle", label: { en: "Close to tray", zh: "关闭到托盘" }, reserved: true },
        { key: "startMinimized", type: "toggle", label: { en: "Start minimized", zh: "启动即最小化" }, reserved: true },
        { key: "rememberWindow", type: "toggle", label: { en: "Remember window size/position", zh: "记住窗口大小/位置" }, reserved: true },
        { key: "trayIcon", type: "toggle", label: { en: "Show tray icon", zh: "显示托盘图标" }, reserved: true },
        { key: "autostart", type: "toggle", label: { en: "Launch at login", zh: "开机自启" }, reserved: true },
        { key: "autoRejoin", type: "toggle", label: { en: "Auto-rejoin last cluster", zh: "自动重连上次集群" }, reserved: true },
        { key: "autoUpdate", type: "toggle", label: { en: "Check for updates automatically", zh: "自动检查更新" }, reserved: true },
        { key: "updateChannel", type: "select", label: { en: "Update channel", zh: "更新通道" }, reserved: true, options: [
          { value: "stable", label: { en: "Stable", zh: "稳定版" } }, { value: "beta", label: { en: "Beta", zh: "测试版" } } ] },
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
    props.initialCategory && CATEGORIES.some((c) => c.id === props.initialCategory) ? props.initialCategory : "quick"
  );
  const [query, setQuery] = useState("");
  // "Clear my cache now" feedback (philosophy 15: every action has clear
  // loading/success/failure feedback).
  const [kvClear, setKvClear] = useState<"idle" | "busy" | "ok" | "err">("idle");
  const [diag, setDiag] = useState<"idle" | "busy" | "ok" | "err">("idle");
  const fileRef = useRef<HTMLInputElement>(null);
  const s = props.settings;
  const set = <K extends keyof AppSettings>(k: K, v: AppSettings[K]) => props.onChange({ ...s, [k]: v });
  const setCap = (k: "maxVramMb" | "maxRamMb", v: number) => props.onChange({ ...s, [k]: v, resourcePreset: "custom" });

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
    for (const c of CATEGORIES) {
      for (const sec of c.sections ?? []) {
        for (const f of sec.fields) {
          if (f.label && L(f.label, lang).toLowerCase().includes(q)) hits.push({ cat: c, field: f });
        }
      }
    }
    return hits;
  }, [q, lang]);

  const cat = CATEGORIES.find((c) => c.id === active) ?? CATEGORIES[0];

  const renderField = (f: Field, i: number) => {
    if (f.showIf && !f.showIf(s)) return null;
    const label = f.label ? L(f.label, lang) : "";
    const hint = f.hint ? L(f.hint, lang) : undefined;
    const tag = f.reserved ? <span className="reserved-tag">{t("settings.reservedTag")}</span> : null;

    if (f.type === "note") return <p key={i} className="about-line">{label}</p>;
    if (f.type === "action") {
      let btnLabel = label;
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
          <button className={`btn-secondary${f.action === "clearData" ? " btn-danger" : ""}${f.action === "clearKv" && kvClear === "err" ? " btn-danger" : ""}`} disabled={f.action === "clearKv" && kvClear === "busy"} onClick={() => runAction(f.action)}>{btnLabel}</button>
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
        <select className="select" value={val as string} disabled={dis} onChange={(e) => set(key, e.target.value as AppSettings[typeof key])}>
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
          {MODELS.map((m) => (
            <label key={m.id} className={`model-opt${s.modelId === m.id ? " is-on" : ""}${m.available ? "" : " is-disabled"}`}>
              <input
                type="radio"
                name="model"
                checked={s.modelId === m.id}
                disabled={!m.available}
                // Switching model resets precision to that model's default so
                // we never carry a quant the new model doesn't offer.
                onChange={() => props.onChange({ ...s, modelId: m.id, quant: defaultQuant(m.id) })}
              />
              <span className="model-opt__name">{m.label}</span>
              <span className="model-opt__params">{m.params}</span>
              {!m.available ? <span className="model-opt__soon">{t("settings.soon")}</span> : null}
              {/* Weight status sits on the SELECTED row only. It used to be a
                  floating bar pinned to every screen, which is wrong twice:
                  "no weights yet" is the resting state of a fresh install (so
                  the bar never left), and the thing it is about — which model —
                  is chosen right here. One probe, on the model it describes. */}
              {s.modelId === m.id && props.weights ? <WeightsRow w={props.weights} /> : null}
            </label>
          ))}
        </div>
        {hasQuantChoice(s.modelId) ? (
          <div className="setting-row setting-row--inline" style={{ marginTop: 10 }}>
            <div className="setting-row__label">
              <span className="setting-row__k">{t("settings.precision")}</span>
              <span className="setting-row__hint">{t("settings.precisionHint")}</span>
            </div>
            <div className="setting-row__control">
              <select
                className="select"
                value={s.quant || defaultQuant(s.modelId)}
                onChange={(e) => set("quant", e.target.value)}
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
        <p className="setting-hint">{t("settings.tierHint")}</p>
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
        <CapSlider label={t("settings.maxVram")} noCap={t("settings.noCap")} totalBytes={props.snap.vram_total} valueMb={s.maxVramMb} onChange={(mb) => setCap("maxVramMb", mb)} />
        <CapSlider label={t("settings.maxRam")} noCap={t("settings.noCap")} totalBytes={props.snap.ram_total} valueMb={s.maxRamMb} onChange={(mb) => setCap("maxRamMb", mb)} />
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
          <input className="settings-search" placeholder="🔍" value={query} onChange={(e) => setQuery(e.target.value)} aria-label="search settings" />
          <div className="settings-nav__list">
            {CATEGORIES.map((c) => (
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
              {searchHits.length === 0 ? <p className="setting-hint">{L({ en: "No matches.", zh: "无匹配。" }, lang)}</p> : null}
              <div className="setting-group">{searchHits.map((h, i) => renderField(h.field, i))}</div>
            </>
          ) : (
            <>
              <h3 className="settings-content__title">{L(cat.label, lang)}</h3>
              {cat.note ? <div className="cat-note">{L(cat.note, lang)}</div> : null}
              {cat.bespoke === "quick" ? renderQuick() : null}
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
