// SCRATCH HARNESS, not part of the app — see chatlab.html. Mounts <Chat> with a
// fake "online" cluster and a seeded transcript so the layout can be judged at
// any window size without an engine. Delete when the chat UI work lands.
import React, { useState } from "react";
import ReactDOM from "react-dom/client";
import Chat from "./Chat";
import { LangContext, type Lang } from "./i18n";
import type { ClusterApi } from "./pairing";
import "./fonts.css";
import "./styles.css";

const api: ClusterApi = { baseUrl: "http://127.0.0.1:8000", status: "online" } as ClusterApi;

const SEED = [
  {
    id: "c-lab-1",
    title: "Pipeline parallelism over a home LAN",
    updatedAt: Date.now(),
    msgs: [
      { role: "user", text: "How does pipeline parallelism work across several machines on a home LAN?" },
      {
        role: "assistant",
        text:
          "Pipeline parallelism splits the model by layer rather than by tensor. Each machine holds a contiguous " +
          "range of layers — a stage — and the only thing that crosses the network is the activation tensor at the " +
          "stage boundary.\n\nThat is what makes it viable on a gigabit LAN. Tensor parallelism would have every " +
          "machine exchange partial sums twice per layer, tens of megabytes each time, which is why it needs NVLink. " +
          "A pipeline boundary moves one hidden-state vector per token: for a 4096-wide model in FP16 that is 8 KB.\n\n" +
          "The cost is latency, not bandwidth. A token has to walk the whole pipeline before the next one starts, so " +
          "your per-token time is the sum of every stage plus the round trips between them.",
        stats: { ttftMs: 820, totalMs: 9400, tokens: 186 },
      },
      { role: "user", text: "So where does the KV cache live?" },
      {
        role: "assistant",
        text:
          "With each stage, on the machine that owns those layers. Layer 12's keys and values are only ever read by " +
          "layer 12's attention, so there is no reason to move them.",
        stats: { ttftMs: 640, totalMs: 3100, tokens: 47 },
      },
    ],
  },
  {
    id: "c-lab-2",
    title: "Why is my first token slow?",
    updatedAt: Date.now() - 3600_000,
    msgs: [
      { role: "user", text: "If a stage takes 40 ms and I have 4 machines, what is my per-token time?" },
      {
        role: "assistant",
        text:
          "<think>Four stages at 40 ms each is 160 ms of compute. Then three hops between them plus the return; " +
          "on a gigabit LAN each hop is about 1 ms for an 8 KB tensor, so roughly 4 ms of network. Total ~164 ms." +
          "</think>About 164 ms per token: 160 ms of compute (4 stages x 40 ms) plus roughly 4 ms of network for " +
          "the three inter-stage hops and the return trip.\n\nThat works out to just over 6 tokens per second.",
        stats: { ttftMs: 410, totalMs: 2600, tokens: 58 },
      },
    ],
  },
  {
    // The overflow case: a first message far longer than the sidebar is wide,
    // with no spaces to break on (CJK), which is where character-count
    // truncation and CSS ellipsis both get tested.
    id: "c-lab-4",
    // Stored in the shape titleOf() produces today: 34 chars + an ellipsis.
    title: "我想知道在千兆局域网里用流水线并行跑一个三百亿参数的大模型到底可不可…",
    updatedAt: Date.now() - 200_000,
    msgs: [
      { role: "user", text: "我想知道在千兆局域网里用流水线并行跑一个三百亿参数的大模型到底可不可行，需要几台机器，每台机器至少要多少显存和内存，首字延迟大概是多少，能不能接进 Claude Code 当日常用的后端？" },
      { role: "assistant", text: "可行，但要看你的网络。", stats: { ttftMs: 400, totalMs: 1200, tokens: 12 } },
    ],
  },
  {
    id: "c-lab-3",
    title: "Quantisation trade-offs for Q2",
    updatedAt: Date.now() - 86400_000,
    // Deliberately markdown-heavy: headings, nested lists, a table, inline code
    // and a fence. This is the case the plain-text renderer used to mangle, so
    // it is the one the harness must show.
    msgs: [
      { role: "user", text: "Summarise the Q2 quantisation trade-offs, with a table." },
      {
        role: "assistant",
        text: [
          "Short answer: **Q2 is what makes a 304B model fit in 80 GB**, and you pay for it in the experts.",
          "",
          "### Where the bits go",
          "",
          "* MoE expert `up`/`gate` -> **IQ2_XXS**, the most aggressive tier",
          "  * these dominate the file, so this is where the savings are",
          "  * quality loss shows up first on long reasoning chains",
          "* MoE expert `down` -> **Q2_K**",
          "* Everything else stays **FP8** - attention, norms, the router",
          "",
          "### What it costs",
          "",
          "| Variant | Size | Quality | Notes |",
          "| --- | --- | --- | --- |",
          "| IQ2_XXS + Q2_K | 80.8 GiB | baseline | fits 4 home machines |",
          "| Last 6 layers Q4_K | 90.9 GiB | better tail | needs ~10 GiB more |",
          "| FP8 | 304 GB | reference | not happening at home |",
          "",
          "You can check which variant a file is without downloading it:",
          "",
          "```bash",
          "python3 tools/gguf_shard.py --inspect \\",
          "  --url https://example/DeepSeek-V4-Flash-Q2.gguf",
          "```",
          "",
          "> The layer/shared split is read over HTTP Range, so this costs a few KB, not 80 GiB.",
        ].join("\n"),
        stats: { ttftMs: 520, totalMs: 14200, tokens: 412 },
      },
    ],
  },
];

try {
  localStorage.setItem("idletoken.chat.v2", JSON.stringify(SEED));
} catch {
  /* harness only */
}

function Lab() {
  const [lang, setLang] = useState<Lang>("en");
  const [theme, setTheme] = useState<"dark" | "light">("light");
  document.documentElement.setAttribute("data-theme", theme);
  return (
    <LangContext.Provider value={{ lang, setLang }}>
      <div className="app">
        <header className="topbar">
          <div className="brand">
            <span className="brand__name">IdleToken</span>
          </div>
          <div className="topbar__spacer" />
          <button className="chip" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>
            {theme === "dark" ? "☀" : "☾"}
          </button>
          <button className="chip" onClick={() => setLang(lang === "en" ? "zh" : "en")}>
            {lang === "en" ? "中文" : "EN"}
          </button>
        </header>
        <Chat api={api} apiToken="" modelId="qwen3.5-4b" quant="Q4_K_M" maxTokens={0} identity={{ name: "Zhidong Gao", hue: 268 }} onGoCluster={() => {}} />
      </div>
    </LangContext.Provider>
  );
}

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <Lab />
  </React.StrictMode>
);
