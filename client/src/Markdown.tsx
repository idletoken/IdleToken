// Markdown rendering for assistant replies.
//
// WHY (2026-08-11). The thread rendered replies as `white-space: pre-wrap`
// plain text, so every model answer arrived full of literal `**bold**`,
// `### heading` and `* item` — the single thing that made the chat read as
// unfinished next to ChatGPT/Claude/Gemini, all of which render markdown.
//
// WHY A LIBRARY. Hand-rolling looks like an afternoon (bold, headings, fences)
// and then never ends: nested lists, fences containing backticks, tables,
// links, escaping. And a hand-rolled renderer that reaches for
// dangerouslySetInnerHTML turns model output into an XSS vector in a desktop
// app that also holds the user's platform token. react-markdown builds a React
// tree and never touches innerHTML, so untrusted text stays text — raw HTML in
// the reply is inert by default because we do not enable rehype-raw.
//
// Streaming: this re-parses on every delta. At LAN token rates that is far
// below the cost of the token itself, and it is what makes a table or a fence
// resolve as it arrives rather than snapping into place at the end.
import { memo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useI18n } from "./i18n";

function CodeBlock(props: { text: string; lang: string }) {
  const { t } = useI18n();
  const [copied, setCopied] = useState(false);
  return (
    <div className="md-code">
      <div className="md-code__bar">
        <span className="md-code__lang">{props.lang || "text"}</span>
        <button
          className="md-code__copy"
          onClick={() => {
            void navigator.clipboard.writeText(props.text).then(
              () => { setCopied(true); setTimeout(() => setCopied(false), 1400); },
              () => {},   // clipboard denied: stay silent rather than claim success
            );
          }}
        >
          {copied ? t("chat.copied") : t("chat.copy")}
        </button>
      </div>
      <pre><code>{props.text}</code></pre>
    </div>
  );
}

/** Assistant text as markdown. Memoised on the text so an unrelated re-render
 *  (a sibling message streaming) does not re-parse every finished message. */
const Markdown = memo(function Markdown(props: { text: string }) {
  return (
    <div className="md">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          // Fenced blocks get the bar + copy button; inline code stays inline.
          code({ className, children, ...rest }) {
            const text = String(children ?? "");
            // react-markdown 10 drops the `inline` prop: a fenced block is the
            // one whose parent is <pre>, which shows up here as a trailing
            // newline plus (usually) a language class. Checking for the newline
            // keeps single-line fences working too, since remark always ends a
            // fence's content with one.
            const lang = /language-(\w+)/.exec(className || "")?.[1] ?? "";
            const fenced = lang !== "" || text.includes("\n");
            if (!fenced) return <code className="md-inline" {...rest}>{children}</code>;
            return <CodeBlock text={text.replace(/\n$/, "")} lang={lang} />;
          },
          // <pre> is emitted around our own block, which already renders one.
          pre({ children }) { return <>{children}</>; },
          // Links open in the system browser, never navigate the app shell —
          // a model-authored href that replaced the window would strand the
          // user in a page with no way back to the client.
          a({ href, children }) {
            return (
              <a href={href} target="_blank" rel="noreferrer noopener"
                 onClick={(e) => {
                   e.preventDefault();
                   if (!href) return;
                   void import("@tauri-apps/plugin-shell")
                     .then((m) => m.open(href))
                     .catch(() => window.open(href, "_blank", "noopener"));
                 }}>
                {children}
              </a>
            );
          },
          // Remote images are NOT loaded (2026-08-20, audit A-P2-3).
          //
          // A reply is untrusted text, and `![](https://someone/pixel.png)` in
          // one turns this window into a beacon: the fetch happens with no
          // interaction, and it tells whoever owns that host that this machine
          // read this reply — from the product whose entire claim is that
          // nothing about a conversation leaves the machine. There is nothing
          // the model can legitimately illustrate an answer with, either: it
          // has no images to serve.
          //
          // The CSP in tauri.conf.json (`img-src 'self' data:`) is the
          // backstop; this renderer is the part that says so out loud instead
          // of leaving a silently broken image icon. `data:` images — the ones
          // that arrive inside the reply and need no network — still render.
          img({ src, alt }) {
            const inline = typeof src === "string" && src.startsWith("data:");
            if (inline) return <img src={src} alt={alt ?? ""} className="md-img" />;
            return (
              <span className="md-img-blocked" title={typeof src === "string" ? src : undefined}>
                {alt ? `🖼 ${alt}` : "🖼"}
              </span>
            );
          },
          table({ children }) { return <div className="md-tablewrap"><table>{children}</table></div>; },
        }}
      >
        {props.text}
      </ReactMarkdown>
    </div>
  );
});

export default Markdown;
