// Getting-started code blocks (cURL / OpenAI / Anthropic / Claude Code), shared
// by every consumer. apiBase is injected (web = https://api.idletoken.ai; the
// client can point at localhost or anywhere else), so no domain is hardcoded.
import { useState } from 'react';

export function connectSnippets(apiBase: string): Record<string, string> {
  return {
    curl: `curl ${apiBase}/v1/chat/completions \\
  -H "Authorization: Bearer $IDLETOKEN_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"model":"dsv4-flash","messages":[{"role":"user","content":"Hello"}]}'`,
    openai: `from openai import OpenAI
client = OpenAI(base_url="${apiBase}/v1",
                api_key="itk_live_...")
r = client.chat.completions.create(
  model="dsv4-flash",
  messages=[{"role":"user","content":"Hello"}])
print(r.choices[0].message.content)`,
    anthropic: `import anthropic
client = anthropic.Anthropic(base_url="${apiBase}",
                             api_key="itk_live_...")
m = client.messages.create(model="dsv4-flash", max_tokens=512,
  messages=[{"role":"user","content":"Hello"}])
print(m.content[0].text)`,
    claudecode: `# ~/.claude/settings.json, or environment variables
export ANTHROPIC_BASE_URL="${apiBase}"
export ANTHROPIC_API_KEY="itk_live_..."
# then run claude as usual`,
  };
}

// Optional `t`: a consumer (the web portal) passes a translation function to
// localize the heading. Without it the default string is used as is.
export function QuickStart({ apiBase, t }: { apiBase: string; t?: (s: string) => string }) {
  const tr = t ?? ((s: string) => s);
  const snippets = connectSnippets(apiBase);
  const [tab, setTab] = useState<keyof typeof snippets>('curl');
  const tabs: Array<[keyof typeof snippets, string]> = [
    ['curl', 'cURL'], ['openai', 'OpenAI SDK'], ['anthropic', 'Anthropic SDK'], ['claudecode', 'Claude Code'],
  ];
  return (
    <>
      <h3 className="mk-psec__t" style={{ marginTop: 24 }}>{tr('Quick start')}</h3>
      <div className="qs">
        <div className="qs-tabs">
          {tabs.map(([k, l]) => <button key={k} className={`qs-tab ${tab === k ? 'is-on' : ''}`} onClick={() => setTab(k)}>{l}</button>)}
        </div>
        <pre className="qs-code"><code>{snippets[tab]}</code></pre>
      </div>
    </>
  );
}
