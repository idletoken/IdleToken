// Pure formatting helpers (no framework dependency), shared by the web portal
// and the client.

/** Deterministic hue: derives 0-359 from any id (the fallback when no avatar
 *  hue is set). */
export function hueFor(id: string): number {
  let h = 0;
  for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) & 0xffffff;
  return h % 360;
}

/** Name initials: the first two letters or digits, uppercased; ".." when empty. */
export function initials(name: string): string {
  return name.replace(/[^\p{L}\p{N}]/gu, '').slice(0, 2).toUpperCase() || '··';
}

/** Short context-length label: 131072 -> 128K, 1048576 -> 1M, absent -> em dash. */
export function ctxShort(c?: number): string {
  return c ? (c >= 1_048_576 ? '1M' : `${c / 1024}K`) : '—';
}

/**
 * Display identity of a service, mirroring the service tuple without changing
 * the API model id users send. The context window belongs immediately to the
 * model name, e.g. opus-5(128K):Q4_K_M.
 *
 * EVERY tier is labelled since 2026-09-02, not just 1M. With two windows the
 * unlabelled case could only mean 256K, so omitting it was unambiguous; with
 * three it would leave a buyer unable to tell a 128K service from a 256K one —
 * and the window is one of the two things they filter on. `ctx = 0` is a
 * pre-2026-08 agent that never declared one; it stays unlabelled because
 * inventing a number there would be a guess about someone else's machine.
 *
 * Shared because two pages render it: the cluster page (what is on sale now)
 * and the public user page (the recent service records). A buyer who sees the
 * same service spelled two ways reads it as two different products.
 */
export function serviceModelLabel(model: string, quant?: string | null, ctx?: number | null): string {
  return `${model}${ctx ? `(${ctxShort(ctx)})` : ''}${quant ? `:${quant}` : ''}`;
}

/** Short latency label: milliseconds -> seconds with one decimal, absent -> em dash. */
export function latShort(ms: number | null): string {
  return ms != null ? `${(ms / 1000).toFixed(1)}s` : '—';
}

/**
 * Millicredits -> a credits display string.
 * - Default (the web ledger): a localized number, negatives with the native "-",
 *   and a plus sign added by the caller when it wants one.
 * - `signed` (the client ledger): always signed, using U+2212 for the minus,
 *   rounded to an integer when |value| >= 100 and with trailing zeros stripped
 *   otherwise.
 */
export function fmtCredits(milli: number, opts?: { signed?: boolean }): string {
  const v = milli / 1000;
  if (opts?.signed) {
    const mag = Math.abs(v) >= 100
      ? Math.round(Math.abs(v)).toString()
      : Math.abs(v).toFixed(2).replace(/\.?0+$/, '');
    return `${v < 0 ? '−' : '+'}${mag}`;
  }
  return (Number.isInteger(v) ? v : Number(v.toFixed(3))).toLocaleString();
}
