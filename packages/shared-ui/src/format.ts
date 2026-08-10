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
