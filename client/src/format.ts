// Byte / number formatting for the readouts. Binary units (GiB) to match the
// engine's own reporting in resource.c.
const GiB = 1024 ** 3;
const MiB = 1024 ** 2;

export function fmtBytes(b: number): string {
  if (b >= GiB) return `${(b / GiB).toFixed(1)} GiB`;
  if (b >= MiB) return `${(b / MiB).toFixed(0)} MiB`;
  return `${b} B`;
}

/** GiB to one decimal, ALWAYS ROUNDED DOWN.
 *
 *  Not toFixed(1): rounding 15.674 up to 15.7 in one readout while the capacity
 *  line floors the same bytes to 15.6 puts two different numbers for one
 *  quantity on one card (seen on a 16 GB test node, 2026-09-02). Flooring is
 *  also the safe
 *  direction for anything describing memory you HAVE — an available figure
 *  rounded up can read as covering a requirement it does not cover. */
export function floorGiB1(b: number): number {
  return Math.floor((b / GiB) * 10) / 10;
}

// Just the numeric part, for readouts that render the unit separately.
export function fmtGiB(b: number): { value: string; unit: string } {
  return { value: floorGiB1(b).toFixed(1), unit: "GiB" };
}

/** Short context-window label: 131072 -> "128K", 1048576 -> "1M".
 *  Mirrors ctxShort() in packages/shared-ui so the client and the portal name
 *  the same window the same way — the portal labels every service with its
 *  window since 2026-09-02, and two spellings would read as two products. */
export function ctxLabel(tokens: number): string {
  return tokens >= 1_048_576 ? "1M" : `${Math.round(tokens / 1024)}K`;
}

/** Compact a cumulative counter before it becomes wide enough to reflow the
 * cluster activity row. The exact value remains available to callers for a
 * tooltip; this function is only the bounded-width visual label. */
export function compactCount(value: number, lang: "en" | "zh"): string {
  const safe = Number.isFinite(value) ? Math.max(0, Math.trunc(value)) : 0;
  return new Intl.NumberFormat(lang === "zh" ? "zh-CN" : "en-US", {
    notation: safe >= 10_000 ? "compact" : "standard",
    maximumFractionDigits: safe >= 10_000 ? 1 : 0,
  }).format(safe);
}

export function pct(part: number, whole: number): number {
  if (whole <= 0) return 0;
  return Math.max(0, Math.min(100, (part / whole) * 100));
}
