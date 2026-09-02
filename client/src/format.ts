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

export function pct(part: number, whole: number): number {
  if (whole <= 0) return 0;
  return Math.max(0, Math.min(100, (part / whole) * 100));
}
