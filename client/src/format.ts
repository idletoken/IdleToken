// Byte / number formatting for the readouts. Binary units (GiB) to match the
// engine's own reporting in resource.c.
const GiB = 1024 ** 3;
const MiB = 1024 ** 2;

export function fmtBytes(b: number): string {
  if (b >= GiB) return `${(b / GiB).toFixed(1)} GiB`;
  if (b >= MiB) return `${(b / MiB).toFixed(0)} MiB`;
  return `${b} B`;
}

// Just the numeric part, for readouts that render the unit separately.
export function fmtGiB(b: number): { value: string; unit: string } {
  return { value: (b / GiB).toFixed(1), unit: "GiB" };
}

export function pct(part: number, whole: number): number {
  if (whole <= 0) return 0;
  return Math.max(0, Math.min(100, (part / whole) * 100));
}
