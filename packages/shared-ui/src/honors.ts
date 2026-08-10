// Honor badges: a mirror of the backend's deriveHonors plus presentation
// metadata, shared by the web portal and the client.
// The backend (gateway src/catalog/identity.ts) derives {key,label}; this module
// adds title/label for display, derives honors locally for samples and fallbacks,
// and merges the two with the backend's honors taking precedence.

export interface Honor { key: string; label: string; title?: string }

// These strings are the source language of the consumer UI and double as i18n
// keys: consumers render them through t(), whose dictionary is keyed on them.
export const HONOR_LABEL: Record<string, string> = {
  reliable: 'Reliable', fast: 'Fast', popular: 'Popular', veteran: 'Veteran', rising: 'Rising',
};
export const HONOR_TITLE: Record<string, string> = {
  reliable: '24h uptime >= 95%', fast: 'Average latency < 2.5s', popular: 'Served 100+ requests',
  veteran: 'Member for 90 days', rising: 'Joined in the last two weeks',
};

/** Derive honors on the front end (same rules as the backend's deriveHonors),
 *  for samples and as a fallback when there is no owner. */
export function honorsFrom(input: {
  uptime24h: number | null;
  avgLatencyMs: number | null;
  served: number;
  memberSince: string | Date;
}): Honor[] {
  const out: Honor[] = [];
  const ageDays = (Date.now() - new Date(input.memberSince).getTime()) / 86_400_000;
  if (input.uptime24h != null && input.uptime24h >= 95) out.push({ key: 'reliable', label: HONOR_LABEL.reliable });
  if (input.avgLatencyMs != null && input.avgLatencyMs > 0 && input.avgLatencyMs < 2500) out.push({ key: 'fast', label: HONOR_LABEL.fast });
  if (input.served >= 100) out.push({ key: 'popular', label: HONOR_LABEL.popular });
  if (ageDays >= 90) out.push({ key: 'veteran', label: HONOR_LABEL.veteran });
  else if (ageDays <= 14) out.push({ key: 'rising', label: HONOR_LABEL.rising });
  return out.map((h) => ({ ...h, title: HONOR_TITLE[h.key] }));
}

/** Fill in title/label uniformly (the backend returns only {key,label}). */
export function decorateHonors(honors: Honor[]): Honor[] {
  return honors.map((h) => ({ ...h, label: h.label ?? HONOR_LABEL[h.key], title: HONOR_TITLE[h.key] ?? h.label }));
}
