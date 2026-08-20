// Honor badges: a mirror of the backend's deriveHonors plus presentation
// metadata, shared by the web portal and the client.
//
// The KEY is the contract; the label is presentation. The backend derives which
// honors a provider has and returns their keys; every user-visible string is
// resolved here and then run through the consumer's t(). A backend-supplied
// label is only a last-resort fallback for a key this table doesn't know yet —
// it must never win over the local table, or a backend that speaks one language
// stamps that language onto every locale (which is exactly what happened: the
// English site rendered Chinese badges because `h.label ?? HONOR_LABEL[h.key]`
// preferred whatever the API sent).

export interface Honor { key: string; label?: string; title?: string }

// These strings are the source language of the consumer UI and double as i18n
// keys: consumers render them through t(), whose dictionary is keyed on them.
export const HONOR_LABEL: Record<string, string> = {
  reliable: 'Reliable', fast: 'Fast', popular: 'Popular', veteran: 'Veteran', rising: 'Rising',
};
export const HONOR_TITLE: Record<string, string> = {
  reliable: '24h uptime >= 95%', fast: 'Average latency < 2.5s', popular: 'Served 100+ requests',
  veteran: 'Member for 90 days', rising: 'Joined in the last two weeks',
};

/** The label to render for an honor key. Local table first; a backend-supplied
 *  label only covers keys added server-side ahead of this package. */
export function honorLabel(h: Honor): string {
  return HONOR_LABEL[h.key] ?? h.label ?? h.key;
}
/** The tooltip for an honor key (why the provider earned it). */
export function honorTitle(h: Honor): string | undefined {
  return HONOR_TITLE[h.key];
}

/** Derive honors on the front end (same rules as the backend's deriveHonors),
 *  used when a listing carries stats but no server-derived honors. */
export function honorsFrom(input: {
  uptime24h: number | null;
  avgLatencyMs: number | null;
  served: number;
  memberSince: string | Date;
}): Honor[] {
  const keys: string[] = [];
  const ageDays = (Date.now() - new Date(input.memberSince).getTime()) / 86_400_000;
  if (input.uptime24h != null && input.uptime24h >= 95) keys.push('reliable');
  if (input.avgLatencyMs != null && input.avgLatencyMs > 0 && input.avgLatencyMs < 2500) keys.push('fast');
  if (input.served >= 100) keys.push('popular');
  if (ageDays >= 90) keys.push('veteran');
  else if (ageDays <= 14) keys.push('rising');
  return decorateHonors(keys.map((key) => ({ key })));
}

/** Resolve {key} → {key,label,title} for display. The backend returns keys. */
export function decorateHonors(honors: Honor[]): Honor[] {
  return honors.map((h) => ({ key: h.key, label: honorLabel(h), title: honorTitle(h) }));
}
