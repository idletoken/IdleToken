// Chat avatars.
//
// The user's avatar has to be the SAME face the platform shows. The portal's
// identity is a generated one — `hsl(hue 62% 52%)` behind the name's initials,
// where the hue is the account's chosen `avatarHue` (the gateway derives one
// from the user id when unset). There is no image upload anywhere in the
// product, so "the avatar I set" means exactly that hue + name. This file
// mirrors packages/shared-ui/src/{Avatar,format}.ts deliberately: the client
// cannot import from that package (separate build, and it must keep working
// with no platform at all), so the derivation is duplicated — if you change it
// there, change it here, or one user gets two different faces.
import type { PlatformMe } from "./platform";

/** First two letters/digits, uppercased. Matches shared-ui's `initials`. */
export function initials(name: string): string {
  return name.replace(/[^\p{L}\p{N}]/gu, "").slice(0, 2).toUpperCase() || "··";
}

/** Stable hue from a string. Matches shared-ui's `hueFor`. */
export function hueFor(id: string): number {
  let h = 0;
  for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) & 0xffffff;
  return h % 360;
}

/** What the chat needs to draw a user: a label and a colour, or nothing. */
export interface UserIdentity {
  name: string;
  hue: number;
}

/** Identity from the platform profile, or null when signed out / local-only.
 *  Falls back through displayName -> username -> the email's local part, which
 *  is the same order the portal uses for its own header. */
export function identityFrom(me: PlatformMe | null | undefined): UserIdentity | null {
  if (!me) return null;
  const name = me.displayName || me.username || me.email.split("@")[0] || me.email;
  return { name, hue: me.avatarHue ?? hueFor(me.id) };
}

/** The person's avatar: their initials on their colour when we know who they
 *  are, otherwise a neutral glyph (drawn in CSS). */
export function UserAvatar(props: { identity: UserIdentity | null }) {
  if (!props.identity) return <div className="chat-avatar chat-avatar--user" />;
  return (
    <div
      className="chat-avatar chat-avatar--person"
      style={{ background: `hsl(${props.identity.hue} 62% 52%)` }}
      title={props.identity.name}
    >
      {initials(props.identity.name)}
    </div>
  );
}
