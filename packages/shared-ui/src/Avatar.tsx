// Round letter avatar (presentation only). The caller supplies the hue
// (owner.avatarHue, falling back to hueFor(id)).
import { initials } from './format';

export function Avatar({ hue, name, size = 40 }: { hue: number; name: string; size?: number }) {
  return (
    <span
      className="mk-avatar"
      style={{ width: size, height: size, background: `hsl(${hue} 62% 52%)`, color: '#fff', fontSize: size * 0.4 }}
      aria-hidden
    >
      {initials(name)}
    </span>
  );
}
