/** The OS owns startup registration. Reading settings must never recreate it. */
export function createAutostartController(io: {
  read: () => Promise<boolean | null>;
  write: (enabled: boolean) => Promise<void>;
  state: (enabled: boolean) => void;
  busy: (pending: boolean) => void;
  error: (message: string | null) => void;
}) {
  let revision = 0;
  let pending = false;
  let disposed = false;
  return {
    async refresh() {
      if (pending || disposed) return;
      const request = ++revision;
      try {
        const actual = await io.read();
        if (!disposed && request === revision && actual !== null) {
          io.state(actual);
          io.error(null);
        }
      } catch (error) {
        if (!disposed && request === revision) io.error(String(error));
      }
    },
    async set(enabled: boolean) {
      if (disposed || pending) throw new Error("A startup setting change is already in progress.");
      ++revision; // Invalidate an earlier read before changing the registration.
      pending = true;
      io.busy(true);
      io.error(null);
      let observed = false;
      try {
        await io.write(enabled);
        const actual = await io.read();
        if (actual === null) throw new Error("Startup settings are unavailable outside the desktop app.");
        observed = true;
        if (!disposed) io.state(actual);
        if (actual !== enabled) throw new Error("The operating system did not apply the startup setting.");
      } catch (error) {
        // Registration can succeed before a later OS operation fails. Observe
        // that partial result instead of leaving the toggle at a stale value.
        if (!observed) {
          try {
            const actual = await io.read();
            if (!disposed && actual !== null) io.state(actual);
          } catch { /* Keep the original failure visible. */ }
        }
        if (!disposed) io.error(String(error));
        throw error;
      } finally {
        pending = false;
        if (!disposed) io.busy(false);
      }
    },
    dispose() {
      disposed = true;
      ++revision;
    },
  };
}
