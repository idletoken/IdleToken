// Development-only update simulation, used when the client runs in a plain
// browser (no Tauri shell, therefore nothing to update). It exists so the
// prompt, the progress bar and the failure copy can be built and reviewed
// offline — the states are otherwise unreachable without cutting a release.
//
// It never claims to be real: every version it reports carries the
// `dev-fixture` marker and `state().source` says so, which is what the UI
// keys off to label the panel (philosophy 9/15).
import type { UpdateInfo, UpdateProgress, UpdateProvider, UpdateState } from "./update";

const FIXTURE_VERSION = "0.0.0-dev-fixture";

class UpdateFixture implements UpdateProvider {
  private subs = new Set<(p: UpdateProgress) => void>();
  private bytes = 0;

  async check(channel: string): Promise<UpdateInfo | null> {
    return {
      version: FIXTURE_VERSION,
      currentVersion: FIXTURE_VERSION,
      notes: "Simulated release notes — there is no shell to update in a browser.",
      date: null,
      channel,
    };
  }

  async download(): Promise<number> {
    const total = 96 * 1024 * 1024;
    this.bytes = 0;
    await new Promise<void>((resolve) => {
      const step = () => {
        this.bytes = Math.min(total, this.bytes + total / 12);
        this.subs.forEach((cb) => cb({ downloaded: this.bytes, total }));
        if (this.bytes >= total) resolve();
        else setTimeout(step, 120);
      };
      setTimeout(step, 120);
    });
    return this.bytes;
  }

  async install(): Promise<void> {
    // Deliberately a failure rather than a no-op: a fixture that pretends to
    // have installed something is exactly the hollow success philosophy 15
    // forbids.
    throw new Error("dev-fixture: there is no application to install into from a browser");
  }

  async state(channel: string): Promise<UpdateState> {
    return {
      currentVersion: FIXTURE_VERSION,
      feed: `dev-fixture://${channel}`,
      feedOverridden: false,
      pending: false,
      downloaded: this.bytes || null,
      source: "dev-fixture",
    };
  }

  onProgress(cb: (p: UpdateProgress) => void): () => void {
    this.subs.add(cb);
    return () => this.subs.delete(cb);
  }
}

export const updateFixtureProvider: UpdateProvider = new UpdateFixture();
