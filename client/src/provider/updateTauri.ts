// Real update provider: RPC to src-tauri/src/update.rs, which owns the
// signature check and the installer handover.
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import type { UpdateInfo, UpdateProgress, UpdateProvider, UpdateState } from "./update";

// Event name must match src-tauri/src/update.rs.
const EV_PROGRESS = "update-progress";

export const updateTauriProvider: UpdateProvider = {
  async check(channel: string): Promise<UpdateInfo | null> {
    // Rejections propagate on purpose: "could not reach the update server" is
    // a different sentence from "you are up to date", and the caller shows it.
    return await invoke<UpdateInfo | null>("update_check", { channel });
  },

  async download(): Promise<number> {
    return await invoke<number>("update_download");
  },

  async install(): Promise<void> {
    // The app is replaced and relaunched, so on success this promise never
    // settles — the webview is gone before it could.
    await invoke("update_install");
  },

  async state(channel: string): Promise<UpdateState> {
    const raw = await invoke<Omit<UpdateState, "source">>("update_state", { channel });
    return { ...raw, source: "shell" };
  },

  onProgress(cb: (p: UpdateProgress) => void): () => void {
    const un = listen<UpdateProgress>(EV_PROGRESS, (e) => cb(e.payload));
    return () => {
      un.then((f) => f());
    };
  },
};
