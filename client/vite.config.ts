import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";

// Tauri expects a fixed dev port and serves the built assets from dist/.
// `clearScreen: false` keeps Rust/cargo logs visible during `tauri dev`.
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      // Shared consumer-side UI (the in-repo packages/shared-ui sources, the same
      // source of truth as the portal).
      // Moved out of platform/ on 2026-08-08: it is a pure UI helper (143 lines,
      // no business logic), and living inside the commercial layer meant **the
      // public repo's client would not compile** -- that directory never ships
      // with the mirror.
      "@idletoken/shared-ui": fileURLToPath(new URL("../packages/shared-ui/src", import.meta.url)),
      // The shared-ui sources live outside the client directory, so Node
      // resolution walks up from their own directory looking for node_modules --
      // and on the build machine only the shared-ui sources are synced (without
      // platform's workspace dependencies), so react is not on that path. Pin it
      // explicitly to the client's own copy, which also guarantees a single react
      // instance across the app.
      react: fileURLToPath(new URL("./node_modules/react", import.meta.url)),
      "react-dom": fileURLToPath(new URL("./node_modules/react-dom", import.meta.url)),
    },
    dedupe: ["react", "react-dom"],
  },
  clearScreen: false,
  server: {
    port: 1420,
    strictPort: true,
  },
  build: {
    target: "es2021",
    outDir: "dist",
  },
});
