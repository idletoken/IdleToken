import { defineConfig } from "vite";

export default defineConfig({
  build: {
    ssr: "scripts/resource_estimate_check.ts",
    outDir: "../build/client-resource-check",
    emptyOutDir: true,
    minify: false,
    rollupOptions: {
      output: { entryFileNames: "resource_estimate_check.mjs" },
    },
  },
});
