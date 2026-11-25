import { svelte } from "@sveltejs/vite-plugin-svelte";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig, searchForWorkspaceRoot } from "vite";

export default defineConfig({
  plugins: [svelte(), tailwindcss()],
  base: "./",
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    // The settings' About page shows the license, kept at the top of the
    // repository
    fs: { allow: [searchForWorkspaceRoot(process.cwd()), "../LICENSE"] },
    // Main-process sources are bundled by esbuild; a change there must not
    // reload the renderer and wipe the open conversation
    watch: { ignored: ["**/electron/**", "**/dist-electron/**"] },
  },
  build: {
    outDir: "dist",
  },
});
