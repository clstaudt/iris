import { svelte } from "@sveltejs/vite-plugin-svelte";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig, searchForWorkspaceRoot, type Plugin } from "vite";

/** Lists the packages from node_modules that the bundle holds, as
 *  dist/bundled-packages.json, for the app's third-party notices
 *  (packaging/notices.py) */
function bundledPackages(): Plugin {
  return {
    name: "bundled-packages",
    apply: "build",
    generateBundle(_, bundle) {
      const names = new Set<string>();
      for (const output of Object.values(bundle)) {
        if (output.type !== "chunk") continue;
        for (const id of output.moduleIds) {
          const name = id.match(/\/node_modules\/((?:@[^/]+\/)?[^/]+)\//)?.[1];
          if (name) names.add(name);
        }
      }
      this.emitFile({
        type: "asset",
        fileName: "bundled-packages.json",
        source: JSON.stringify([...names].sort(), null, 2) + "\n",
      });
    },
  };
}

export default defineConfig({
  plugins: [svelte(), tailwindcss(), bundledPackages()],
  base: "./",
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    // The settings' About page shows the license and the notices, kept at
    // the top of the repository
    fs: { allow: [searchForWorkspaceRoot(process.cwd()), "../LICENSE", "../NOTICES.md"] },
    // Main-process sources are bundled by esbuild; a change there must not
    // reload the renderer and wipe the open conversation
    watch: { ignored: ["**/electron/**", "**/dist-electron/**"] },
  },
  build: {
    outDir: "dist",
  },
});
