<script lang="ts">
  // The legal notices the license asks an interactive program to show: who
  // holds the copyright, that there is no warranty, that Iris may be passed
  // on under the license, and the license itself. Besides, the notices: the
  // makers and licenses of the speech models and voices Iris downloads.
  import license from "../../../../LICENSE?raw";
  import notices from "../../../../NOTICES.md?raw";
  import { build, version } from "../../../package.json";
  import { renderMarkdown } from "../../lib/markdown";

  const SOURCE = "https://github.com/clstaudt/iris";

  let shown = $state<"license" | "notices" | null>(null);
  const show = (what: "license" | "notices") => (shown = shown === what ? null : what);
</script>

<div class="flex max-w-[640px] flex-col gap-4" data-page="about">
  <div class="flex flex-col gap-1">
    <h3 class="text-[15px] font-semibold text-ink">Iris</h3>
    <p class="text-[12px] tabular-nums text-ink-3">Version {version}</p>
  </div>
  <div class="flex flex-col gap-2 text-[12.5px] leading-relaxed text-ink-2">
    <p>{build.copyright}</p>
    <p>
      Iris is free software: you can redistribute it and/or modify it under the terms of the GNU
      Affero General Public License, version 3 or any later version. It comes without any warranty.
    </p>
  </div>
  <div class="flex flex-wrap items-center gap-2">
    <button class="s-btn" aria-expanded={shown === "license"} onclick={() => show("license")}>
      License
    </button>
    <button class="s-btn" aria-expanded={shown === "notices"} onclick={() => show("notices")}>
      Notices
    </button>
    <button class="s-btn" onclick={() => window.open(SOURCE, "_blank")}>Source code</button>
  </div>
  {#if shown === "license"}
    <pre class="license selectable">{license}</pre>
  {:else if shown === "notices"}
    <div class="notices selectable">{@html renderMarkdown(notices)}</div>
  {/if}
</div>

<style>
  .license {
    margin: 0;
    padding: 0.85rem 1rem;
    border-radius: 0.6rem;
    border: 1px solid var(--color-line);
    background: var(--color-raised);
    font-family: var(--font-mono);
    font-size: 11px;
    line-height: 1.55;
    color: var(--color-ink-2);
    white-space: pre-wrap;
  }
  .notices {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    font-size: 12px;
    line-height: 1.55;
    color: var(--color-ink-2);
  }
  /* The page's own heading is About: the file's title is not repeated */
  .notices :global(h1) {
    display: none;
  }
  .notices :global(h2) {
    margin-top: 0.5rem;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--color-ink-3);
  }
  .notices :global(table) {
    border-collapse: collapse;
    width: 100%;
  }
  .notices :global(th) {
    text-align: left;
    font-weight: 500;
    color: var(--color-ink-3);
  }
  .notices :global(th),
  .notices :global(td) {
    padding: 0.4rem 0.75rem 0.4rem 0;
    border-bottom: 1px solid var(--color-line);
    vertical-align: top;
  }
  .notices :global(a) {
    color: var(--color-ink);
    text-decoration: underline;
    text-decoration-color: var(--color-line-strong);
    text-underline-offset: 2px;
  }
  .notices :global(a:hover) {
    text-decoration-color: var(--color-accent);
  }
</style>
