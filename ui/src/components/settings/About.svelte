<script lang="ts">
  // The legal notices the license asks an interactive program to show: who
  // holds the copyright, that there is no warranty, that Iris may be passed
  // on under the license, and the license itself
  import license from "../../../../LICENSE?raw";
  import { build, version } from "../../../package.json";

  const SOURCE = "https://github.com/clstaudt/iris";

  let licenseShown = $state(false);
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
    <button class="s-btn" aria-expanded={licenseShown} onclick={() => (licenseShown = !licenseShown)}>
      License
    </button>
    <button class="s-btn" onclick={() => window.open(SOURCE, "_blank")}>Source code</button>
  </div>
  {#if licenseShown}
    <pre class="license selectable">{license}</pre>
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
</style>
