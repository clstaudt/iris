<script lang="ts">
  import type { Snippet } from "svelte";

  /** A step of the walkthrough: its title, at most one short line, and what
   *  it asks, which scrolls when the window is small. */
  let { title, line = "", children }: { title: string; line?: string; children?: Snippet } =
    $props();
</script>

<div class="flex min-h-0 flex-1 flex-col">
  <div class="flex shrink-0 flex-col items-center gap-1.5 pb-5 text-center">
    <h1 class="text-[20px] font-semibold text-ink">{title}</h1>
    {#if line}
      <p class="text-[13px] text-ink-3">{line}</p>
    {/if}
  </div>
  <div class="-mx-3 flex min-h-0 flex-1 flex-col gap-5 overflow-y-auto px-3 pb-2">
    {@render children?.()}
  </div>
</div>
