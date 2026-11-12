<script lang="ts">
  import { STATE_LABELS, store } from "../lib/state.svelte";
  import Circle from "./Circle.svelte";
  import Progress from "./Progress.svelte";

  // The orb canvas is wider than its ring (the halo needs room to fade), so
  // the label is placed from the ring's edge rather than the canvas edge
  const ORB = 112;
  const RING_BOTTOM = ORB / 2 + ORB * 0.25;
  const GAP = 16;
</script>

<!-- Compact status under the composer: the orb and what it means -->
<div class="relative flex shrink-0 flex-col items-center" style="height: {RING_BOTTOM + GAP + 30}px">
  <button
    onclick={() => store.togglePause()}
    disabled={store.state === "starting"}
    class="rounded-full"
    title={store.paused ? "Resume the conversation" : "Pause the conversation"}
  >
    <Circle size={ORB} />
  </button>
  <div
    class="absolute flex flex-col items-center gap-1.5 text-[11.5px] tracking-wide text-ink-3"
    style="top: {RING_BOTTOM + GAP}px"
  >
    <!-- A model download (first start, or one asked for in Settings) -->
    {#if store.setup}
      {store.setup.label}
      {#key store.setup.model}
        <Progress
          value={store.setup.total ? store.setup.done / store.setup.total : 0}
          class="w-[180px]"
        />
      {/key}
    {:else}
      {STATE_LABELS[store.state] ?? ""}
    {/if}
  </div>
</div>
