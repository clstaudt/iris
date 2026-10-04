<script lang="ts">
  import { STATE_LABELS, store } from "../lib/state.svelte";
  import Circle from "./Circle.svelte";

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
    class="absolute text-[11.5px] tracking-wide text-ink-3"
    style="top: {RING_BOTTOM + GAP}px"
  >
    {store.state === "dormant"
      ? `Waiting for “${store.config?.wake_word ?? ""}”`
      : (STATE_LABELS[store.state] ?? "")}
  </div>
</div>
