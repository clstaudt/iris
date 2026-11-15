<script lang="ts">
  import { STATE_LABELS, store } from "../lib/state.svelte";
  import { percent } from "../lib/voices";
  import { wakeHint } from "../lib/wakehint";
  import Circle from "./Circle.svelte";
  import Progress from "./Progress.svelte";

  // A voice the conversation waits on, being made ready
  const job = $derived(store.runningJobs.find((j) => j.origin === "conversation"));

  // The orb canvas is wider than its ring (the halo needs room to fade), so
  // the label is placed from the ring's edge rather than the canvas edge
  const ORB = 112;
  const RING_BOTTOM = ORB / 2 + ORB * 0.25;
  const GAP = 16;

  // Asleep with the wake word on: now and then the status names the word
  const wakeWord = $derived(store.state === "dormant" ? store.config?.wake_word?.trim() ?? "" : "");
  const asleep = $derived(!!wakeWord);
  let hinting = $state(false);
  $effect(() => {
    if (asleep) return wakeHint((on) => (hinting = on));
  });
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
    {:else if job}
      <span class="tabular-nums" data-job={job.id}>
        {job.title}{percent(job) !== null && job.state === "downloading" ? ` ${percent(job)} %` : ""}
      </span>
      {#key job.id}
        <Progress
          value={job.total ? job.done / job.total : 0}
          waiting={job.state === "making"}
          class="w-[180px]"
        />
      {/key}
    {:else if asleep}
      <span class="grid justify-items-center">
        <span class="wake-label" class:shown={!hinting} aria-hidden={hinting}>{STATE_LABELS.dormant}</span>
        <span class="wake-label" class:shown={hinting} aria-hidden={!hinting} data-wake-hint
          >Say "{wakeWord}" to wake</span>
      </span>
    {:else}
      {STATE_LABELS[store.state] ?? ""}
    {/if}
  </div>
</div>

<style>
  /* The sleeping label and the hint share a place and fade into each other */
  .wake-label {
    grid-area: 1 / 1;
    white-space: nowrap;
    opacity: 0;
    transition: opacity 900ms ease-in-out;
  }
  .wake-label.shown {
    opacity: 1;
  }
</style>
