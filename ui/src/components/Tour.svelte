<script lang="ts">
  import { onMount } from "svelte";
  import { fade } from "svelte/transition";
  import { calloutAt, TOUR, type Box } from "../lib/tour";
  import { tour } from "../lib/tour.svelte";

  /** How far the light reaches past a control */
  const PAD = 6;

  type Hole = Box & { name: string; radius: number };

  let { delay = 0 }: { delay?: number } = $props();

  let holes = $state<Hole[]>([]);
  let innerWidth = $state(0);
  let innerHeight = $state(0);
  let width = $state(0);
  let height = $state(0);
  let callout = $state<HTMLElement>();

  const at = $derived(calloutAt(holes, { width, height }, { width: innerWidth, height: innerHeight }));

  /** What shows of a control: a list longer than its page shows as far as
   *  the page scrolls to, and no further than the window. */
  function shown(control: HTMLElement) {
    const box = control.getBoundingClientRect();
    let [left, top, right, bottom] = [box.left, box.top, box.right, box.bottom];
    for (let outer = control.parentElement; outer; outer = outer.parentElement) {
      if (getComputedStyle(outer).overflow === "visible") continue;
      const clip = outer.getBoundingClientRect();
      [left, top] = [Math.max(left, clip.left), Math.max(top, clip.top)];
      [right, bottom] = [Math.min(right, clip.right), Math.min(bottom, clip.bottom)];
    }
    [right, bottom] = [Math.min(right, innerWidth), Math.min(bottom, innerHeight)];
    return { left, top, width: Math.max(0, right - left), height: Math.max(0, bottom - top) };
  }

  /** The lit controls where they are now: they move as the window and the
   *  app do, and appear once what the step opened shows. */
  function measure() {
    const found: Hole[] = [];
    for (const name of tour.place?.targets ?? []) {
      const control = document.querySelector<HTMLElement>(`[data-tour="${name}"]`);
      const box = control && shown(control);
      if (!control || !box?.width || !box.height) continue;
      const rounded = parseFloat(getComputedStyle(control).borderTopLeftRadius) || 0;
      const hole = {
        name,
        left: box.left - PAD,
        top: box.top - PAD,
        width: box.width + 2 * PAD,
        height: box.height + 2 * PAD,
      };
      found.push({ ...hole, radius: Math.min(rounded + PAD, Math.min(hole.width, hole.height) / 2) });
    }
    const same =
      found.length === holes.length &&
      found.every((h, i) =>
        (["name", "left", "top", "width", "height", "radius"] as const).every((k) => h[k] === holes[i][k])
      );
    if (!same) holes = found;
  }

  function onKey(event: KeyboardEvent) {
    if (event.metaKey || event.ctrlKey) return;
    const buttons = [...(callout?.querySelectorAll("button") ?? [])];
    // A button of the callout takes Space; nothing else reaches the app
    if (event.key === " " && buttons.includes(event.target as HTMLButtonElement)) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    if (event.key === "Escape") tour.skip();
    else if (event.key === "Enter" && !event.isComposing) tour.next();
    else if (event.key === "Tab" && buttons.length) {
      const index = buttons.indexOf(document.activeElement as HTMLButtonElement);
      const step = event.shiftKey ? -1 : 1;
      buttons[(index + step + buttons.length) % buttons.length].focus();
    }
  }

  onMount(() => {
    let frame = requestAnimationFrame(function loop() {
      measure();
      frame = requestAnimationFrame(loop);
    });
    window.addEventListener("keydown", onKey, true);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("keydown", onKey, true);
    };
  });

  function focus(node: HTMLElement) {
    node.focus();
  }
</script>

<svelte:window bind:innerWidth bind:innerHeight />

<div
  class="tour"
  role="dialog"
  aria-modal="true"
  aria-label="Tour"
  in:fade={{ duration: 300, delay }}
  out:fade={{ duration: 200 }}
>
  <svg class="pointer-events-none absolute inset-0" width={innerWidth} height={innerHeight} aria-hidden="true">
    <defs>
      <filter id="tour-soft" x="-50%" y="-50%" width="200%" height="200%">
        <feGaussianBlur stdDeviation="3" />
      </filter>
      <mask id="tour-holes" maskUnits="userSpaceOnUse" x="0" y="0" width={innerWidth} height={innerHeight}>
        <rect width={innerWidth} height={innerHeight} fill="white" />
        {#each holes as hole (hole.name)}
          <rect
            data-hole={hole.name}
            x={hole.left}
            y={hole.top}
            width={hole.width}
            height={hole.height}
            rx={hole.radius}
            fill="black"
            filter="url(#tour-soft)"
            transition:fade={{ duration: 200 }}
          />
        {/each}
      </mask>
    </defs>
    <rect class="dim" width={innerWidth} height={innerHeight} mask="url(#tour-holes)" />
  </svg>

  <!-- A click on a control that opens what comes next goes on -->
  {#if tour.place?.click}
    {#each holes as hole (hole.name)}
      <button
        class="catch"
        style="left: {hole.left}px; top: {hole.top}px; width: {hole.width}px; height: {hole.height}px; border-radius: {hole.radius}px"
        onclick={() => tour.next()}
        tabindex="-1"
        aria-hidden="true"
      ></button>
    {/each}
  {/if}

  {#key tour.step?.id + "/" + tour.place?.targets.join()}
    <div
      class="callout"
      class:placed={holes.length > 0}
      style="left: {at.left}px; top: {at.top}px"
      bind:this={callout}
      bind:offsetWidth={width}
      bind:offsetHeight={height}
      in:fade={{ duration: 200 }}
    >
      <p>{tour.step?.text}</p>
      <div class="flex items-center gap-1.5">
        <span class="count">{tour.number} of {TOUR.length}</span>
        <button class="skip" onclick={() => tour.skip()}>Skip tour</button>
        <button class="s-btn primary" onclick={() => tour.next()} use:focus>
          {tour.last ? "Done" : "Next"}
        </button>
      </div>
    </div>
  {/key}
</div>

<style>
  .tour {
    position: fixed;
    inset: 0;
    z-index: 50;
    -webkit-app-region: no-drag;
  }
  .dim {
    fill: rgb(6 7 9 / 0.62);
  }
  .catch {
    position: absolute;
    cursor: pointer;
  }
  .callout {
    position: absolute;
    display: flex;
    width: 290px;
    flex-direction: column;
    gap: 0.75rem;
    padding: 0.875rem 1rem 0.75rem;
    border-radius: 12px;
    border: 1px solid var(--color-line-strong);
    background: var(--color-raised);
    box-shadow: 0 16px 40px rgb(0 0 0 / 0.45);
    opacity: 0;
  }
  /* Shown once the controls it stands beside are found */
  .callout.placed {
    opacity: 1;
  }
  .callout p {
    font-size: 13px;
    line-height: 1.5;
    color: var(--color-ink);
  }
  .count {
    flex: 1;
    font-size: 11.5px;
    font-variant-numeric: tabular-nums;
    color: var(--color-ink-3);
  }
  .skip {
    height: 1.9rem;
    border-radius: 0.45rem;
    padding: 0 0.5rem;
    font-size: 12px;
    color: var(--color-ink-3);
    transition: color 150ms;
  }
  .skip:hover {
    color: var(--color-ink);
  }
</style>
