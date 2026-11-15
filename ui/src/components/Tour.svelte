<script lang="ts">
  import { onMount, untrack } from "svelte";
  import { fade } from "svelte/transition";
  import { store } from "../lib/state.svelte";
  import { calloutAt, TOUR, type Box, type TourView } from "../lib/tour";
  import { tour } from "../lib/tour.svelte";

  /** How far the light reaches past a control */
  const PAD = 6;
  /** The callout's widths, the widest first: narrower where there is no
   *  room beside the lit controls at its width */
  const WIDTHS = [300, 240, 180];

  type Hole = Box & { name: string; radius: number };

  let { delay = 0 }: { delay?: number } = $props();

  let holes = $state<Hole[]>([]);
  let innerWidth = $state(0);
  let innerHeight = $state(0);
  let heights = $state(WIDTHS.map(() => 0));
  let root = $state<HTMLElement>();
  let callout = $state<HTMLElement>();

  const at = $derived(
    calloutAt(
      holes,
      WIDTHS.map((width, i) => ({ width, height: heights[i] })),
      { width: innerWidth, height: innerHeight }
    )
  );

  /** Everything but the lit controls takes no clicks: the window, less a
   *  rounded hole for each. */
  const blocked = $derived.by(() => {
    let d = `M0 0H${innerWidth}V${innerHeight}H0Z`;
    for (const { left: x, top: y, width: w, height: h, radius: r } of holes) {
      d +=
        `M${x + r} ${y}H${x + w - r}A${r} ${r} 0 0 1 ${x + w} ${y + r}V${y + h - r}` +
        `A${r} ${r} 0 0 1 ${x + w - r} ${y + h}H${x + r}A${r} ${r} 0 0 1 ${x} ${y + h - r}` +
        `V${y + r}A${r} ${r} 0 0 1 ${x + r} ${y}Z`;
    }
    return d;
  });

  const lit = () =>
    (tour.place?.targets ?? [])
      .map((name) => document.querySelector<HTMLElement>(`[data-tour="${name}"]`))
      .filter((control): control is HTMLElement => !!control);

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
    if (!tour.place) return;
    const found: Hole[] = [];
    for (const control of lit()) {
      const box = shown(control);
      if (!box.width || !box.height) continue;
      const rounded = parseFloat(getComputedStyle(control).borderTopLeftRadius) || 0;
      const hole = {
        name: control.dataset.tour!,
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

  // A click on a lit control does what it does; then the tour learns what
  // changed, and goes on if it was the step's control
  let before: TourView | null = null;
  const outside = (event: Event) => !root?.contains(event.target as Node);
  function beforeClick(event: MouseEvent) {
    before = outside(event)
      ? { settingsPage: store.settingsPage, sidebar: store.sidebar, panel: store.panel }
      : null;
  }
  function afterClick(event: MouseEvent) {
    if (!before) return;
    const name = (event.target as Element).closest<HTMLElement>("[data-tour]")?.dataset.tour ?? "";
    tour.clicked(name, before);
    before = null;
  }

  function onKey(event: KeyboardEvent) {
    if (event.metaKey || event.ctrlKey) return;
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopImmediatePropagation();
      tour.skip();
      return;
    }
    // Typing in a lit field, the message box, is the user's
    const focused = document.activeElement as HTMLElement | null;
    const field = focused?.matches("input, textarea, select, [contenteditable]");
    if (field && lit().some((control) => control.contains(focused))) return;
    const buttons = [...(callout?.querySelectorAll("button") ?? [])];
    // A button of the callout takes Space; nothing else reaches the app
    if (event.key === " " && buttons.includes(event.target as HTMLButtonElement)) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    if (event.key === "Enter" && !event.isComposing) tour.next();
    else if (event.key === "Tab" && buttons.length) {
      const index = buttons.indexOf(document.activeElement as HTMLButtonElement);
      const step = event.shiftKey ? -1 : 1;
      buttons[(index + step + buttons.length) % buttons.length].focus();
    }
  }

  // Back from presence, by its mark or any other way
  $effect(() => {
    if (tour.away && !store.presence) untrack(() => tour.cameBack());
  });

  onMount(() => {
    let frame = requestAnimationFrame(function loop() {
      measure();
      frame = requestAnimationFrame(loop);
    });
    window.addEventListener("keydown", onKey, true);
    window.addEventListener("click", beforeClick, true);
    window.addEventListener("click", afterClick);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("keydown", onKey, true);
      window.removeEventListener("click", beforeClick, true);
      window.removeEventListener("click", afterClick);
    };
  });

  function focus(node: HTMLElement) {
    node.focus();
  }
</script>

<svelte:window bind:innerWidth bind:innerHeight />

{#snippet body(live: boolean)}
  <span class="count">{tour.step?.chapter} · {tour.number} of {TOUR.length}</span>
  <p>{tour.step?.text}</p>
  <div class="flex items-center justify-end gap-1.5">
    {#if live}
      <button class="skip" onclick={() => tour.skip()}>Skip tour</button>
      <button class="s-btn primary" onclick={() => tour.next()} use:focus>
        {tour.last ? "Done" : "Next"}
      </button>
    {:else}
      <span class="skip">Skip tour</span>
      <span class="s-btn primary">{tour.last ? "Done" : "Next"}</span>
    {/if}
  </div>
{/snippet}

<div
  class="tour"
  role="dialog"
  aria-modal="true"
  aria-label="Tour"
  bind:this={root}
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
    <!-- A quiet ring marks each lit control, small ones too -->
    {#each holes as hole (hole.name)}
      <rect
        class="ring"
        x={hole.left + 1}
        y={hole.top + 1}
        width={hole.width - 2}
        height={hole.height - 2}
        rx={Math.max(0, hole.radius - 1)}
        transition:fade={{ duration: 200 }}
      />
    {/each}
  </svg>

  <div class="block" style="clip-path: path(evenodd, '{blocked}')"></div>

  {#key tour.step?.id + "/" + tour.place?.targets.join()}
    <div
      class="callout"
      class:placed={holes.length > 0}
      style="left: {at.left}px; top: {at.top}px; width: {at.width}px"
      bind:this={callout}
      in:fade={{ duration: 200 }}
    >
      {@render body(true)}
    </div>
  {/key}
</div>

<!-- The callout at each width, unseen, for the height it takes there -->
<div class="measure" aria-hidden="true" inert>
  {#each WIDTHS as width, i}
    <div class="callout" style="width: {width}px" bind:offsetHeight={heights[i]}>
      {@render body(false)}
    </div>
  {/each}
</div>

<style>
  .tour {
    position: fixed;
    inset: 0;
    z-index: 50;
    pointer-events: none;
    -webkit-app-region: no-drag;
  }
  .dim {
    fill: rgb(6 7 9 / 0.62);
  }
  .ring {
    fill: none;
    stroke: color-mix(in srgb, var(--color-accent) 40%, transparent);
    stroke-width: 1.5;
  }
  .block {
    position: absolute;
    inset: 0;
    pointer-events: auto;
  }
  .callout {
    position: absolute;
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
    padding: 0.8rem 1rem 0.75rem;
    border-radius: 12px;
    border: 1px solid var(--color-line-strong);
    background: var(--color-raised);
    box-shadow: 0 16px 40px rgb(0 0 0 / 0.45);
    opacity: 0;
    pointer-events: auto;
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
    font-size: 11px;
    font-variant-numeric: tabular-nums;
    color: var(--color-ink-3);
  }
  .skip {
    display: inline-flex;
    align-items: center;
    height: 1.9rem;
    border-radius: 0.45rem;
    padding: 0 0.5rem;
    font-size: 12px;
    color: var(--color-ink-3);
    transition: color 150ms;
  }
  button.skip:hover {
    color: var(--color-ink);
  }
  .measure {
    position: fixed;
    top: 0;
    left: -10000px;
    visibility: hidden;
    pointer-events: none;
  }
  .measure .callout {
    position: relative;
    opacity: 1;
  }
</style>
