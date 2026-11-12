<script lang="ts">
  import { onMount } from "svelte";
  import Circle from "./components/Circle.svelte";
  import { connectPresence, ERROR_TINT, presence } from "./lib/orb.svelte";
  import { store } from "./lib/state.svelte";

  // The window is the orb's canvas: the ring and the glow that fades out
  // before the edge. The orb answers within its ring and glow; elsewhere
  // clicks go through to the windows below.
  const SIZE = Math.min(window.innerWidth, window.innerHeight) || 112;
  const REACH = SIZE * 0.36;
  // A second click this soon after the first opens the conversation
  const DOUBLE_CLICK = 400;

  let over = false;
  let pressed = false;
  let pendingClick: ReturnType<typeof setTimeout> | null = null;

  function onOrb(event: MouseEvent) {
    return Math.hypot(event.clientX - SIZE / 2, event.clientY - SIZE / 2) <= REACH;
  }

  function hover(on: boolean) {
    if (on === over) return;
    over = on;
    window.iris.presenceHover(on);
  }

  function leave() {
    if (pendingClick) clearTimeout(pendingClick);
    pendingClick = null;
    window.iris.presenceCommand("leave");
  }

  /** A click pauses or resumes, as on the orb in the conversation; a double
   *  click opens the conversation. With an error showing, a click opens the
   *  conversation, where the error is. */
  function click() {
    if (presence.error || pendingClick) {
      leave();
      return;
    }
    pendingClick = setTimeout(() => {
      pendingClick = null;
      if (store.state !== "starting") window.iris.presenceCommand("pause");
    }, DOUBLE_CLICK);
  }

  function pointerDown(event: PointerEvent) {
    if (event.button !== 0 || !onOrb(event)) return;
    pressed = true;
    (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
    window.iris.presenceDrag("start");
  }

  function pointerMove(event: PointerEvent) {
    if (pressed) window.iris.presenceDrag("move");
    else hover(onOrb(event));
  }

  async function pointerUp(event: PointerEvent) {
    if (!pressed) return;
    pressed = false;
    const moved = await window.iris.presenceDrag("end");
    if (!moved) click();
    hover(onOrb(event));
  }

  onMount(() =>
    connectPresence(() => {
      // Hidden: the next showing starts with clicks going through
      over = false;
      pressed = false;
    })
  );
</script>

<svelte:window
  onkeydown={(event) => {
    if (event.key === "Escape") leave();
  }}
/>

<div
  class="orb"
  role="button"
  tabindex="-1"
  aria-label="iris"
  title={presence.error ?? undefined}
  data-state={store.state}
  onpointerdown={pointerDown}
  onpointermove={pointerMove}
  onpointerup={pointerUp}
  onmouseleave={() => {
    if (!pressed) hover(false);
  }}
>
  <Circle size={SIZE} tint={presence.error ? ERROR_TINT : null} />
</div>

<style>
  .orb {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100vw;
    height: 100vh;
    clip-path: circle(50%);
    outline: none;
  }
</style>
