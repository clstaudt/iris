<script lang="ts">
  import { onMount } from "svelte";
  import Circle from "./components/Circle.svelte";
  import Icon from "./components/Icon.svelte";
  import { connectPresence, ERROR_TINT, presence } from "./lib/orb.svelte";
  import { store } from "./lib/state.svelte";

  // The window is the orb's canvas: the ring and the glow that fades out
  // before the edge. The orb answers within its ring and glow; elsewhere
  // clicks go through to the windows below.
  const SIZE = Math.min(window.innerWidth, window.innerHeight) || 112;
  const REACH = SIZE * 0.36;
  // The expand mark, shown while the pointer is on the orb: up and to the
  // right of the centre, inside the ring. Clicking it opens the conversation
  const MARK_AT = { x: SIZE / 2 + REACH * 0.45, y: SIZE / 2 - REACH * 0.45 };
  const MARK_REACH = Math.max(10, SIZE * 0.1);

  let over = $state(false);
  let overMark = $state(false);
  let pressed = $state(false);

  function onOrb(event: MouseEvent) {
    return Math.hypot(event.clientX - SIZE / 2, event.clientY - SIZE / 2) <= REACH;
  }

  function onMark(event: MouseEvent) {
    return Math.hypot(event.clientX - MARK_AT.x, event.clientY - MARK_AT.y) <= MARK_REACH;
  }

  function hover(event: MouseEvent | null) {
    const on = !!event && onOrb(event);
    overMark = on && !!event && onMark(event);
    if (on === over) return;
    over = on;
    window.iris.presenceHover(on);
  }

  function leave() {
    window.iris.presenceCommand("leave");
  }

  /** A click on the expand mark opens the conversation; elsewhere on the
   *  orb it pauses or resumes, as the orb in the conversation does. With an
   *  error showing, a click opens the conversation, where the error is. */
  function click(event: MouseEvent) {
    if (presence.error || onMark(event)) {
      leave();
      return;
    }
    if (store.state !== "starting") window.iris.presenceCommand("pause");
  }

  function pointerDown(event: PointerEvent) {
    if (event.button !== 0 || !onOrb(event)) return;
    pressed = true;
    (event.currentTarget as HTMLElement).setPointerCapture(event.pointerId);
    window.iris.presenceDrag("start");
  }

  function pointerMove(event: PointerEvent) {
    if (pressed) window.iris.presenceDrag("move");
    else hover(event);
  }

  async function pointerUp(event: PointerEvent) {
    if (!pressed) return;
    pressed = false;
    const moved = await window.iris.presenceDrag("end");
    if (!moved) click(event);
    hover(event);
  }

  onMount(() =>
    connectPresence(() => {
      // Hidden: the next showing starts with clicks going through
      over = false;
      overMark = false;
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
    if (!pressed) hover(null);
  }}
>
  <Circle size={SIZE} tint={presence.error ? ERROR_TINT : null} />
  <span
    class="mark"
    class:shown={over && !pressed && !presence.error}
    class:near={overMark}
    style="left: {MARK_AT.x}px; top: {MARK_AT.y}px"
  >
    <Icon name="expand" size={12} stroke={1.75} />
  </span>
</div>

<style>
  .orb {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
    width: 100vw;
    height: 100vh;
    clip-path: circle(50%);
    outline: none;
  }

  /* The way back to the conversation: quiet until the pointer is on the
     orb, clearer when it is on the mark itself */
  .mark {
    position: absolute;
    display: grid;
    place-items: center;
    width: 20px;
    height: 20px;
    margin: -10px 0 0 -10px;
    border-radius: 50%;
    color: rgba(255, 255, 255, 0.92);
    background: rgba(20, 22, 26, 0.35);
    opacity: 0;
    transition: opacity 160ms ease, background 160ms ease;
    pointer-events: none;
  }
  .mark.shown {
    opacity: 0.55;
  }
  .mark.shown.near {
    opacity: 1;
    background: rgba(20, 22, 26, 0.55);
  }
</style>
