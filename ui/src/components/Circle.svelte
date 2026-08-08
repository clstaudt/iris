<script lang="ts">
  import { onMount } from "svelte";
  import { STATE_COLORS, store, type AssistantState } from "../lib/state.svelte";

  // The canvas is deliberately much wider than the ring: the halo has to reach
  // full transparency before the edge, or the fade ends in a visible square.
  const SIZE = 280;
  const R = SIZE * 0.25;
  const HALO_REACH = SIZE / 2;

  let canvas: HTMLCanvasElement;

  function hexToRgb(hex: string): [number, number, number] {
    const n = parseInt(hex.slice(1), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }

  /** States rendered as a camera aperture opening inside the circle. */
  const APERTURE_STATES = new Set(["looking", "capturing_screen"]);

  /** Work with nothing to meter: the ring opens a gap and spins slowly.
   *  Listening and speaking stay closed rings and pulse with audio instead. */
  const SPINNER_STATES = new Set([
    "starting",
    "understanding",
    "thinking",
    "summarizing",
    "searching",
    "browsing",
  ]);

  const GAP = 0.42; // radians of missing arc
  const SPIN_SPEED = 0.7; // radians per second

  const BLADES = 6;
  const SHUTTER_DURATION = 1.2; // seconds for one open-and-close

  onMount(() => {
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = SIZE * dpr;
    canvas.height = SIZE * dpr;
    ctx.scale(dpr, dpr);

    const cx = SIZE / 2;
    const cy = SIZE / 2;

    let col = hexToRgb(STATE_COLORS.idle);
    let level = 0; // first smoothing stage, tracks the incoming level events
    let eased = 0; // second stage, removes the steps between them
    let shutter = 1; // progress through one open-and-close; 1 means finished
    let wasLooking = false;
    let spinner = 0; // eases the gap open, so the ring never snaps apart
    let spin = 0; // accumulated rotation of the gap

    let last = performance.now();
    let raf: number;

    const colStr = (alpha: number) =>
      `rgba(${col[0] | 0}, ${col[1] | 0}, ${col[2] | 0}, ${alpha})`;

    /** A real diaphragm: each blade covers a half-plane, and the opening is the
     *  polygon none of them reach. Blades overlap and rotate as they swing, so
     *  the hole shrinks and grows the way a camera iris does. */
    function drawAperture(progress: number, radius: number) {
      // Closed, wide, closed again — one exposure, eased at both ends
      const cycle = 0.5 - 0.5 * Math.cos(progress * Math.PI * 2);
      const apothem = radius * (0.06 + 0.6 * cycle);
      const twist = (1 - cycle) * 0.6; // blades rotate shut as they close

      ctx!.save();
      ctx!.beginPath();
      ctx!.arc(cx, cy, radius - 3, 0, Math.PI * 2);
      ctx!.clip();
      ctx!.shadowBlur = 0;

      for (let i = 0; i < BLADES; i++) {
        ctx!.save();
        ctx!.translate(cx, cy);
        ctx!.rotate((i / BLADES) * Math.PI * 2 + twist);

        // The blade body, overlapping its neighbours as it sweeps in
        ctx!.fillStyle = colStr(0.07);
        ctx!.fillRect(apothem, -radius, radius, radius * 2);

        // Its leading edge catches the light. Only the stretch bounded by the
        // neighbouring blades is exposed — that stretch is one side of the hole.
        const half = apothem * Math.tan(Math.PI / BLADES) * 1.2;
        ctx!.beginPath();
        ctx!.moveTo(apothem, -half);
        ctx!.lineTo(apothem, half);
        ctx!.strokeStyle = colStr(0.55);
        ctx!.lineWidth = 1.4;
        ctx!.stroke();

        ctx!.restore();
      }

      ctx!.restore();
    }

    function draw(t: number) {
      const now = performance.now();
      const dt = Math.min((now - last) / 1000, 0.1);
      last = now;
      const time = t / 1000;

      const state: AssistantState = store.state;
      const target = hexToRgb(STATE_COLORS[state] ?? STATE_COLORS.idle);
      for (let i = 0; i < 3; i++) col[i] += (target[i] - col[i]) * 0.05;

      // One exposure per trigger: the shutter fires when vision starts and
      // stays shut afterwards, however long the capture takes.
      const looking = APERTURE_STATES.has(state);
      if (looking && !wasLooking) shutter = 0;
      wasLooking = looking;
      if (shutter < 1) shutter = Math.min(1, shutter + dt / SHUTTER_DURATION);

      spinner += ((SPINNER_STATES.has(state) ? 1 : 0) - spinner) * Math.min(1, dt * 3);
      spin = (spin + dt * SPIN_SPEED) % (Math.PI * 2);

      // Mic level while listening, playback level while speaking
      const source =
        state === "speaking"
          ? store.outputLevel
          : state === "listening"
            ? store.level
            : 0;
      // Levels arrive at 15 Hz and syllables make them jump, so following them
      // directly reads as flicker. Two lagged stages turn the steps into a slow
      // swell; the fall is much longer than the rise so it settles rather than
      // chatters.
      level += (source - level) * (1 - Math.exp(-dt / (source > level ? 0.28 : 0.75)));
      eased += (level - eased) * (1 - Math.exp(-dt / 0.3));

      // Speech RMS is small in absolute terms; sqrt lifts normal talking into
      // a clearly visible range without letting loud peaks saturate.
      const shown = Math.sqrt(Math.min(eased * 1.6, 1));

      const breathe = 0.5 + 0.5 * Math.sin(time * 0.55);
      // Kept well off zero: the ring should brighten with the voice, not blink
      const glow = 0.4 + shown * 0.6;
      const r = R * (1 + shown * 0.06); // the ring swells with the audio it meters

      ctx!.clearRect(0, 0, SIZE, SIZE);

      // Halo carries the colour and the volume across the room
      const halo = ctx!.createRadialGradient(cx, cy, r * 0.4, cx, cy, HALO_REACH);
      halo.addColorStop(0, colStr(0.04 + glow * 0.2));
      halo.addColorStop(1, colStr(0));
      ctx!.fillStyle = halo;
      ctx!.fillRect(0, 0, SIZE, SIZE);

      // Closed at rest, a slowly rotating gapped arc while busy
      const gap = GAP * spinner;
      const ring = () => {
        ctx!.beginPath();
        ctx!.arc(cx, cy, r, spin, spin + Math.PI * 2 - gap);
      };
      ctx!.lineCap = "round";

      // The blur radius stays fixed: browsers round it to whole pixels, so
      // animating it makes the glow jump a step at a time. Volume is expressed
      // through the shadow's opacity instead, which varies continuously.
      ctx!.shadowColor = colStr(0.2 + glow * 0.75);
      ctx!.shadowBlur = 30;

      // Thick luminous ring, never filled; brightness tracks the audio level
      ring();
      ctx!.strokeStyle = colStr(0.35 + glow * 0.55);
      ctx!.lineWidth = 6;
      ctx!.stroke();

      // Bright core line keeps the edge crisp against the glow
      ctx!.shadowBlur = 0;
      ctx!.strokeStyle = `rgba(255, 255, 255, ${0.4 + breathe * 0.12 + glow * 0.35})`;
      ctx!.lineWidth = 2;
      ctx!.stroke();

      if (shutter < 1) drawAperture(shutter, r);
    }

    function frame(t: number) {
      draw(t);
      raf = requestAnimationFrame(frame);
    }
    raf = requestAnimationFrame(frame);

    return () => cancelAnimationFrame(raf);
  });
</script>

<canvas
  bind:this={canvas}
  style="width: {SIZE}px; height: {SIZE}px"
  aria-label="assistant status"
></canvas>
