<script lang="ts">
  import { onMount } from "svelte";
  import { STATE_COLORS, store, type AssistantState } from "../lib/state.svelte";

  interface Props {
    /** Canvas edge in CSS pixels; the ring and its glow scale with it. */
    size?: number;
  }

  let { size = 280 }: Props = $props();


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

  // Startup bloom: while the models load an iris unfolds where the ring will
  // be, then its petals flatten out and fade into the ring.
  const BLOOM_OPEN = 3.2; // seconds for the bud to open
  const BLOOM_MERGE = 1.6; // seconds for the flower to become the ring
  const TAU = Math.PI * 2;
  // The title image's petals: violet to blue, lit edges, a warm beard
  const PETAL_BASE = hexToRgb("#8d84ee");
  const PETAL_TIP = hexToRgb("#79a3ee");
  const PETAL_EDGE = hexToRgb("#b2dcf2");
  const PETAL_BEARD = hexToRgb("#e6ab7a");

  const easeOut = (t: number) => 1 - (1 - t) ** 3;
  const easeInOut = (t: number) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);
  const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
  const rgba = (c: [number, number, number], a: number) =>
    `rgba(${c[0]}, ${c[1]}, ${c[2]}, ${a})`;

  onMount(() => {
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // The canvas is deliberately much wider than the ring: the halo has to
    // reach full transparency before the edge, or the fade ends in a square.
    const SIZE = size;
    const R = SIZE * 0.25;
    const HALO_REACH = SIZE / 2;
    // Strokes and blur were tuned at 280 px; thin rings keep a legible minimum
    const K = SIZE / 280;

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
    let blooming = store.state === "starting";
    let unfold = 0; // 0 a closed bud, 1 fully open
    let merge = 0; // 0 the flower, 1 the ring

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

    /** One petal along -y from the centre: narrow at the base, broad and
     *  rounded at the tip, translucent with a lit edge and fine veins. */
    function petal(angle: number, length: number, width: number, alpha: number, beard: boolean) {
      if (length < 0.5) return;
      const c = ctx!;
      c.save();
      c.translate(cx, cy);
      c.rotate(angle);
      c.beginPath();
      c.moveTo(0, 0);
      c.bezierCurveTo(width * 0.9, -length * 0.22, width * 1.15, -length * 0.82, 0, -length);
      c.bezierCurveTo(-width * 1.15, -length * 0.82, -width * 0.9, -length * 0.22, 0, 0);
      c.closePath();

      const fill = c.createLinearGradient(0, 0, 0, -length);
      fill.addColorStop(0, rgba(PETAL_BASE, 0.34 * alpha));
      fill.addColorStop(1, rgba(PETAL_TIP, 0.12 * alpha));
      c.fillStyle = fill;
      c.fill();

      c.shadowColor = rgba(PETAL_EDGE, 0.5 * alpha);
      c.shadowBlur = 6;
      c.strokeStyle = rgba(PETAL_EDGE, 0.55 * alpha);
      c.lineWidth = Math.max(0.7, 1.4 * K);
      c.stroke();
      c.shadowBlur = 0;

      // Veins fan out from the base
      c.strokeStyle = rgba(PETAL_EDGE, 0.16 * alpha);
      c.lineWidth = 0.5;
      for (const v of [-0.55, -0.25, 0, 0.25, 0.55]) {
        c.beginPath();
        c.moveTo(0, -length * 0.06);
        c.quadraticCurveTo(width * v * 1.2, -length * 0.5, width * v * 0.75, -length * 0.88);
        c.stroke();
      }

      if (beard) {
        c.fillStyle = rgba(PETAL_BEARD, 0.4 * alpha);
        c.beginPath();
        c.ellipse(0, -length * 0.26, width * 0.16, length * 0.14, 0, 0, TAU);
        c.fill();
      }
      c.restore();
    }

    /** The iris: three broad falls opening flat, three standards between
     *  them staying more upright. Seen from above, a petal tilting open looks
     *  longer, so its drawn length follows the sine of its tilt. */
    function drawBloom(time: number, u: number, m: number) {
      const fade = 1 - m;
      if (fade <= 0) return;
      const c = ctx!;
      // Fully open while merging; a slow breath while the models still load
      const open = lerp(u * (0.95 + 0.05 * Math.sin(time * 0.8)), 1, m);
      const sway = Math.sin(time * 0.6) * 0.03 * fade;
      const twist = (1 - open) * 0.6 + m * 0.45; // petals turn as they open

      c.save();
      c.globalCompositeOperation = "lighter"; // overlaps glow, like the photo

      const glow = c.createRadialGradient(cx, cy, 0, cx, cy, SIZE / 2);
      glow.addColorStop(0, rgba(PETAL_BASE, 0.18 * fade * (0.4 + 0.6 * open)));
      glow.addColorStop(1, rgba(PETAL_BASE, 0));
      c.fillStyle = glow;
      c.fillRect(0, 0, SIZE, SIZE);

      for (let i = 0; i < 3; i++) {
        const tilt = (0.1 + 0.4 * open) * Math.PI;
        const length = lerp(R * 1.6, R, m) * Math.sin(tilt);
        petal((i / 3) * TAU + twist + sway, length, R * (0.3 + 0.28 * open), fade, true);
      }
      for (let i = 0; i < 3; i++) {
        const tilt = (0.06 + 0.26 * open + 0.18 * m) * Math.PI;
        const length = lerp(R * 1.15, R * 0.92, m) * Math.sin(tilt);
        petal((i / 3 + 1 / 6) * TAU + twist * 0.7 - sway, length, R * (0.24 + 0.2 * open), fade, false);
      }

      c.fillStyle = rgba([255, 255, 255], 0.4 * fade);
      c.beginPath();
      c.arc(cx, cy, Math.max(1.2, R * 0.07), 0, TAU);
      c.fill();
      c.restore();
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

      // The bloom restarts whenever the assistant starts (again)
      if (state === "starting" && !blooming) {
        blooming = true;
        unfold = 0;
        merge = 0;
      }
      if (blooming) {
        if (state === "starting") {
          unfold = Math.min(1, unfold + dt / BLOOM_OPEN);
        } else {
          // Barely begun, as when a reload finds the assistant running
          if (unfold < 0.15) merge = 1;
          merge = Math.min(1, merge + dt / BLOOM_MERGE);
          if (merge >= 1) blooming = false;
        }
      }
      // How much of the ring shows: none while the flower stands
      const ringShown = blooming ? easeInOut(merge) : 1;

      // Mic level while listening, playback level while speaking
      const source =
        state === "speaking"
          ? store.outputLevel
          : state === "listening"
            ? store.level
            : 0;
      // Levels arrive at 15 Hz in steps; two lagged stages turn them into a
      // soft swell that follows the voice's volume. The fall is slower than
      // the rise so the orb settles between words rather than snapping.
      level += (source - level) * (1 - Math.exp(-dt / (source > level ? 0.18 : 0.45)));
      eased += (level - eased) * (1 - Math.exp(-dt / 0.12));

      // Speech RMS is small in absolute terms; sqrt lifts normal talking into
      // a clearly visible range without letting loud peaks saturate.
      const shown = Math.sqrt(Math.min(eased * 1.6, 1));

      const breathe = 0.5 + 0.5 * Math.sin(time * 0.55);
      // Kept well off zero: the ring should brighten with the voice, not blink
      const glow = 0.4 + shown * 0.6;
      const r = R * (1 + shown * 0.045); // the ring swells with the audio it meters

      ctx!.clearRect(0, 0, SIZE, SIZE);
      if (blooming) drawBloom(time, easeOut(unfold), easeInOut(merge));
      ctx!.globalAlpha = ringShown;

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
        // A closed ring must be drawn as a full circle: with the gap near zero,
        // spin + 2π - gap can round to just under a full turn, which the
        // canvas reads as an almost empty arc, and the ring vanishes for a
        // frame. That one-frame dropout was the orb's flicker.
        if (gap < 1e-3) ctx!.arc(cx, cy, r, 0, Math.PI * 2);
        else ctx!.arc(cx, cy, r, spin, spin + Math.PI * 2 - gap);
      };
      ctx!.lineCap = "round";

      // The blur radius stays fixed: browsers round it to whole pixels, so
      // animating it makes the glow jump a step at a time. Volume is expressed
      // through the shadow's opacity instead, which varies continuously.
      ctx!.shadowColor = colStr(0.3 + glow * 0.55);
      ctx!.shadowBlur = Math.max(10, 30 * K);

      // Thick luminous ring, never filled; brightness tracks the audio level
      ring();
      ctx!.strokeStyle = colStr(0.35 + glow * 0.55);
      ctx!.lineWidth = Math.max(2.5, 6 * K);
      ctx!.stroke();

      // Bright core line keeps the edge crisp against the glow
      ctx!.shadowBlur = 0;
      ctx!.strokeStyle = `rgba(255, 255, 255, ${0.4 + breathe * 0.12 + glow * 0.35})`;
      ctx!.lineWidth = Math.max(1.1, 2 * K);
      ctx!.stroke();

      if (shutter < 1) drawAperture(shutter, r);
      ctx!.globalAlpha = 1;
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
  style="width: {size}px; height: {size}px"
  aria-label="assistant status"
></canvas>
