<script lang="ts">
  import { onMount } from "svelte";
  import { BLOOM_STYLE, drawIris } from "../lib/iris";
  import { Keep, keepLight } from "../lib/keep";
  import { registerOrb } from "../lib/orb.svelte";
  import { Shutter } from "../lib/shutter";
  import { STATE_COLORS, store, type AssistantState } from "../lib/state.svelte";

  interface Props {
    /** Canvas edge in CSS pixels; the ring and its glow scale with it. */
    size?: number;
    /** A color the orb turns to whatever the state, such as for an error */
    tint?: string | null;
  }

  let { size = 280, tint = null }: Props = $props();

  let canvas: HTMLCanvasElement;

  function hexToRgb(hex: string): [number, number, number] {
    const n = parseInt(hex.slice(1), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }

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

  // Startup bloom: while the models load an iris unfolds where the ring will
  // be, then its petals flatten out and fade into the ring.
  const BLOOM_OPEN = 3.2; // seconds for the bud to open
  // Over the last seconds of the awake window the orb drifts toward sleep
  const FALL_ASLEEP = 10;
  const BLOOM_MERGE = 1.6; // seconds for the flower to become the ring

  const easeOut = (t: number) => 1 - (1 - t) ** 3;
  const easeInOut = (t: number) => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);
  const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

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
    // A camera's aperture opens and closes once for each look
    const shutter = new Shutter(store.shots);
    // A ring of light drawn in for each memory written
    const keep = new Keep(store.kept);
    let spinner = 0; // eases the gap open, so the ring never snaps apart
    let spin = 0; // accumulated rotation of the gap
    let blooming = store.state === "starting";
    // 0 a closed bud, 1 fully open. After the first-launch setup the flower
    // is already partly open; the bloom carries on from there.
    let unfold = 1 - Math.cbrt(1 - store.bloomFrom); // easeOut(unfold) = bloomFrom
    store.bloomFrom = 0;
    let merge = 0; // 0 the flower, 1 the ring

    let last = performance.now();
    let raf = 0;

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

    /** A memory written: a ring of light leaves the orb's ring and is
     *  drawn in toward its heart, fading as it goes, and the heart glows
     *  softly as it takes it in. While Iris works the ring of light has
     *  the ring's gap, and turns with it. */
    function drawKeep(light: ReturnType<typeof keepLight>, radius: number, gap: number) {
      ctx!.save();
      if (light.heart > 1e-3) {
        const heart = ctx!.createRadialGradient(cx, cy, 0, cx, cy, radius * 0.6);
        heart.addColorStop(0, colStr(0.28 * light.heart));
        heart.addColorStop(1, colStr(0));
        ctx!.fillStyle = heart;
        ctx!.beginPath();
        ctx!.arc(cx, cy, radius * 0.6, 0, Math.PI * 2);
        ctx!.fill();
      }
      if (light.glow > 1e-3) {
        const r = radius * light.reach;
        ctx!.beginPath();
        if (gap < 1e-3) ctx!.arc(cx, cy, r, 0, Math.PI * 2);
        else ctx!.arc(cx, cy, r, spin, spin + Math.PI * 2 - gap);
        ctx!.lineCap = "round";
        // A fixed blur, as the ring's: its opacity carries the fade
        ctx!.shadowColor = colStr(0.7 * light.glow);
        ctx!.shadowBlur = Math.max(8, 20 * K);
        ctx!.strokeStyle = colStr(0.45 * light.glow);
        // Thinner as it goes in
        ctx!.lineWidth = Math.max(1.2, 4 * K * light.reach);
        ctx!.stroke();
        ctx!.shadowBlur = 0;
        ctx!.strokeStyle = `rgba(255, 255, 255, ${0.25 * light.glow})`;
        ctx!.lineWidth = Math.max(0.7, 1.2 * K);
        ctx!.stroke();
      }
      ctx!.restore();
    }

    /** The startup bloom: the iris opening while the models load, then
     *  merging into the ring. The flower itself is drawn by lib/iris. */
    function drawBloom(time: number, u: number, m: number) {
      const fade = 1 - m;
      if (fade <= 0) return;
      drawIris(
        ctx!,
        {
          cx,
          cy,
          R,
          // Fully open while merging; a slow breath while the models still load
          open: lerp(u * (0.95 + 0.05 * Math.sin(time * 0.8)), 1, m),
          merge: m,
          sway: Math.sin(time * 0.6) * 0.03 * fade,
          fade,
          glowRadius: SIZE / 2,
        },
        { ...BLOOM_STYLE, edgeWidth: Math.max(0.7, 1.4 * K) }
      );
    }

    function draw(t: number) {
      const now = performance.now();
      const dt = Math.min((now - last) / 1000, 0.1);
      last = now;
      // Wall-clock time, so the breathing is in step in every window
      const time = (performance.timeOrigin + t) / 1000;

      const state: AssistantState = store.state;

      // Falling asleep: while it listens awake, the last seconds of the
      // window tint the orb toward its sleeping tone; a new turn, or the user
      // speaking, restarts the window and it brightens again
      let drowsy = 0;
      if (store.awakeUntil !== null && state === "listening") {
        const remaining = (store.awakeUntil - performance.now()) / 1000;
        drowsy = Math.min(1, Math.max(0, 1 - remaining / FALL_ASLEEP));
      }
      const awake = hexToRgb(STATE_COLORS[state] ?? STATE_COLORS.idle);
      const asleep = hexToRgb(STATE_COLORS.dormant);
      const target = tint
        ? hexToRgb(tint)
        : awake.map((c, i) => c + (asleep[i] - c) * drowsy);
      for (let i = 0; i < 3; i++) col[i] += (target[i] - col[i]) * 0.05;

      // One exposure per look: the shutter fires when vision starts and
      // stays shut afterwards, however long the capture takes. The canvas
      // says how often it fired, for the end-to-end test
      if (shutter.frame(store, dt)) canvas.dataset.shots = String(shutter.fired);
      // One keep per memory written, said on the canvas likewise
      if (keep.frame(store, dt)) canvas.dataset.kept = String(keep.played);
      const light = keepLight(keep.progress);

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
      const glow = (0.4 + shown * 0.6) * (1 - 0.35 * drowsy);
      // The ring swells with the audio it meters, and gives as it keeps
      const r = R * (1 + shown * 0.045) * (1 - 0.03 * light.give);

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

      if (shutter.progress < 1) drawAperture(shutter.progress, r);
      if (keep.progress < 1) drawKeep(light, r, gap);
      ctx!.globalAlpha = 1;
    }

    function frame(t: number) {
      draw(t);
      raf = requestAnimationFrame(frame);
    }

    // Only the window on screen draws; at a switch the orb's motion passes
    // to the other window, which carries on from it
    const unregister = registerOrb({
      motion: () => ({
        col: [col[0], col[1], col[2]],
        level,
        eased,
        shutter: shutter.progress,
        shots: shutter.shown,
        keep: keep.progress,
        kept: keep.shown,
        spinner,
        spin,
        blooming,
        unfold,
        merge,
      }),
      adopt: (motion) => {
        col = [motion.col[0], motion.col[1], motion.col[2]];
        level = motion.level;
        eased = motion.eased;
        shutter.progress = motion.shutter;
        shutter.shown = motion.shots;
        keep.progress = motion.keep;
        keep.shown = motion.kept;
        spinner = motion.spinner;
        spin = motion.spin;
        blooming = motion.blooming;
        unfold = motion.unfold;
        merge = motion.merge;
      },
      rest: () => {
        cancelAnimationFrame(raf);
        raf = 0;
        shutter.shown = null;
        keep.shown = null;
      },
      wake: () => {
        if (raf) return;
        last = performance.now();
        raf = requestAnimationFrame(frame);
      },
    });

    return () => {
      cancelAnimationFrame(raf);
      unregister();
    };
  });
</script>

<canvas
  bind:this={canvas}
  style="width: {size}px; height: {size}px"
  aria-label="assistant status"
></canvas>
