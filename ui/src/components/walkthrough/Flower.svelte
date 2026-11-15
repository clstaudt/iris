<script lang="ts">
  import { onMount } from "svelte";
  import { drawIris } from "../../lib/iris";
  import { store } from "../../lib/state.svelte";

  /** A closed bud that opens as the walkthrough comes along; the orb's
   *  bloom carries on from where it got to. */
  let { open, size = 96 }: { open: number; size?: number } = $props();

  const OPEN_PACE = 1.4; // seconds; the flower eases toward the progress made

  let canvas: HTMLCanvasElement;

  onMount(() => {
    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvas.height = size * dpr;
    const c = canvas.getContext("2d")!;
    c.scale(dpr, dpr);
    let shown = 0;
    let last = performance.now();
    let raf = 0;
    const frame = (t: number) => {
      const dt = Math.min((t - last) / 1000, 0.1);
      last = t;
      // Only ever opens: progress that runs back would close it again
      shown += Math.max(0, open - shown) * (1 - Math.exp(-dt / OPEN_PACE));
      store.bloomFrom = shown;
      const time = t / 1000;
      c.clearRect(0, 0, size, size);
      drawIris(
        c,
        {
          cx: size / 2,
          cy: size / 2 + 3,
          R: size * 0.22,
          // Breathes like the orb's bloom: not at all while closed
          open: shown * (0.95 + 0.05 * Math.sin(time * 0.8)),
          merge: 0,
          sway: Math.sin(time * 0.6) * 0.03,
          fade: 1,
          glowRadius: size / 2,
        },
        { fill: 1.4, edgeWidth: 1.3, edgeBlur: 8, veinWidth: 0.4, heart: 0.5 }
      );
      raf = requestAnimationFrame(frame);
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  });
</script>

<canvas bind:this={canvas} style="width: {size}px; height: {size}px" aria-hidden="true"></canvas>
