<script lang="ts">
  import { onMount } from "svelte";
  import { fade } from "svelte/transition";
  import { drawIris } from "../lib/iris";
  import { store } from "../lib/state.svelte";
  import Icon from "./Icon.svelte";
  import Progress from "./Progress.svelte";

  const PREFERRED_MODEL = "qwen3:1.7b";
  const OPEN_PACE = 1.4; // seconds; the flower eases toward the progress made

  /** What the core offers; null until it answers */
  const info = $derived(store.setupInfo);

  // -- the flower: a closed bud that opens as setup comes along -------------

  /** Downloaded share of each speech model, from the core's progress */
  let downloaded = $state({ stt: 0, tts: 0 });
  $effect(() => {
    const p = store.setup;
    if (!p?.total) return;
    // The voice and the wake word are fetched after speech recognition
    if (p.model === "stt") downloaded.stt = p.done / p.total;
    else downloaded = { stt: 1, tts: p.model === "tts" ? p.done / p.total : downloaded.tts };
  });

  /** How far setup has come, 0 to 1: the language model, each speech model
   *  in place (or as far as it has downloaded), and pressing Start. */
  const progress = $derived.by(() => {
    const llm = connected && model ? 1 : 0;
    const stt = (ownFolders && sttPath) || info?.stt.cached ? 1 : downloaded.stt;
    const tts = (ownFolders && ttsDir) || info?.tts.found ? 1 : downloaded.tts;
    return 0.3 * llm + 0.15 * stt + 0.15 * tts + (store.installing ? 0.1 : 0);
  });

  let canvas: HTMLCanvasElement;
  onMount(() => {
    const size = 120;
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
      shown += (progress - shown) * (1 - Math.exp(-dt / OPEN_PACE));
      // The orb's bloom takes over from here once Iris starts
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

  // -- language model ------------------------------------------------------

  let server = $state("");
  let connecting = $state(false);
  let connectError = $state("");
  let connected = $state(false);
  let model = $state("");

  // Try the usual server as soon as the core answers
  let tried = false;
  $effect(() => {
    if (info && !tried) {
      tried = true;
      connect();
    }
  });

  async function connect() {
    connecting = true;
    connectError = "";
    try {
      const error = await store.setOllamaHost(server.trim());
      connected = !error;
      connectError = error ?? "";
      if (connected && !store.models.some((m) => m.name === model)) {
        const preferred = store.models.find((m) => m.name === PREFERRED_MODEL);
        model = (preferred ?? store.models.find((m) => m.tools) ?? store.models[0])?.name ?? "";
      }
    } finally {
      connecting = false;
    }
  }

  // -- speech models -------------------------------------------------------

  let ownFolders = $state(false);
  let sttPath = $state("");
  let ttsDir = $state("");
  let folderError = $state("");

  const sizeLabel = (mb: number | null | undefined) =>
    mb == null ? "" : mb >= 1000 ? `${(mb / 1000).toFixed(1)} GB` : `${mb} MB`;

  async function choose(kind: "stt" | "tts") {
    const title = kind === "stt" ? "Speech recognition model folder" : "Voice model folder";
    const folder = await window.iris.chooseFolder(title);
    if (!folder) return;
    const error = await store.setModelPaths(kind === "stt" ? { stt_path: folder } : { tts_dir: folder });
    folderError = error ?? "";
    if (error) return;
    if (kind === "stt") sttPath = folder;
    else ttsDir = folder;
  }

  function start() {
    store.finishSetup({
      ollama_host: server.trim() || undefined,
      ...(model ? { llm_model: model } : {}),
      stt_model_path: ownFolders ? sttPath : "",
      tts_models_dir: ownFolders ? ttsDir : "",
    });
  }

  const percent = $derived(
    store.setup?.total ? Math.min(100, Math.round((100 * store.setup.done) / store.setup.total)) : null
  );
</script>

<div class="flex min-h-0 flex-1 justify-center overflow-y-auto px-6 pb-10">
  <div class="flex w-full max-w-[460px] flex-col items-center gap-6 pt-6">
    <canvas bind:this={canvas} style="width: 120px; height: 120px" aria-hidden="true"></canvas>
    <h1 class="-mt-4 text-[20px] font-semibold text-ink">Set up Iris</h1>

    {#if info}
      <div
        class="flex w-full flex-col items-center gap-6 transition-opacity duration-500 {store.installing
          ? 'opacity-40'
          : ''}"
        inert={store.installing}
        in:fade={{ duration: 400 }}
      >
        <section class="card">
          <h2 class="heading">Language model</h2>
          <p class="hint">
            Conversations run on a large language model, served by Ollama or another
            OpenAI-compatible server, on this computer or another one.
          </p>
          <div class="flex gap-1.5">
            <input
              bind:value={server}
              onkeydown={(e) => e.key === "Enter" && connect()}
              placeholder={info.ollama_url}
              spellcheck="false"
              class="field min-w-0 flex-1"
            />
            <button onclick={connect} disabled={connecting} class="secondary">
              {connecting ? "Connecting" : "Connect"}
            </button>
          </div>
          {#if connectError}
            <p class="hint !text-danger">
              No server answers there. Install Ollama from ollama.com and start it, then download a
              model in a terminal, for example: <code>ollama pull {PREFERRED_MODEL}</code>
            </p>
          {:else if connected && store.models.length === 0}
            <p class="hint !text-danger">
              The server has no models yet. With Ollama, in a terminal:
              <code>ollama pull {PREFERRED_MODEL}</code>, then Connect again.
            </p>
          {:else if connected}
            <div class="relative">
              <select bind:value={model} class="field">
                {#each store.models as m}
                  <option value={m.name}>{m.name}{m.tools ? "" : " (no tools)"}</option>
                {/each}
              </select>
              <Icon name="chevron" size={14} class="chevron" />
            </div>
          {/if}
        </section>

        <section class="card">
          <h2 class="heading">Speech models</h2>
          <div class="flex flex-col gap-2">
            <button class="choice {ownFolders ? '' : 'chosen'}" onclick={() => (ownFolders = false)}>
              <span class="text-[13px] text-ink">Download them</span>
              <span class="hint">
                Speech recognition: Whisper {info.stt.model} ({info.stt.cached
                  ? "already downloaded"
                  : sizeLabel(info.stt.size_mb)})
                <br />
                Voice: Kokoro ({info.tts.found ? "already downloaded" : sizeLabel(info.tts.size_mb)})
              </span>
            </button>
            <button class="choice {ownFolders ? 'chosen' : ''}" onclick={() => (ownFolders = true)}>
              <span class="text-[13px] text-ink">Use models I already have</span>
            </button>
          </div>
          {#if ownFolders}
            <div class="flex flex-col gap-2 pt-1">
              {#each [{ kind: "stt" as const, label: "Speech recognition", path: sttPath }, { kind: "tts" as const, label: "Voice", path: ttsDir }] as row}
                <div class="flex items-center gap-2">
                  <span class="w-[8.5rem] shrink-0 text-[12.5px] text-ink-2">{row.label}</span>
                  <span class="min-w-0 flex-1 truncate text-[12px] {row.path ? 'text-ink' : 'text-ink-3'}" title={row.path}>
                    {row.path || "Download"}
                  </span>
                  <button class="secondary" onclick={() => choose(row.kind)}>Choose…</button>
                </div>
              {/each}
              {#if folderError}
                <p class="hint !text-danger">{folderError}</p>
              {/if}
            </div>
          {/if}
        </section>
      </div>

      {#if store.installing}
        <div class="flex h-10 w-[300px] shrink-0 flex-col justify-center gap-2" in:fade>
          {#if store.setup}
            <div class="flex justify-between text-[12px] text-ink-2">
              <span>{store.setup.label}</span>
              <span class="tabular-nums text-ink-3">{percent === null ? "" : `${percent}%`}</span>
            </div>
            <!-- A new bar for each download, so it never runs backwards -->
            {#key store.setup.model}
              <Progress value={store.setup.total ? store.setup.done / store.setup.total : 0} />
            {/key}
          {/if}
        </div>
      {:else}
        <button
          onclick={start}
          class="h-10 shrink-0 rounded-full bg-accent px-6 text-[13.5px] font-medium text-frame transition-opacity hover:opacity-90"
        >
          Start Iris
        </button>
      {/if}
    {/if}
  </div>
</div>

<style>
  .card {
    display: flex;
    flex-shrink: 0;
    width: 100%;
    flex-direction: column;
    gap: 0.6rem;
    border-radius: 14px;
    border: 1px solid var(--color-line);
    background: var(--color-panel);
    padding: 1.1rem 1.25rem;
  }
  .heading {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--color-ink-3);
  }
  .hint {
    font-size: 12px;
    line-height: 1.5;
    color: var(--color-ink-3);
  }
  code {
    font-family: var(--font-mono);
    font-size: 11.5px;
    color: var(--color-ink-2);
  }
  .field {
    appearance: none;
    width: 100%;
    height: 2.25rem;
    border-radius: 0.5rem;
    border: 1px solid var(--color-line);
    background: var(--color-raised);
    padding: 0 2rem 0 0.75rem;
    font-size: 12.5px;
    color: var(--color-ink);
    outline: none;
  }
  .field:focus {
    border-color: var(--color-line-strong);
  }
  .secondary {
    height: 2.25rem;
    flex-shrink: 0;
    border-radius: 0.5rem;
    border: 1px solid var(--color-line);
    padding: 0 0.8rem;
    font-size: 12.5px;
    color: var(--color-ink-2);
    transition: background-color 150ms;
  }
  .secondary:hover:not(:disabled) {
    background: var(--color-hover);
    color: var(--color-ink);
  }
  .choice {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
    border-radius: 0.6rem;
    border: 1px solid var(--color-line);
    padding: 0.65rem 0.85rem;
    text-align: left;
    transition: border-color 150ms, background-color 150ms;
  }
  .choice.chosen {
    border-color: color-mix(in srgb, var(--color-accent) 55%, transparent);
    background: color-mix(in srgb, var(--color-accent) 7%, transparent);
  }
  .relative :global(.chevron) {
    position: absolute;
    right: 0.7rem;
    top: 50%;
    transform: translateY(-50%);
    pointer-events: none;
    color: var(--color-ink-3);
  }
</style>
