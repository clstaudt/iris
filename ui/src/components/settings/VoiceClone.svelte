<script lang="ts">
  import { onDestroy, onMount } from "svelte";
  import { onEvent, rpc } from "../../lib/bridge";
  import { store, type PersonaSettings, type VoiceInfo, type VoiceJob } from "../../lib/state.svelte";
  import { spokenBy, toggled } from "../../lib/voices";
  import Icon from "../Icon.svelte";
  import JobLine from "../JobLine.svelte";
  import Progress from "../Progress.svelte";
  import LanguageFilter from "./LanguageFilter.svelte";

  // Cloning a voice for a persona, with Kyutai's Pocket TTS: Hugging Face
  // access to its cloning model, the model's download, a sample recorded or
  // chosen, which is prepared (trimmed and levelled) and heard, then the
  // languages to make the voice for, confirmed: only then is the voice made,
  // for English and the languages picked (each language's cloning model
  // downloaded first, shown as it goes). Back or Cancel before that leave no
  // voice behind. The voice made is one of the engine's voices from then on,
  // listed with the others; this offers to make one. Every voice made stays,
  // saved with a persona or not, until it is deleted on the Voices page.
  // There (keep), a cloned voice cloned again is made again in place,
  // keeping its name and its personas; for a persona, cloning again makes
  // another voice.

  let {
    draft,
    onchoose,
    keep = false,
  }: { draft: PersonaSettings; onchoose: (voice: string) => void; keep?: boolean } = $props();

  const MODEL_PAGE = "https://huggingface.co/kyutai/pocket-tts";
  const PREFIX = "clone:";
  /** Read aloud for a recorded sample: about 25 s, said calmly */
  const SCRIPT =
    "Have you ever noticed how different a familiar street looks early in the morning? " +
    "The shops are still closed, the air is cool, and the only sounds are a few birds " +
    "and a bicycle going by. I like to walk down to the river then, sit on the bench by " +
    "the old bridge, and watch the light move across the water until the town slowly wakes up.";
  /** A recording stops by itself after this long, in seconds */
  const LONGEST = 45;
  const BARS = 24;
  const AUDIO = ".wav,.mp3,.m4a,.aac,.flac,.ogg,.aiff,audio/*";

  type Step =
    | "closed"
    | "checking"
    | "connect"
    | "download"
    | "sample"
    | "recording"
    | "preparing"
    | "confirm"
    | "making";
  /** Pocket TTS's languages, which a cloned voice speaks */
  const pocket = $derived(
    (store.voices?.languages ?? []).filter((l) =>
      (store.voices?.engines.kyutai ?? []).some((v) => spokenBy(v).includes(l.code))
    )
  );
  /** The languages to make the voice for: English, which it is made with
   *  first, and those picked, Iris's own unless others are */
  let picked = $state<string[]>([]);
  /** The languages whose cloning model is downloaded; the others download
   *  it first, once the clone is confirmed */
  let downloaded = $state<string[]>([]);
  const downloads = $derived(
    Object.fromEntries(
      pocket
        .filter((l) => l.code !== "en" && !downloaded.includes(l.code))
        .map((l) => [l.code, `Downloads Kyutai's cloning model for ${l.name} first`])
    )
  );
  /** The sample prepared, to hear and confirm: nothing is made of it before */
  type Sample = { id: string; seconds: number; name: string };
  let sample = $state<Sample | null>(null);
  /** The jobs making the voice for its other languages, once made for English */
  let making = $state<VoiceJob[]>([]);
  let madeVoice = "";
  type Access = {
    state: "login" | "terms" | "ready";
    cached: boolean;
    error?: string;
    downloaded?: string[];
  };
  type Prepared = { ok: boolean; error?: string; sample?: string; seconds?: number };

  let step = $state<Step>("closed");
  let access = $state<Access["state"]>("login");
  let error = $state("");
  let token = $state("");
  let busy = $state(false);
  let progress = $state({ done: 0, total: 0 });
  /** What making the voice is doing, while no job shows it */
  let doing = $state("");
  let level = $state(0);
  let seconds = $state(0);
  let picker = $state<HTMLInputElement | null>(null);
  /** Samples asked to be prepared so far: only the latest is taken */
  let preparing = 0;

  const cloned = $derived(draft.voice.startsWith(PREFIX));
  const TITLES: Record<Step, string> = {
    closed: "",
    checking: "Clone a voice",
    connect: "Connect to Hugging Face",
    download: "Downloading the cloning model",
    sample: "Clone a voice",
    recording: "Recording",
    preparing: "Clone a voice",
    confirm: "Clone a voice",
    making: "Making the voice",
  };
  /** The level on a decibel scale, as the sound check shows it */
  const lit = $derived.by(() => {
    const rms = level * 0.15;
    return rms <= 0 ? 0 : Math.max(0, Math.min(1, (20 * Math.log10(rms) + 60) / 50));
  });
  const minutes = (s: number) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
  const clock = $derived(minutes(seconds));
  const mb = (bytes: number) => Math.round(bytes / 1e6);
  const playing = $derived(!!sample && store.previewing === `sample:${sample.id}`);

  let unsubscribe = () => {};
  onMount(() => {
    unsubscribe = onEvent((type, data) => {
      if (type === "clone.progress") {
        progress = { done: data.done as number, total: data.total as number };
      } else if (type === "clone.level" && step === "recording") {
        level = data.level as number;
        seconds = data.seconds as number;
        if (seconds >= LONGEST) stop();
      }
    });
  });

  onDestroy(() => {
    unsubscribe();
    if (step === "recording") rpc("voice.record", { on: false, keep: false }).catch(() => {});
    if (step !== "making") forget();
  });

  // Cloning is Kyutai's: another engine closes it
  $effect(() => {
    if (draft.engine !== "kyutai" && step !== "closed") close();
  });

  /** English, Iris's languages, and for a voice cloned again in place the
   *  ones it is made for now, which its new sample replaces */
  function defaults() {
    const voice = store.voices?.engines.kyutai?.find((v) => v.id === draft.voice);
    const before = keep && cloned ? (voice?.cloned?.made ?? []) : [];
    const wanted = new Set(["en", ...store.languages, ...before]);
    return pocket.map((l) => l.code).filter((code) => wanted.has(code));
  }

  /** A request of the clone's. One that fails (the window could not send
   *  it, or the core answered with an error) says why here and gives null:
   *  the panel never waits for an answer that will not come. */
  async function request<T>(method: string, params: Record<string, unknown> = {}): Promise<T | null> {
    try {
      return await rpc<T>(method, params);
    } catch (err) {
      error = `Iris could not do it: ${err instanceof Error ? err.message : String(err)}`;
      return null;
    }
  }

  async function open() {
    error = "";
    making = [];
    sample = null;
    picked = defaults();
    step = "checking";
    await ask();
  }

  async function ask() {
    const result = await request<Access>("voice.clone_access");
    if (result) follow(result);
    else step = "connect";
  }

  /** On from the access there is: connect, download, or the sample, which
   *  is confirmed next once there is one. */
  function follow(result: Access) {
    access = result.state;
    error = result.error ?? "";
    if (result.downloaded) downloaded = result.downloaded;
    if (result.state !== "ready") step = "connect";
    else if (!result.cached) download();
    else step = sample ? "confirm" : "sample";
  }

  async function check() {
    busy = true;
    try {
      const result = token.trim()
        ? await request<Access>("voice.clone_connect", { token: token.trim() })
        : await request<Access>("voice.clone_access");
      token = "";
      if (result) follow(result);
    } finally {
      busy = false;
    }
  }

  async function download() {
    step = "download";
    error = "";
    progress = { done: 0, total: 0 };
    const result = await request<{ ok: boolean; error?: string; access?: boolean }>("voice.clone_fetch");
    if (!result) return;
    if (result.ok) {
      downloaded = [...new Set(["en", ...downloaded])];
      step = sample ? "confirm" : "sample";
    } else if (result.access) {
      step = "checking";
      await ask();
    } else error = result.error ?? "The cloning model could not be downloaded";
  }

  async function record() {
    error = "";
    level = 0;
    seconds = 0;
    const result = await request<{ ok: boolean; error?: string }>("voice.record", { on: true });
    if (!result) return;
    if (result.ok) step = "recording";
    else error = result.error ?? "The microphone could not be opened";
  }

  async function redo() {
    if ((await request("voice.record", { on: false, keep: false })) === null) {
      step = "sample";
      return;
    }
    await record();
  }

  async function stop() {
    if (step !== "recording") return;
    step = "preparing";
    doing = "Reading the sample";
    const mine = ++preparing;
    const result = await request<Prepared>("voice.record", { on: false });
    took(result ?? { ok: false, error }, "Recording", mine);
  }

  function choose() {
    error = "";
    picker?.click();
  }

  async function chosen(event: Event) {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    input.value = "";
    if (!file) return;
    step = "preparing";
    doing = "Reading the sample";
    const mine = ++preparing;
    const data = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result).split(",", 2)[1] ?? "");
      reader.onerror = () => reject(reader.error);
      reader.readAsDataURL(file);
    }).catch(() => "");
    const result = await request<Prepared>("voice.sample", { data, name: file.name });
    took(result ?? { ok: false, error }, file.name, mine);
  }

  /** The sample prepared: heard and confirmed next. One that comes once the
   *  clone is closed, or another was asked for, is dropped. */
  function took(result: Prepared, name: string, mine: number) {
    if (step !== "preparing" || mine !== preparing) {
      if (result.sample) rpc("voice.sample_drop", { sample: result.sample }).catch(() => {});
      return;
    }
    if (!result.ok || !result.sample) {
      error = result.error ?? "The sample could not be prepared";
      step = "sample";
      return;
    }
    error = "";
    sample = { id: result.sample, seconds: result.seconds ?? 0, name };
    step = "confirm";
  }

  async function listen() {
    if (!sample) return;
    if (playing) await store.stopSample();
    else error = (await store.playSample(sample.id)) ?? "";
  }

  /** Drop the sample, which is not to be cloned: nothing of it stays. */
  function forget() {
    if (playing) store.stopSample();
    if (sample) rpc("voice.sample_drop", { sample: sample.id }).catch(() => {});
    sample = null;
  }

  /** Back to the sample, to record or choose another */
  function retake() {
    forget();
    error = "";
    step = "sample";
  }

  /** Make the voice of the sample, for English and each language picked,
   *  then choose it, and say the preview in it. */
  async function confirm() {
    if (!sample) return;
    if (playing) store.stopSample();
    const id = sample.id;
    error = "";
    step = "making";
    const replace = keep && cloned ? draft.voice : "";
    doing = "Making the voice for English";
    progress = { done: 0, total: 0 };
    const result = await request<{
      ok: boolean;
      error?: string;
      access?: boolean;
      voice?: VoiceInfo;
      jobs?: VoiceJob[];
    }>("voice.clone", {
      name: draft.name.trim(),
      replace,
      // Plain, not the state's proxy, which the window's IPC cannot copy
      languages: $state.snapshot(picked),
      sample: id,
    });
    // Closed meanwhile: a voice made is listed, a sample not cloned dropped
    if (step !== "making" || sample?.id !== id) {
      if (result?.ok) await store.loadVoices().catch(() => {});
      else rpc("voice.sample_drop", { sample: id }).catch(() => {});
      return;
    }
    // The request failed, and says why: the sample is there to try again
    if (!result) {
      step = "confirm";
      return;
    }
    if (!result.ok || !result.voice) {
      if (result.access) {
        step = "checking";
        await ask();
        error = result.error ?? "";
      } else {
        error = result.error ?? "The voice could not be made";
        step = "confirm";
      }
      return;
    }
    sample = null;
    madeVoice = result.voice.id;
    making = result.jobs ?? [];
    if (!making.length) await finish();
  }

  // Made for every language picked: the voice is cloned
  $effect(() => {
    if (step !== "making" || !making.length) return;
    const now = making.map((job) => store.jobs[job.id] ?? job);
    if (now.every((job) => job.state === "ready")) finish();
  });

  async function finish() {
    const voice = madeVoice;
    if (!voice) return;
    madeVoice = "";
    making = [];
    // The voice is made: a list that does not come back keeps it no less chosen
    await store.loadVoices().catch(() => {});
    onchoose(voice);
    step = "closed";
    const named = store.voices?.engines.kyutai?.find((v) => v.id === voice)?.name ?? "";
    await store.previewVoice({ ...$state.snapshot(draft), name: draft.name.trim() || named, voice });
  }

  function close() {
    if (step === "recording") rpc("voice.record", { on: false, keep: false }).catch(() => {});
    if (step !== "making") forget();
    making = [];
    madeVoice = "";
    step = "closed";
    error = "";
    token = "";
  }
</script>

{#if draft.engine === "kyutai"}
  {#if step === "closed"}
    <div>
      <button class="s-btn" onclick={open}>
        <Icon name={cloned ? "mic" : "plus"} size={13} />
        {cloned ? "Clone again" : "Clone a voice"}
      </button>
    </div>
  {:else}
    <div class="clone" data-step={step}>
      <div class="flex items-center justify-between gap-2">
        <h4 class="text-[12.5px] font-medium text-ink">{TITLES[step]}</h4>
        <button class="s-icon-btn" aria-label="Close" onclick={close}>
          <Icon name="close" size={13} />
        </button>
      </div>

      {#if step === "making" && making.length}
        {#each making as job (job.id)}
          <JobLine job={store.jobs[job.id] ?? job} quiet />
        {/each}
        {#if making.some((job) => (store.jobs[job.id] ?? job).state === "failed")}
          <div>
            <button class="s-btn" onclick={finish}>Done</button>
          </div>
        {/if}
      {:else if step === "making" && progress.total}
        <!-- The cloning model for English, downloaded first -->
        <Progress value={progress.done / progress.total} />
        <span class="text-[11.5px] tabular-nums text-ink-3">
          Downloading Kyutai's cloning model for English: {mb(progress.done)} of {mb(progress.total)} MB
        </span>
      {:else if step === "checking" || step === "making" || step === "preparing"}
        <div class="waiting" aria-hidden="true"></div>
        {#if (step === "making" || step === "preparing") && doing}
          <span class="text-[11.5px] text-ink-3">{doing}</span>
        {/if}
      {:else if step === "connect"}
        <p class="line">
          {access === "login"
            ? "Accept Kyutai's terms on the model page, then paste a Hugging Face access token."
            : "Accept Kyutai's terms on the model page, then check again."}
        </p>
        <div class="flex flex-wrap items-center gap-2">
          <button class="s-btn" onclick={() => window.open(MODEL_PAGE, "_blank")}>
            Open the model page
          </button>
          {#if access === "login"}
            <input
              type="password"
              class="s-field token"
              placeholder="Access token"
              aria-label="Hugging Face access token"
              autocomplete="off"
              spellcheck="false"
              bind:value={token}
              onkeydown={(e) => e.key === "Enter" && token.trim() && check()}
            />
          {/if}
          <button
            class="s-btn primary"
            disabled={busy || (access === "login" && !token.trim())}
            onclick={check}
          >
            {busy ? "Checking" : "Check"}
          </button>
        </div>
      {:else if step === "download"}
        <Progress value={progress.total ? progress.done / progress.total : 0} />
        <div class="flex items-center justify-between gap-2">
          <span class="text-[11.5px] tabular-nums text-ink-3">
            {progress.total ? `${mb(progress.done)} of ${mb(progress.total)} MB` : ""}
          </span>
          {#if error}
            <button class="s-btn" onclick={download}>Retry</button>
          {/if}
        </div>
      {:else if step === "sample"}
        <div class="flex flex-wrap items-center gap-2">
          <button class="s-btn primary" onclick={record}>
            <Icon name="mic" size={13} />
            Record
          </button>
          <button class="s-btn" onclick={choose}>Choose a file</button>
          <input
            bind:this={picker}
            type="file"
            accept={AUDIO}
            class="hidden"
            aria-label="Audio file"
            onchange={chosen}
          />
        </div>
        <p class="line">Only clone a voice you have permission to use.</p>
      {:else if step === "recording"}
        <p class="script">{SCRIPT}</p>
        <div class="flex items-center gap-3">
          <div class="s-meter" aria-hidden="true">
            {#each Array(BARS) as _, i}
              <i class:lit={i < Math.round(lit * BARS)}></i>
            {/each}
          </div>
          <span class="text-[12px] tabular-nums text-ink-3">{clock}</span>
          <span class="flex-1"></span>
          <button class="s-btn" onclick={redo}>Redo</button>
          <button class="s-btn primary" onclick={stop}>
            <Icon name="stop" size={12} />
            Stop
          </button>
        </div>
      {:else if step === "confirm" && sample}
        <div class="sample" data-sample={sample.id}>
          <button
            class="s-play {playing ? 'playing' : ''}"
            onclick={listen}
            aria-label={playing ? "Stop the sample" : "Play the sample"}
          >
            <Icon name={playing ? "stop" : "play"} size={10} />
          </button>
          <span class="name truncate" title={sample.name}>{sample.name}</span>
          <span class="length tabular-nums">{minutes(sample.seconds)}</span>
        </div>
        {#if pocket.length > 1}
          <div class="flex flex-col gap-1.5">
            <span class="text-[11.5px] text-ink-3">Make it for</span>
            <LanguageFilter
              languages={pocket}
              asked={picked}
              addable={new Set(pocket.map((l) => l.code))}
              fixed={["en"]}
              {downloads}
              onwant={(code) => {
                if (code !== "en") picked = toggled(picked, code);
              }}
              label="Languages to make the voice for"
            />
          </div>
        {/if}
        <div class="flex items-center gap-2">
          <button class="s-btn" onclick={retake}>
            <Icon name="back" size={12} />
            Back
          </button>
          <span class="flex-1"></span>
          <button class="s-btn" onclick={close}>Cancel</button>
          <button class="s-btn primary" onclick={confirm}>Clone voice</button>
        </div>
      {/if}

      {#if error}
        <p class="s-status error" title={error}>{error}</p>
      {/if}
    </div>
  {/if}
{/if}

<style>
  .clone {
    display: flex;
    flex-direction: column;
    gap: 0.7rem;
    padding: 0.75rem 0.85rem 0.85rem;
    border-radius: 0.6rem;
    border: 1px solid var(--color-line);
    background: var(--color-raised);
  }
  .line {
    font-size: 12px;
    line-height: 1.5;
    color: var(--color-ink-3);
  }
  .token {
    width: auto;
    min-width: 12rem;
    flex: 1;
    height: 1.9rem;
  }
  /* The sample prepared, heard before a voice is made of it */
  .sample {
    display: flex;
    min-width: 0;
    align-items: center;
    gap: 0.55rem;
  }
  .sample .name {
    min-width: 0;
    font-size: 12.5px;
    color: var(--color-ink);
  }
  .sample .length {
    flex-shrink: 0;
    font-size: 12px;
    color: var(--color-ink-3);
  }
  .script {
    padding: 0.6rem 0.8rem;
    border-left: 2px solid color-mix(in srgb, var(--color-accent) 45%, transparent);
    font-size: 13.5px;
    line-height: 1.65;
    color: var(--color-ink);
  }
  /* Something is under way whose length is not known */
  .waiting {
    height: 2px;
    border-radius: 9999px;
    background: linear-gradient(
      90deg,
      transparent,
      var(--color-accent),
      var(--color-petal-cyan),
      transparent
    );
    background-size: 40% 100%;
    background-repeat: no-repeat;
    animation: waiting 1.6s ease-in-out infinite;
  }
  @keyframes waiting {
    from {
      background-position: -40% 0;
    }
    to {
      background-position: 140% 0;
    }
  }
</style>
