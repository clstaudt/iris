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
  // chosen, and the voice made of it, for English and the languages chosen
  // with it (each language's cloning model downloaded first): the voice is
  // cloned once it is made for all of them. The voice made is one of the
  // engine's voices from then on, listed with the others; this offers to
  // make one. In the voice library (keep), the voices made are kept, and a
  // cloned voice cloned again is made again in place, keeping its name and
  // its personas; for a persona, a voice made and not saved with it goes
  // when the editor closes, and cloning again makes another.

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

  type Step = "closed" | "checking" | "connect" | "download" | "sample" | "recording" | "making";
  /** Pocket TTS's languages, which a cloned voice speaks */
  const pocket = $derived(
    (store.voices?.languages ?? []).filter((l) =>
      (store.voices?.engines.kyutai ?? []).some((v) => spokenBy(v).includes(l.code))
    )
  );
  /** The languages to make the voice for: English and Iris's languages,
   *  which its personas will speak */
  let picked = $state<string[]>([]);
  /** The jobs making the voice for its other languages, once made for English */
  let making = $state<VoiceJob[]>([]);
  let madeVoice = "";
  type Access = { state: "login" | "terms" | "ready"; cached: boolean; error?: string };

  let step = $state<Step>("closed");
  let access = $state<Access["state"]>("login");
  let error = $state("");
  let token = $state("");
  let busy = $state(false);
  let progress = $state({ done: 0, total: 0 });
  let level = $state(0);
  let seconds = $state(0);
  let picker = $state<HTMLInputElement | null>(null);
  /** Voices made in this editor: the ones not kept go when it closes */
  const made: string[] = [];

  const cloned = $derived(draft.voice.startsWith(PREFIX));
  const TITLES: Record<Step, string> = {
    closed: "",
    checking: "Clone a voice",
    connect: "Connect to Hugging Face",
    download: "Downloading the cloning model",
    sample: "Clone a voice",
    recording: "Recording",
    making: "Making the voice",
  };
  /** The level on a decibel scale, as the sound check shows it */
  const lit = $derived.by(() => {
    const rms = level * 0.15;
    return rms <= 0 ? 0 : Math.max(0, Math.min(1, (20 * Math.log10(rms) + 60) / 50));
  });
  const clock = $derived(`${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`);
  const mb = (bytes: number) => Math.round(bytes / 1e6);

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
    // A voice made here and not saved with a persona goes; the core keeps
    // one a persona speaks with
    for (const voice of made) store.forgetVoice(voice).catch(() => {});
  });

  // Cloning is Kyutai's: another engine closes it
  $effect(() => {
    if (draft.engine !== "kyutai" && step !== "closed") close();
  });

  async function open() {
    error = "";
    making = [];
    const wanted = new Set(["en", ...store.languages]);
    picked = pocket.map((l) => l.code).filter((code) => wanted.has(code));
    step = "checking";
    follow(await rpc<Access>("voice.clone_access"));
  }

  /** On from the access there is: connect, download, or the sample. */
  function follow(result: Access) {
    access = result.state;
    error = result.error ?? "";
    if (result.state !== "ready") step = "connect";
    else if (!result.cached) download();
    else step = "sample";
  }

  async function check() {
    busy = true;
    try {
      const result = token.trim()
        ? await rpc<Access>("voice.clone_connect", { token: token.trim() })
        : await rpc<Access>("voice.clone_access");
      token = "";
      follow(result);
    } finally {
      busy = false;
    }
  }

  async function download() {
    step = "download";
    error = "";
    progress = { done: 0, total: 0 };
    const result = await rpc<{ ok: boolean; error?: string; access?: boolean }>("voice.clone_fetch");
    if (result.ok) step = "sample";
    else if (result.access) await open();
    else error = result.error ?? "The cloning model could not be downloaded";
  }

  async function record() {
    error = "";
    level = 0;
    seconds = 0;
    const result = await rpc<{ ok: boolean; error?: string }>("voice.record", { on: true });
    if (result.ok) step = "recording";
    else error = result.error ?? "The microphone could not be opened";
  }

  async function redo() {
    await rpc("voice.record", { on: false, keep: false });
    await record();
  }

  async function stop() {
    if (step !== "recording") return;
    step = "making";
    const result = await rpc<{ ok: boolean; error?: string }>("voice.record", { on: false });
    if (result.ok) await make();
    else back(result.error);
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
    step = "making";
    const data = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result).split(",", 2)[1] ?? "");
      reader.onerror = () => reject(reader.error);
      reader.readAsDataURL(file);
    }).catch(() => "");
    const result = await rpc<{ ok: boolean; error?: string }>("voice.sample", { data, name: file.name });
    if (result.ok) await make();
    else back(result.error);
  }

  /** Make the voice of the sample, for English and each language picked,
   *  then choose it, and say the preview in it. */
  async function make() {
    const replace = keep && cloned ? draft.voice : "";
    const result = await rpc<{
      ok: boolean;
      error?: string;
      access?: boolean;
      voice?: VoiceInfo;
      jobs?: VoiceJob[];
    }>("voice.clone", { name: draft.name.trim(), replace, languages: picked });
    if (!result.ok || !result.voice) {
      if (result.access) {
        await open();
        error = result.error ?? "";
      } else back(result.error);
      return;
    }
    const voice = result.voice.id;
    if (!keep) made.push(voice);
    madeVoice = voice;
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
    // Cloned again before saving: the voice made before is not kept
    const replaced = !keep && made.includes(draft.voice) && draft.voice !== voice ? draft.voice : null;
    await store.loadVoices();
    onchoose(voice);
    step = "closed";
    if (replaced) store.forgetVoice(replaced).catch(() => {});
    const named = store.voices?.engines.kyutai?.find((v) => v.id === voice)?.name ?? "";
    await store.previewVoice({ ...$state.snapshot(draft), name: draft.name.trim() || named, voice });
  }

  function back(message?: string) {
    error = message ?? "The voice could not be made";
    step = "sample";
  }

  function close() {
    if (step === "recording") rpc("voice.record", { on: false, keep: false }).catch(() => {});
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
      {:else if step === "checking" || step === "making"}
        <div class="waiting" aria-hidden="true"></div>
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
        {#if pocket.length > 1}
          <div class="flex flex-col gap-1.5">
            <span class="text-[11.5px] text-ink-3">Make it for</span>
            <LanguageFilter
              languages={pocket}
              asked={picked}
              addable={new Set(pocket.map((l) => l.code))}
              onwant={(code) => {
                if (code !== "en") picked = toggled(picked, code);
              }}
              label="Languages to make the voice for"
            />
          </div>
        {/if}
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
