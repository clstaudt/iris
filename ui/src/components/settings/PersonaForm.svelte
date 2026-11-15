<script lang="ts">
  import { onMount } from "svelte";
  import {
    store,
    type PersonaSettings,
    type SpeechEngine,
    type VoiceInfo,
  } from "../../lib/state.svelte";
  import { byLanguages, byOrigin, speakAlike, spokenBy, toggled } from "../../lib/voices";
  import Icon from "../Icon.svelte";
  import LanguageFilter from "./LanguageFilter.svelte";
  import VoiceClone from "./VoiceClone.svelte";
  import VoiceLanguages from "./VoiceLanguages.svelte";

  // A persona is who Iris is: a name, a personality, the words that wake it
  // and send it to sleep, and a voice. Models, tools, turn-taking and
  // whether the wake word is on belong to the pipeline.

  // The engines the core has, by name; the shipped ones until it has said
  const ENGINES: { id: SpeechEngine; label: string }[] = [
    { id: "kyutai", label: "Kyutai" },
    { id: "kokoro", label: "Kokoro" },
  ];
  const DEFAULT_VOICE: Record<SpeechEngine, string> = { kyutai: "anna", kokoro: "af_heart" };
  const NEW_PERSONA: PersonaSettings = {
    name: "",
    engine: "kyutai",
    voice: "anna",
    speed: 1.0,
    audio_filter: "",
    description: "",
    wake_word: "",
    sleep_word: "",
  };

  let draft = $state<PersonaSettings | null>(null);
  let saved = $state("");
  let error = $state("");
  let saving = $state(false);
  let confirmDelete = $state(false);

  /** What saving would write, to tell whether anything changed. */
  function editable(p: PersonaSettings) {
    const { name, engine, voice, speed, audio_filter, description, wake_word, sleep_word } = p;
    return JSON.stringify({
      name,
      engine,
      voice,
      speed,
      audio_filter,
      description,
      wake_word: wake_word?.trim() || name.trim(),
      sleep_word: sleep_word?.trim() ?? "",
    });
  }

  /** Show a persona as it is saved, with nothing to save. False when it is
   *  gone. */
  async function load(id: string) {
    const loaded = id ? await store.loadPersona(id) : { ...NEW_PERSONA };
    if (!loaded) return false;
    loaded.wake_word ||= loaded.name;
    loaded.sleep_word ??= "";
    draft = loaded;
    saved = editable(loaded);
    return true;
  }

  onMount(async () => {
    if (!(await load(store.editingPersona))) store.editingPersona = store.persona;
  });

  const isNew = $derived(!draft?.id);
  const changed = $derived(draft !== null && editable(draft) !== saved);
  const canSave = $derived(!!draft && changed && !!draft.name.trim() && !!draft.voice && !saving);
  const voices = $derived<VoiceInfo[]>(store.voices?.engines[draft?.engine ?? "kyutai"] ?? []);
  const engines = $derived(
    store.voices?.engine_info?.map((engine) => ({ id: engine.id, label: engine.name })) ?? ENGINES
  );
  // An engine without voices (Kokoro before it is downloaded) cannot be chosen
  const voiceless = (engine: SpeechEngine) =>
    !!store.voices && (store.voices.engines[engine]?.length ?? 0) === 0;
  const takesSpeed = $derived(
    store.voices?.engine_info?.find((engine) => engine.id === draft?.engine)?.takes_speed ??
      draft?.engine === "kokoro"
  );

  // The languages Iris talks in, and those a voice speaks properly: a reply
  // in any other is shown, not spoken
  const languages = $derived(store.voices?.languages ?? []);
  const nameOf = (code: string) => languages.find((l) => l.code === code)?.name ?? code;
  /** The languages the voices shown must speak; none shows them all. Of
   *  those, the ones a voice of this engine speaks (asked), the voices that
   *  speak them all, and the languages one of those also speaks */
  let wanted = $state<string[]>([]);
  const filtered = $derived(byLanguages(voices, wanted));
  const asked = $derived(filtered.asked);
  const shown = $derived(filtered.shown);

  function want(code: string) {
    wanted = toggled(asked, code);
  }

  /** Whether the engine's voices all speak the same languages: the
   *  persona's languages above then tell them for every voice */
  const alike = $derived(speakAlike(voices));
  /** Voices by the language they were made for, in the order the engine
   *  lists them, each with the languages it speaks */
  const groups = $derived(byOrigin(shown, alike));

  // What the persona's voice speaks of Iris's languages, is not ready for
  // yet, and does not speak
  const chosen = $derived(voices.find((v) => v.id === draft?.voice));

  function setEngine(engine: SpeechEngine) {
    if (!draft || draft.engine === engine) return;
    draft.engine = engine;
    const list = store.voices?.engines[engine] ?? [];
    const preferred = DEFAULT_VOICE[engine];
    draft.voice =
      preferred && list.some((v) => v.id === preferred) ? preferred : (list[0]?.id ?? "");
  }

  /** A wake word that is the name (the default) follows it. */
  function rename(name: string) {
    if (!draft) return;
    const wake = draft.wake_word?.trim() ?? "";
    if (!wake || wake.toLowerCase() === draft.name.trim().toLowerCase()) draft.wake_word = name;
    draft.name = name;
  }

  const filterLabel = (name: string) => name.charAt(0).toUpperCase() + name.slice(1);

  async function preview(voice: string) {
    if (!draft) return;
    error = (await store.previewVoice({ ...$state.snapshot(draft), voice })) ?? "";
  }

  async function save() {
    if (!draft || !canSave) return;
    saving = true;
    error = "";
    try {
      const failed = await store.savePersona($state.snapshot(draft));
      if (failed) {
        error = failed;
      } else {
        // A new persona is edited from here on (savePersona opens it)
        saved = editable(draft);
      }
    } catch (err) {
      error = String(err);
    } finally {
      saving = false;
    }
  }

  async function revert() {
    if (!draft) return;
    if (isNew) {
      store.editingPersona = store.persona;
      return;
    }
    await load(draft.id!);
  }

  async function exportFile() {
    if (!draft?.id) return;
    error = "";
    try {
      error = (await store.exportPersona(draft.id, draft.name)) ?? "";
    } catch (err) {
      error = String(err);
    }
  }

  async function remove() {
    if (!draft?.id) return;
    if (!confirmDelete) {
      confirmDelete = true;
      return;
    }
    const id = draft.id;
    const shipped = draft.shipped;
    await store.deletePersona(id);
    confirmDelete = false;
    // A persona that ships with Iris is shown as it ships once reset: the
    // editor stays on it, so nothing else would show it anew
    if (shipped && (await load(id))) return;
    store.editingPersona = store.persona;
  }
</script>

{#if draft}
  <div class="flex min-w-0 flex-col gap-6">
    <div class="flex items-center gap-3">
      <input
        value={draft.name}
        oninput={(e) => rename((e.target as HTMLInputElement).value)}
        placeholder="Name"
        spellcheck="false"
        aria-label="Name"
        class="title min-w-0 flex-1"
      />
      {#if draft.id && draft.id === store.persona}
        <span class="s-chip accent">Active</span>
      {:else if draft.id}
        <button
          class="s-btn"
          disabled={changed}
          onclick={() => store.newThread(draft!.id!, { choose: false })}
        >
          Talk to {draft.name}
        </button>
      {/if}
    </div>

    <section class="flex flex-col gap-2">
      <label class="s-section" for="persona-personality">Personality</label>
      <textarea
        id="persona-personality"
        bind:value={draft.description}
        spellcheck="false"
        rows="7"
        class="s-field"
      ></textarea>
    </section>

    <div class="grid grid-cols-2 gap-6">
      <section class="flex flex-col gap-2">
        <label class="s-section" for="persona-wake-word">Wake word</label>
        <input
          id="persona-wake-word"
          bind:value={draft.wake_word}
          spellcheck="false"
          class="s-field"
        />
      </section>
      <section class="flex flex-col gap-2">
        <label class="s-section" for="persona-sleep-word">Sleep word</label>
        <input
          id="persona-sleep-word"
          bind:value={draft.sleep_word}
          spellcheck="false"
          class="s-field"
        />
      </section>
    </div>

    <section class="flex flex-col gap-3">
      <div class="flex items-center justify-between gap-2">
        <h3 class="s-section">Voice</h3>
        <button class="manage" onclick={() => store.openSettings("voices")}>Manage voices</button>
      </div>
      {#if engines.length > 1}
        <div class="s-seg self-start" role="radiogroup" aria-label="Speech engine">
          {#each engines as engine}
            <button
              role="radio"
              aria-checked={draft.engine === engine.id}
              disabled={voiceless(engine.id)}
              onclick={() => setEngine(engine.id)}
              class={draft.engine === engine.id ? "active" : ""}
            >
              {engine.label}
            </button>
          {/each}
        </div>
      {/if}
      {#if chosen && languages.length}
        <VoiceLanguages
          engine={draft.engine}
          voice={chosen}
          label="Languages of {draft.name || 'the persona'}"
        />
      {/if}
      {#if languages.length}
        <LanguageFilter {languages} {asked} addable={filtered.addable} onwant={want} />
      {/if}
      {#if !shown.length && asked.length}
        <p class="text-[12px] text-ink-3">
          No voice here speaks {asked.map(nameOf).join(" and ")}
        </p>
      {/if}
      {#each groups as [language, list]}
        <div class="flex flex-col gap-1.5">
          {#if language}
            <span class="text-[11.5px] text-ink-3">{language}</span>
          {/if}
          <div class="voices">
            {#each list as voice}
              <div
                class="voice {draft.voice === voice.id ? 'active' : ''}"
                title={spokenBy(voice).map(nameOf).join(", ")}
              >
                <button
                  class="play {store.previewing === `${draft.engine}:${voice.id}` ? 'playing' : ''}"
                  onclick={() => preview(voice.id)}
                  aria-label="Play {voice.name}"
                >
                  <Icon name="play" size={10} />
                </button>
                <button
                  class="pick"
                  role="radio"
                  aria-checked={draft.voice === voice.id}
                  onclick={() => (draft!.voice = voice.id)}
                >
                  <span class="name">{voice.name}</span>
                  {#if !alike}
                    <span class="codes">{spokenBy(voice).join(" ")}</span>
                  {/if}
                </button>
              </div>
            {/each}
          </div>
        </div>
      {/each}
      <VoiceClone {draft} onchoose={(voice) => (draft!.voice = voice)} />
    </section>

    <div class="grid grid-cols-2 gap-6">
      {#if takesSpeed}
        <section class="flex flex-col gap-2">
          <label class="s-section" for="persona-speed">Speed</label>
          <div class="flex items-center gap-3">
            <input
              id="persona-speed"
              type="range"
              min="0.7"
              max="1.5"
              step="0.05"
              bind:value={draft.speed}
              class="s-range min-w-0 flex-1"
            />
            <span class="w-10 shrink-0 text-right text-[12px] tabular-nums text-ink-2">
              {Number(draft.speed).toFixed(2)}×
            </span>
          </div>
        </section>
      {/if}
      <section class="flex flex-col gap-2">
        <h3 class="s-section">Effect</h3>
        <div class="flex flex-wrap gap-1.5">
          {#each ["", ...(store.voices?.filters ?? [])] as name}
            <button
              onclick={() => (draft!.audio_filter = name)}
              class="toggle {draft.audio_filter === name ? 'active' : ''}"
            >
              {name ? filterLabel(name) : "None"}
            </button>
          {/each}
        </div>
      </section>
    </div>

    <div class="flex items-center justify-between gap-2 border-t border-line pt-4">
      <div class="flex min-w-0 gap-1.5">
        {#if error}
          <p class="s-status error truncate" title={error}>{error}</p>
        {:else if draft.id}
          {#if !draft.shipped || draft.edited}
            <button class="s-btn danger" onclick={remove} onblur={() => (confirmDelete = false)}>
              {#if draft.shipped}
                {confirmDelete ? "Confirm reset" : "Reset"}
              {:else}
                {confirmDelete ? "Confirm delete" : "Delete"}
              {/if}
            </button>
          {/if}
          <button class="s-btn" disabled={changed} onclick={exportFile}>Export</button>
        {/if}
      </div>
      <div class="flex shrink-0 gap-1.5">
        <button class="s-btn" disabled={!changed && !isNew} onclick={revert}>Cancel</button>
        <button class="s-btn primary" disabled={!canSave} onclick={save}>
          {saving ? "Saving" : "Save"}
        </button>
      </div>
    </div>
  </div>
{/if}

<style>
  .title {
    height: 2.25rem;
    margin-left: -0.5rem;
    padding: 0 0.5rem;
    border-radius: 0.5rem;
    border: 1px solid transparent;
    background: transparent;
    font-size: 17px;
    font-weight: 600;
    color: var(--color-ink);
    outline: none;
  }
  .title:hover,
  .title:focus {
    border-color: var(--color-line-strong);
    background: var(--color-raised);
  }
  /* To the Voices page, where voices are managed */
  .manage {
    font-size: 11.5px;
    color: var(--color-ink-3);
  }
  .manage:hover {
    color: var(--color-ink);
  }
  .voices {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(146px, 1fr));
    gap: 6px;
  }
  .voice {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem;
    border-radius: 0.55rem;
    border: 1px solid var(--color-line);
    background: var(--color-raised);
    transition: border-color 150ms;
  }
  .voice:hover {
    border-color: var(--color-line-strong);
  }
  .voice.active {
    border-color: color-mix(in srgb, var(--color-accent) 55%, transparent);
    background: color-mix(in srgb, var(--color-accent) 7%, var(--color-raised));
  }
  .play {
    display: flex;
    height: 1.5rem;
    width: 1.5rem;
    flex-shrink: 0;
    align-items: center;
    justify-content: center;
    border-radius: 9999px;
    background: rgb(255 255 255 / 0.05);
    color: var(--color-ink-2);
  }
  .play:hover {
    background: var(--color-hover);
    color: var(--color-ink);
  }
  /* From the click until the voice has been heard: loading it can take a while */
  .play.playing {
    background: color-mix(in srgb, var(--color-petal-warm) 22%, transparent);
    color: var(--color-petal-warm);
  }
  .pick {
    display: flex;
    min-width: 0;
    flex: 1;
    flex-direction: column;
    text-align: left;
  }
  .pick > span {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .name {
    font-size: 12.5px;
    color: var(--color-ink);
  }
  /* The languages a voice speaks, the one it was made for first */
  .codes {
    font-size: 10px;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    color: var(--color-ink-3);
  }
  .toggle {
    height: 1.75rem;
    padding: 0 0.7rem;
    border-radius: 9999px;
    border: 1px solid var(--color-line);
    font-size: 12px;
    color: var(--color-ink-2);
    transition:
      border-color 150ms,
      color 150ms;
  }
  .toggle:hover {
    border-color: var(--color-line-strong);
    color: var(--color-ink);
  }
  .toggle.active {
    border-color: color-mix(in srgb, var(--color-accent) 45%, transparent);
    background: color-mix(in srgb, var(--color-accent) 10%, transparent);
    color: var(--color-ink);
  }
</style>
