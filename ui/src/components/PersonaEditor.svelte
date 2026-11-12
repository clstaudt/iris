<script lang="ts">
  import { onMount } from "svelte";
  import {
    store,
    type PersonaSettings,
    type SpeechEngine,
    type VoiceInfo,
  } from "../lib/state.svelte";
  import Icon from "./Icon.svelte";

  // The engines the core has, by name; the shipped ones until it has said
  const ENGINES: { id: SpeechEngine; label: string }[] = [
    { id: "kyutai", label: "Kyutai" },
    { id: "kokoro", label: "Kokoro" },
  ];
  const DEFAULT_VOICE: Record<SpeechEngine, string> = { kyutai: "alba", kokoro: "af_heart" };
  const NEW_PERSONA: PersonaSettings = {
    name: "",
    engine: "kyutai",
    voice: "alba",
    speed: 1.0,
    audio_filter: "",
    description: "",
  };

  let draft = $state<PersonaSettings | null>(null);
  let saved = "";
  let error = $state("");
  let saving = $state(false);
  let confirmDelete = $state(false);

  /** What saving would write, to tell whether anything changed. */
  function editable(p: PersonaSettings) {
    const { name, engine, voice, speed, audio_filter, description } = p;
    return JSON.stringify({ name, engine, voice, speed, audio_filter, description });
  }

  onMount(async () => {
    store.loadVoices().catch((err) => console.error("[persona] voices failed:", err));
    const id = store.editingPersona;
    const loaded = id ? await store.loadPersona(id) : { ...NEW_PERSONA };
    if (!loaded) {
      store.panel = "settings";
      return;
    }
    draft = loaded;
    saved = editable(loaded);
  });

  const isNew = $derived(!draft?.id);
  const changed = $derived(draft !== null && editable(draft) !== saved);
  const canSave = $derived(
    !!draft && changed && !!draft.name.trim() && !!draft.voice && !saving
  );
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
  // The voice's engine, speed and effect, for those who asked for more
  const detailed = $derived(store.settingsLevel !== "essentials");

  /** Voices by language, in the order the engine lists them. */
  const groups = $derived.by(() => {
    const byLanguage = new Map<string, VoiceInfo[]>();
    for (const voice of voices) {
      byLanguage.set(voice.language, [...(byLanguage.get(voice.language) ?? []), voice]);
    }
    return [...byLanguage];
  });
  const unlisted = $derived(
    !!draft && voices.length > 0 && !voices.some((v) => v.id === draft?.voice)
  );

  function setEngine(engine: SpeechEngine) {
    if (!draft || draft.engine === engine) return;
    draft.engine = engine;
    const list = store.voices?.engines[engine] ?? [];
    const preferred = DEFAULT_VOICE[engine];
    draft.voice = preferred && list.some((v) => v.id === preferred)
      ? preferred
      : (list[0]?.id ?? preferred ?? "");
  }

  function filterLabel(name: string) {
    return name.charAt(0).toUpperCase() + name.slice(1);
  }

  function preview() {
    if (!draft) return;
    store.previewVoice($state.snapshot(draft)).catch((err) =>
      console.error("[persona] preview failed:", err)
    );
  }

  async function save() {
    if (!draft || !canSave) return;
    saving = true;
    error = "";
    try {
      const failed = await store.savePersona($state.snapshot(draft));
      if (failed) error = failed;
      else store.panel = "settings";
    } catch (err) {
      error = String(err);
    } finally {
      saving = false;
    }
  }

  async function remove() {
    if (!draft?.id) return;
    if (!confirmDelete) {
      confirmDelete = true;
      return;
    }
    await store.deletePersona(draft.id);
    store.panel = "settings";
  }
</script>

<aside
  class="flex w-[360px] shrink-0 flex-col overflow-hidden rounded-[14px] border border-line bg-panel"
>
  <div class="flex h-12 shrink-0 items-center justify-between pl-2.5 pr-2.5">
    <div class="flex min-w-0 items-center gap-1">
      <button
        onclick={() => (store.panel = "settings")}
        class="flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-ink-3 transition-colors hover:bg-hover hover:text-ink"
        title="Back to settings"
      >
        <Icon name="back" size={15} />
      </button>
      <h2 class="truncate text-[13px] font-semibold text-ink">
        {isNew ? "New persona" : (store.personaNames[draft?.id ?? ""] ?? draft?.name ?? "")}
      </h2>
    </div>
    <button
      onclick={() => (store.panel = null)}
      class="flex h-7 w-7 items-center justify-center rounded-md text-ink-3 transition-colors hover:bg-hover hover:text-ink"
      title="Close"
    >
      <Icon name="close" size={15} />
    </button>
  </div>

  {#if draft}
    <div class="flex min-h-0 flex-1 flex-col gap-6 overflow-y-auto px-5 pb-4 pt-2">
      <section class="flex flex-col gap-2">
        <h3 class="section">Name</h3>
        <input bind:value={draft.name} spellcheck="false" class="field !pr-3" />
      </section>

      <section class="flex flex-col gap-2">
        <h3 class="section">Voice</h3>
        {#if detailed}
          <div class="segmented" role="radiogroup" aria-label="Speech engine">
            {#each engines as engine}
              <button
                role="radio"
                aria-checked={draft.engine === engine.id}
                disabled={voiceless(engine.id)}
                onclick={() => setEngine(engine.id)}
                class:active={draft.engine === engine.id}
              >
                {engine.label}
              </button>
            {/each}
          </div>
        {/if}
        <div class="flex gap-1.5">
          <div class="relative min-w-0 flex-1">
            <select bind:value={draft.voice} class="field">
              {#if unlisted || voices.length === 0}
                <option value={draft.voice}>{draft.voice}</option>
              {/if}
              {#each groups as [language, list]}
                <optgroup label={language}>
                  {#each list as voice}
                    <option value={voice.id}>{voice.name}</option>
                  {/each}
                </optgroup>
              {/each}
            </select>
            <Icon name="chevron" size={14} class="chevron" />
          </div>
          <button onclick={preview} class="square" title="Play">
            <Icon name="play" size={13} />
          </button>
        </div>
        {#if detailed}
          <div class="grid grid-cols-[5.5rem_1fr] items-center gap-x-3 gap-y-2 pt-1">
            {#if takesSpeed}
              <label for="persona-speed" class="text-[12.5px] text-ink-2">Speed</label>
              <div class="flex items-center gap-3">
                <input
                  id="persona-speed"
                  type="range"
                  min="0.7"
                  max="1.5"
                  step="0.05"
                  bind:value={draft.speed}
                  class="range min-w-0 flex-1"
                />
                <span class="w-10 shrink-0 text-right text-[12px] tabular-nums text-ink-2">
                  {Number(draft.speed).toFixed(2)}×
                </span>
              </div>
            {/if}
            <label for="persona-effect" class="text-[12.5px] text-ink-2">Effect</label>
            <div class="relative">
              <select id="persona-effect" bind:value={draft.audio_filter} class="field">
                <option value="">None</option>
                {#each store.voices?.filters ?? [] as name}
                  <option value={name}>{filterLabel(name)}</option>
                {/each}
              </select>
              <Icon name="chevron" size={14} class="chevron" />
            </div>
          </div>
        {/if}
      </section>

      <section class="flex min-h-[220px] flex-1 flex-col gap-2">
        <h3 class="section">Personality</h3>
        <textarea
          bind:value={draft.description}
          spellcheck="false"
          class="field personality min-h-0 flex-1"
        ></textarea>
      </section>
    </div>

    <div class="flex shrink-0 items-center justify-between gap-2 border-t border-line px-5 py-3">
      <div class="min-w-0">
        {#if error}
          <p class="truncate text-[11.5px] text-danger" title={error}>{error}</p>
        {:else if draft.id && (!draft.shipped || draft.edited)}
          <button
            onclick={remove}
            onblur={() => (confirmDelete = false)}
            class="rounded-md px-2 py-1 text-[12px] text-danger/80 transition-colors hover:bg-danger/10 hover:text-danger"
          >
            {#if draft.shipped}
              {confirmDelete ? "Confirm reset" : "Reset"}
            {:else}
              {confirmDelete ? "Confirm delete" : "Delete"}
            {/if}
          </button>
        {/if}
      </div>
      <div class="flex shrink-0 gap-1.5">
        <button onclick={() => (store.panel = "settings")} class="small">Cancel</button>
        <button onclick={save} disabled={!canSave} class="small primary">
          {saving ? "Saving" : "Save"}
        </button>
      </div>
    </div>
  {/if}
</aside>

<style>
  .section {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--color-ink-3);
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
    text-overflow: ellipsis;
    transition: border-color 150ms;
  }
  .field:hover,
  .field:focus {
    border-color: var(--color-line-strong);
  }
  .personality {
    height: auto;
    padding: 0.6rem 0.75rem;
    line-height: 1.55;
    resize: none;
  }
  .relative :global(.chevron) {
    position: absolute;
    right: 0.7rem;
    top: 50%;
    transform: translateY(-50%);
    pointer-events: none;
    color: var(--color-ink-3);
  }
  .segmented {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2px;
    padding: 2px;
    border-radius: 0.5rem;
    border: 1px solid var(--color-line);
    background: var(--color-raised);
  }
  .segmented button {
    height: 1.85rem;
    border-radius: 0.38rem;
    font-size: 12.5px;
    color: var(--color-ink-3);
    transition:
      background-color 150ms,
      color 150ms;
  }
  .segmented button:hover:not(:disabled) {
    color: var(--color-ink);
  }
  .segmented button.active {
    background: var(--color-hover);
    color: var(--color-ink);
  }
  .segmented button:disabled {
    opacity: 0.4;
  }
  .square {
    display: flex;
    height: 2.25rem;
    width: 2.25rem;
    flex-shrink: 0;
    align-items: center;
    justify-content: center;
    border-radius: 0.5rem;
    border: 1px solid var(--color-line);
    color: var(--color-ink-2);
    transition: background-color 150ms;
  }
  .square:hover {
    background: var(--color-hover);
    color: var(--color-ink);
  }
  .range {
    accent-color: var(--color-accent);
    height: 1rem;
  }
  .small {
    height: 1.9rem;
    border-radius: 0.45rem;
    border: 1px solid var(--color-line);
    padding: 0 0.75rem;
    font-size: 12px;
    color: var(--color-ink-2);
    transition: background-color 150ms;
  }
  .small:hover:not(:disabled) {
    background: var(--color-hover);
    color: var(--color-ink);
  }
  .small.primary {
    border-color: transparent;
    background: color-mix(in srgb, var(--color-accent) 22%, transparent);
    color: var(--color-ink);
  }
  .small.primary:hover:not(:disabled) {
    background: color-mix(in srgb, var(--color-accent) 32%, transparent);
  }
  .small:disabled {
    opacity: 0.4;
  }
</style>
