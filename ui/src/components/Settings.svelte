<script lang="ts">
  import { onMount } from "svelte";
  import { store } from "../lib/state.svelte";
  import Icon from "./Icon.svelte";
  import Switch from "./Switch.svelte";

  const TOOLS = [
    { id: "camera", label: "Camera" },
    { id: "screen", label: "Screen" },
    { id: "web", label: "Web search" },
  ];

  // Pick up personas added since launch
  onMount(() => {
    store.refreshPersonas().catch(() => {});
  });

  const modelHasTools = $derived(
    store.models.find((m) => m.name === store.config?.llm_model)?.tools ?? true
  );

  // Wake word: the word stays in the field when switched off, so switching
  // back on restores it
  const DEFAULT_WAKE_WORD = "Iris";
  let wakeDraft = $state<string | null>(null);
  const wakeOn = $derived(!!store.config?.wake_word);
  const wakeValue = $derived(wakeDraft ?? (store.config?.wake_word || DEFAULT_WAKE_WORD));

  function applyWakeWord(on: boolean) {
    const word = wakeValue.trim() || DEFAULT_WAKE_WORD;
    store.setWakeWord(on ? word : "").catch((err) =>
      console.error("[settings] wake.set failed:", err)
    );
  }

  // Ollama server: edited as a draft, applied on Enter or Connect
  let hostDraft = $state<string | null>(null);
  let hostError = $state("");
  let connecting = $state(false);
  const hostValue = $derived(hostDraft ?? store.config?.ollama_host ?? "");

  async function connect() {
    connecting = true;
    hostError = "";
    try {
      const error = await store.setOllamaHost(hostValue.trim());
      if (error) hostError = error;
      else hostDraft = null;
    } catch (err) {
      hostError = String(err);
    } finally {
      connecting = false;
    }
  }

  function onPersonaChange(event: Event) {
    store.setPersona((event.target as HTMLSelectElement).value).catch((err) =>
      console.error("[settings] persona.set failed:", err)
    );
  }

  const selectedMic = $derived(
    store.inputDevices.find((d) => d.name === store.config?.input_device)?.index ?? -1
  );

  const modelMissing = $derived(
    !!store.config && !store.models.some((m) => m.name === store.config?.llm_model)
  );

  function onModelChange(event: Event) {
    store.setModel((event.target as HTMLSelectElement).value).catch((err) =>
      console.error("[settings] llm.set failed:", err)
    );
  }

  function onMicChange(event: Event) {
    const index = Number((event.target as HTMLSelectElement).value);
    store.setMic(index < 0 ? null : index).catch((err) =>
      console.error("[settings] audio.set_input failed:", err)
    );
  }
</script>

<aside
  class="flex w-[300px] shrink-0 flex-col overflow-y-auto rounded-[14px] border border-line bg-panel"
>
  <div class="flex h-12 shrink-0 items-center justify-between pl-5 pr-2.5">
    <h2 class="text-[13px] font-semibold text-ink">Settings</h2>
    <button
      onclick={() => (store.panel = null)}
      class="flex h-7 w-7 items-center justify-center rounded-md text-ink-3 transition-colors hover:bg-hover hover:text-ink"
      title="Close settings"
    >
      <Icon name="close" size={15} />
    </button>
  </div>

  <div class="flex flex-col gap-6 px-5 pb-6 pt-2">
    <section class="flex flex-col gap-2">
      <h3 class="section">Persona</h3>
      <div class="relative">
        <select value={store.persona} onchange={onPersonaChange} class="field">
          {#each store.personas as id}
            <option value={id}>{store.personaNames[id] ?? id}</option>
          {/each}
        </select>
        <Icon name="chevron" size={14} class="chevron" />
      </div>
    </section>

    <section class="flex flex-col gap-2">
      <h3 class="section">Ollama server</h3>
      <div class="flex gap-1.5">
        <input
          value={hostValue}
          oninput={(e) => (hostDraft = (e.target as HTMLInputElement).value)}
          onkeydown={(e) => e.key === "Enter" && connect()}
          placeholder={store.config?.ollama_url ?? "http://localhost:11434"}
          spellcheck="false"
          class="field min-w-0 flex-1 !pr-3"
        />
        <button
          onclick={connect}
          disabled={connecting || hostDraft === null}
          class="h-9 shrink-0 rounded-lg border border-line px-3 text-[12.5px] text-ink-2 transition-colors hover:bg-hover hover:text-ink disabled:opacity-40"
        >
          {connecting ? "Connecting" : "Connect"}
        </button>
      </div>
      {#if hostError}
        <p class="hint !text-danger">{hostError}</p>
      {/if}
    </section>

    <section class="flex flex-col gap-2">
      <h3 class="section">Language model</h3>
      <div class="relative">
        <select value={store.config?.llm_model ?? ""} onchange={onModelChange} class="field">
          {#if store.config && modelMissing}
            <option value={store.config.llm_model}>
              {store.config.llm_model} (not installed)
            </option>
          {/if}
          {#each store.models as model}
            <option value={model.name}>
              {model.name}{model.tools ? "" : " (no tools)"}
            </option>
          {/each}
        </select>
        <Icon name="chevron" size={14} class="chevron" />
      </div>
      {#if store.models.length === 0}
        <p class="hint">Asking Ollama for its models…</p>
      {/if}
    </section>

    <section class="flex flex-col gap-2">
      <h3 class="section">Tools</h3>
      <div class="flex flex-col">
        {#each TOOLS as tool}
          {@const state = store.config?.tools?.[tool.id]}
          <div class="flex items-center justify-between gap-3 py-1.5">
            <div class="flex min-w-0 items-baseline gap-2">
              <span class="text-[12.5px] text-ink">{tool.label}</span>
              {#if state && !state.available}
                <span class="hint">Not installed</span>
              {/if}
            </div>
            <Switch
              label={tool.label}
              checked={!!state?.enabled}
              disabled={!state?.available}
              onchange={(on) => store.setTool(tool.id, on)}
            />
          </div>
        {/each}
      </div>
      {#if !modelHasTools}
        <p class="hint">Not supported by this model</p>
      {/if}
    </section>

    <section class="flex flex-col gap-2">
      <h3 class="section">Wake word</h3>
      <div class="flex items-center gap-3">
        <input
          value={wakeValue}
          oninput={(e) => (wakeDraft = (e.target as HTMLInputElement).value)}
          onkeydown={(e) => e.key === "Enter" && wakeOn && applyWakeWord(true)}
          onblur={() => wakeOn && wakeDraft !== null && applyWakeWord(true)}
          spellcheck="false"
          class="field min-w-0 flex-1 !pr-3"
        />
        <Switch label="Wake word" checked={wakeOn} onchange={(on) => applyWakeWord(on)} />
      </div>
    </section>

    <section class="flex flex-col gap-2">
      <h3 class="section">Microphone</h3>
      <div class="relative">
        <select value={selectedMic} onchange={onMicChange} class="field">
          {#each store.inputDevices as device}
            <option value={device.index}>{device.name}</option>
          {/each}
        </select>
        <Icon name="chevron" size={14} class="chevron" />
      </div>
    </section>

    {#if store.config}
      <section class="flex flex-col gap-2">
        <h3 class="section">Speech</h3>
        <dl class="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-[12.5px]">
          <dt class="text-ink-3">Recognition</dt>
          <dd class="text-ink-2">
            Whisper {store.config.whisper_model} ·
            {store.config.whisper_engine === "mlx" ? "MLX" : store.config.whisper_device}
          </dd>
          <dt class="text-ink-3">Voice</dt>
          <dd class="text-ink-2">{store.config.tts_voice}</dd>
        </dl>
      </section>
    {/if}
  </div>
</aside>

<style>
  .section {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--color-ink-3);
  }
  .hint {
    font-size: 11.5px;
    line-height: 1.45;
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
  .relative :global(.chevron) {
    position: absolute;
    right: 0.7rem;
    top: 50%;
    transform: translateY(-50%);
    pointer-events: none;
    color: var(--color-ink-3);
  }
</style>
