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

  const DEFAULT_SLEEP_WORD = "That's all";
  const STAY_AWAKE = [15, 30, 60, 120];
  let sleepDraft = $state<string | null>(null);
  const sleepValue = $derived(sleepDraft ?? store.config?.sleep_word ?? DEFAULT_SLEEP_WORD);

  function applySleepWord() {
    if (sleepDraft === null) return;
    store.setWakeOptions({ sleep_word: sleepValue.trim() }).catch((err) =>
      console.error("[settings] wake.options failed:", err)
    );
    sleepDraft = null;
  }

  function onStayAwake(event: Event) {
    const window = Number((event.target as HTMLSelectElement).value);
    store.setWakeOptions({ window }).catch((err) =>
      console.error("[settings] wake.options failed:", err)
    );
  }

  function applyWakeWord(on: boolean) {
    const word = wakeValue.trim() || DEFAULT_WAKE_WORD;
    store.setWakeWord(on ? word : "").catch((err) =>
      console.error("[settings] wake.set failed:", err)
    );
  }

  // Advanced: model folders the user already has, instead of downloads
  let advancedOpen = $state(false);
  let modelError = $state("");

  async function chooseModelFolder(kind: "stt" | "tts") {
    const title = kind === "stt" ? "Speech recognition model folder" : "Voice model folder";
    const folder = await window.iris.chooseFolder(title);
    if (!folder) return;
    modelError =
      (await store.setModelPaths(kind === "stt" ? { stt_path: folder } : { tts_dir: folder })) ?? "";
  }

  async function useDownloaded(kind: "stt" | "tts") {
    modelError =
      (await store.setModelPaths(kind === "stt" ? { stt_path: "" } : { tts_dir: "" })) ?? "";
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
      <div class="flex gap-1.5">
        <div class="relative min-w-0 flex-1">
          <select value={store.persona} onchange={onPersonaChange} class="field">
            {#each store.personas as id}
              <option value={id}>{store.personaNames[id] ?? id}</option>
            {/each}
          </select>
          <Icon name="chevron" size={14} class="chevron" />
        </div>
        <button onclick={() => store.editPersona(store.persona)} class="square" title="Edit persona">
          <Icon name="pencil" size={13} />
        </button>
        <button onclick={() => store.editPersona()} class="square" title="New persona">
          <Icon name="plus" size={14} />
        </button>
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
      {#if wakeOn}
        <div class="grid grid-cols-[5.5rem_1fr] items-center gap-x-3 gap-y-2 pt-1">
          <label for="sleep-word" class="text-[12.5px] text-ink-2">Sleep word</label>
          <input
            id="sleep-word"
            value={sleepValue}
            oninput={(e) => (sleepDraft = (e.target as HTMLInputElement).value)}
            onkeydown={(e) => e.key === "Enter" && applySleepWord()}
            onblur={applySleepWord}
            spellcheck="false"
            class="field !pr-3"
          />
          <label for="stay-awake" class="text-[12.5px] text-ink-2">Stay awake</label>
          <div class="relative">
            <select
              id="stay-awake"
              value={store.config?.wake_window ?? 30}
              onchange={onStayAwake}
              class="field"
            >
              {#each STAY_AWAKE as seconds}
                <option value={seconds}>{seconds < 60 ? `${seconds} s` : `${seconds / 60} min`}</option>
              {/each}
            </select>
            <Icon name="chevron" size={14} class="chevron" />
          </div>
        </div>
      {/if}
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
      {#if store.config?.barge_in_available}
        <div class="flex items-center justify-between gap-3 py-1.5">
          <span class="text-[12.5px] text-ink">Interrupt by voice</span>
          <Switch
            label="Interrupt by voice"
            checked={store.config?.barge_in ?? true}
            onchange={(on) => store.setBargeIn(on)}
          />
        </div>
      {/if}
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
          <dd class="text-ink-2">
            {store.config.tts_engine === "kokoro" ? "Kokoro" : "Kyutai"} · {store.config.tts_voice}
          </dd>
        </dl>
      </section>

      <section class="flex flex-col gap-2">
        <button
          onclick={() => (advancedOpen = !advancedOpen)}
          class="flex items-center gap-1.5 self-start text-[12.5px] text-ink-3 transition-colors hover:text-ink-2"
        >
          <Icon name="chevron" size={13} class={advancedOpen ? "" : "-rotate-90"} />
          Advanced
        </button>
        {#if advancedOpen}
          <h3 class="section pt-1">Model folders</h3>
          {#each [{ kind: "stt" as const, label: "Recognition", path: store.config.stt_model_path }, { kind: "tts" as const, label: "Voice", path: store.config.tts_models_dir }] as row}
            <div class="flex items-center gap-2">
              <span class="w-[5.5rem] shrink-0 text-[12.5px] text-ink-2">{row.label}</span>
              <span
                class="min-w-0 flex-1 truncate text-[12px] {row.path ? 'text-ink' : 'text-ink-3'}"
                title={row.path}
              >
                {row.path || "Downloaded"}
              </span>
              <button class="small" onclick={() => chooseModelFolder(row.kind)}>Choose…</button>
              {#if row.path}
                <button class="small" onclick={() => useDownloaded(row.kind)}>Download</button>
              {/if}
            </div>
          {/each}
          {#if modelError}
            <p class="hint !text-danger">{modelError}</p>
          {/if}
        {/if}
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
  .small {
    height: 1.9rem;
    flex-shrink: 0;
    border-radius: 0.45rem;
    border: 1px solid var(--color-line);
    padding: 0 0.6rem;
    font-size: 12px;
    color: var(--color-ink-2);
    transition: background-color 150ms;
  }
  .small:hover {
    background: var(--color-hover);
    color: var(--color-ink);
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
