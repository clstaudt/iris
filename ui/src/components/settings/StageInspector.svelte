<script lang="ts">
  import { onMount } from "svelte";
  import {
    DEFAULT_WAKE_WINDOW,
    loadSettings,
    store,
    type TurnDetection,
  } from "../../lib/state.svelte";
  import Icon from "../Icon.svelte";
  import Switch from "../Switch.svelte";

  let { stage }: { stage: string } = $props();

  const TITLES: Record<string, string> = {
    mic: "Microphone",
    wake: "Wake word",
    turn: "Turn end",
    stt: "Recognition",
    llm: "Language model",
    persona: "Persona",
    tools: "Tools",
    memory: "Memory",
    voice: "Voice",
    output: "Output",
  };

  const config = $derived(store.config);

  onMount(() => {
    if (stage === "stt") store.loadRecognitionModels().catch(() => {});
    if (stage === "persona") store.refreshPersonas().catch(() => {});
    if (stage === "voice" && !store.voices) store.loadVoices().catch(() => {});
    if (stage === "memory") store.loadMemory().catch(() => {});
  });

  const title = (text: string) => text.charAt(0).toUpperCase() + text.slice(1);
  const fail = (what: string) => (err: unknown) => console.error(`[settings] ${what} failed:`, err);

  // Microphone ------------------------------------------------------------

  const selectedMic = $derived(
    store.inputDevices.find((d) => d.name === config?.input_device)?.index ?? -1
  );

  function onMicChange(event: Event) {
    const index = Number((event.target as HTMLSelectElement).value);
    store.setMic(index < 0 ? null : index).catch(fail("audio.set_input"));
  }

  const METER_BARS = 24;

  // Wake word: one switch, and the words of the persona in use, which are
  // saved to it. Sleeping when idle needs the wake word to wake again ------

  const IDLE_TIMES = [15, 30, 60, 120, 300];
  let wakeDraft = $state<string | null>(null);
  let sleepDraft = $state<string | null>(null);
  const wakeOn = $derived(!!config?.wake_word);
  const wakeValue = $derived(
    wakeDraft ?? (config?.persona_wake_word || config?.wake_word || config?.persona || "")
  );
  const sleepValue = $derived(sleepDraft ?? config?.sleep_word ?? "");
  const idleOn = $derived((config?.wake_window ?? DEFAULT_WAKE_WINDOW) > 0);
  // The time is kept while it is off, so switching back on restores it
  const idleTime = $derived(
    idleOn
      ? (config?.wake_window ?? DEFAULT_WAKE_WINDOW)
      : (loadSettings().wake_window ?? DEFAULT_WAKE_WINDOW)
  );
  // A time kept from elsewhere is offered too
  const idleTimes = $derived(
    IDLE_TIMES.includes(idleTime) ? IDLE_TIMES : [...IDLE_TIMES, idleTime].sort((a, b) => a - b)
  );

  /** Save the word typed to the persona, then switch as asked. */
  async function applyWakeWord(on = wakeOn) {
    if (wakeDraft !== null) {
      const word = wakeDraft;
      wakeDraft = null;
      await store.setPersonaWords({ wake_word: word }).catch(fail("persona.words"));
    }
    if (on !== wakeOn) await store.setWake(on).catch(fail("wake.set"));
  }

  function applySleepWord() {
    if (sleepDraft === null) return;
    store.setPersonaWords({ sleep_word: sleepDraft }).catch(fail("persona.words"));
    sleepDraft = null;
  }

  function onIdleTime(event: Event) {
    const window = Number((event.target as HTMLSelectElement).value);
    store.setIdleSleep(true, window).catch(fail("wake.options"));
  }

  const idleLabel = (seconds: number) => (seconds < 60 ? `${seconds} s` : `${seconds / 60} min`);

  // Turn end --------------------------------------------------------------

  const turnModes = $derived<{ id: TurnDetection; name: string; meta: string }[]>([
    { id: "smart_turn", name: "Smart Turn", meta: "Pipecat Smart Turn v3" },
    { id: "silence", name: "Silence", meta: `After ${config?.silence_duration ?? 3.5} s` },
  ]);

  // Recognition -------------------------------------------------------------

  let switching = $state<string | null>(null);
  let recognitionError = $state("");
  const whisper = $derived((config?.recognition_engine ?? "whisper") === "whisper");
  const folderName = $derived(
    store.recognitionModels?.folder.split(/[\\/]/).filter(Boolean).pop() ?? ""
  );
  const download = $derived(store.setup?.model === "stt" ? store.setup : null);

  // The switch is over once the core reports the new model in use
  $effect(() => {
    if (switching && config?.whisper_model === switching && !config.stt_model_path) {
      switching = null;
    }
  });

  function sizeLabel(mb: number | null) {
    if (!mb) return "Download";
    return mb >= 1000 ? `${(mb / 1000).toFixed(1)} GB` : `${mb} MB`;
  }

  async function chooseRecognition(model: string) {
    recognitionError = "";
    switching = model;
    const error = await store.setRecognitionModel(model);
    if (error) {
      recognitionError = error;
      switching = null;
    }
  }

  // Model folders the user already has, instead of downloads
  let folderError = $state("");

  async function chooseFolder(kind: "stt" | "tts") {
    const folder = await window.iris.chooseFolder(
      kind === "stt" ? "Speech recognition model folder" : "Voice model folder"
    );
    if (!folder) return;
    folderError =
      (await store.setModelPaths(kind === "stt" ? { stt_path: folder } : { tts_dir: folder })) ??
      "";
  }

  async function useDownloaded(kind: "stt" | "tts") {
    folderError =
      (await store.setModelPaths(kind === "stt" ? { stt_path: "" } : { tts_dir: "" })) ?? "";
  }

  // Language model: the servers kept, which of them answer, and the one in
  // use -------------------------------------------------------------------

  const SERVER_CHECK_MS = 20_000;
  let adding = $state(false);
  let newServer = $state("");
  let serverError = $state("");
  let connecting = $state<string | null>(null);
  const activeServer = $derived(config?.ollama_host ?? "");
  const serverList = $derived([...new Set(["", ...store.servers, activeServer])]);
  const modelMissing = $derived(
    !!config && store.models.length > 0 && !store.models.some((m) => m.name === config?.llm_model)
  );

  $effect(() => {
    if (stage !== "llm") return;
    store.checkServers().catch(fail("llm.check_hosts"));
    const timer = setInterval(
      () => store.checkServers().catch(fail("llm.check_hosts")),
      SERVER_CHECK_MS
    );
    return () => clearInterval(timer);
  });

  /** A server by its address without the scheme; the default one by what it is. */
  function serverName(url: string) {
    if (!url) return "This computer";
    try {
      const parsed = new URL(url);
      return parsed.host + (parsed.pathname === "/" ? "" : parsed.pathname);
    } catch {
      return url;
    }
  }

  async function connect(url: string) {
    if (url === activeServer || connecting !== null) return;
    connecting = url;
    serverError = "";
    try {
      serverError = (await store.setOllamaHost(url)) ?? "";
    } catch (err) {
      serverError = String(err);
    } finally {
      connecting = null;
      store.checkServers().catch(fail("llm.check_hosts"));
    }
  }

  function addServer() {
    if (!newServer.trim()) return;
    store.addServer(newServer);
    newServer = "";
    adding = false;
  }

  function focus(node: HTMLInputElement) {
    node.focus();
  }

  // Tools -----------------------------------------------------------------

  const tools = $derived(
    Object.entries(config?.tools ?? {}).map(([id, state]) => ({ id, ...state }))
  );
  const modelHasTools = $derived(
    store.models.find((m) => m.name === config?.llm_model)?.tools ?? true
  );
  const plugins = $derived(config?.plugins ?? []);

  // Memory: forgetting everything asks first -----------------------------

  let confirmingForget = $state(false);
  let forgetting = $state(false);

  async function forgetEverything() {
    forgetting = true;
    try {
      await store.forgetEverything();
    } finally {
      forgetting = false;
      confirmingForget = false;
    }
  }

  // Voice -----------------------------------------------------------------

  const voiceName = $derived(
    store.voices?.engines[config?.tts_engine ?? ""]?.find((v) => v.id === config?.tts_voice)
      ?.name ??
      config?.tts_voice ??
      ""
  );
  const speechEngines = $derived(config?.engines?.tts ?? []);

  let voiceError = $state("");

  async function previewVoice() {
    const persona = await store.loadPersona(store.persona);
    if (persona) voiceError = (await store.previewVoice(persona)) ?? "";
  }
</script>

<aside
  class="flex w-[320px] shrink-0 flex-col overflow-hidden rounded-[14px] border border-line bg-panel"
>
  <div class="flex h-12 shrink-0 items-center justify-between border-b border-line pl-5 pr-2.5">
    <h2 class="text-[13px] font-semibold text-ink">{TITLES[stage] ?? stage}</h2>
    <button onclick={() => (store.stage = null)} class="s-icon-btn" title="Close">
      <Icon name="close" size={15} />
    </button>
  </div>

  <div class="flex min-h-0 flex-1 flex-col gap-6 overflow-y-auto px-5 pb-6 pt-4">
    {#if stage === "mic"}
      <section class="sec">
        <label class="s-section" for="stage-mic">Input</label>
        <select id="stage-mic" class="s-field" value={selectedMic} onchange={onMicChange}>
          {#each store.inputDevices as device}
            <option value={device.index}>{device.name}</option>
          {/each}
        </select>
        <div class="s-meter" aria-hidden="true">
          {#each Array(METER_BARS) as _, i}
            <i class:lit={i / METER_BARS < Math.min(1, store.level * 6)}></i>
          {/each}
        </div>
      </section>
    {:else if stage === "wake"}
      <section class="sec">
        <div class="flex items-center gap-3">
          <input
            value={wakeValue}
            oninput={(e) => (wakeDraft = (e.target as HTMLInputElement).value)}
            onkeydown={(e) => e.key === "Enter" && applyWakeWord()}
            onblur={() => applyWakeWord()}
            spellcheck="false"
            aria-label="Wake word"
            class="s-field min-w-0 flex-1"
          />
          <Switch label="Wake word" checked={wakeOn} onchange={(on) => applyWakeWord(on)} />
        </div>
      </section>
      <section class="sec">
        <label class="s-section" for="sleep-word">Sleep word</label>
        <input
          id="sleep-word"
          value={sleepValue}
          oninput={(e) => (sleepDraft = (e.target as HTMLInputElement).value)}
          onkeydown={(e) => e.key === "Enter" && applySleepWord()}
          onblur={applySleepWord}
          spellcheck="false"
          class="s-field"
        />
      </section>
      {#if wakeOn}
        <section class="sec">
          <label class="s-section" for="stay-awake">Sleep when idle</label>
          <div class="flex items-center gap-3">
            <select
              id="stay-awake"
              class="s-field min-w-0 flex-1"
              value={idleTime}
              disabled={!idleOn}
              onchange={onIdleTime}
            >
              {#each idleTimes as seconds}
                <option value={seconds}>{idleLabel(seconds)}</option>
              {/each}
            </select>
            <Switch
              label="Sleep when idle"
              checked={idleOn}
              onchange={(on) => store.setIdleSleep(on, idleTime).catch(fail("wake.options"))}
            />
          </div>
        </section>
      {/if}
    {:else if stage === "turn"}
      <section class="sec">
        {#each turnModes as mode}
          <button
            onclick={() => store.setTurnDetection(mode.id).catch(fail("turn.set"))}
            class="s-option {(config?.turn_detection ?? 'smart_turn') === mode.id ? 'active' : ''}"
          >
            <span class="radio"></span>
            <span class="flex min-w-0 flex-col">
              <span class="name">{mode.name}</span>
              <span class="meta">{mode.meta}</span>
            </span>
          </button>
        {/each}
      </section>
    {:else if stage === "stt"}
      {#if !whisper}
        <p class="s-status">
          {config?.engines?.stt?.find((e) => e.id === config?.recognition_engine)?.label ??
            config?.recognition_engine}
        </p>
      {:else}
        <section class="sec">
          {#if store.recognitionModels?.folder}
            <div class="s-option active">
              <span class="radio"></span>
              <span class="flex min-w-0 flex-col">
                <span class="name truncate">{folderName}</span>
                <span class="meta">Model folder</span>
              </span>
            </div>
          {/if}
          {#each store.recognitionModels?.models ?? [] as model}
            <button
              onclick={() => chooseRecognition(model.id)}
              disabled={switching !== null}
              class="s-option {model.active ? 'active' : ''}"
            >
              <span class="radio"></span>
              <span class="flex min-w-0 flex-1 flex-col">
                <span class="name">Whisper {model.id}</span>
                <span class="meta">{model.cached ? "Downloaded" : sizeLabel(model.size_mb)}</span>
              </span>
              {#if switching === model.id}
                <span class="s-status tabular-nums">
                  {download && download.total
                    ? `${Math.round((download.done / download.total) * 100)} %`
                    : "Loading"}
                </span>
              {/if}
            </button>
          {/each}
          {#if recognitionError}
            <p class="s-status error">{recognitionError}</p>
          {/if}
        </section>
        <details class="s-more">
          <summary>Model folder</summary>
          <div class="mt-3 flex items-center gap-2">
            <button class="s-btn" onclick={() => chooseFolder("stt")}>Choose…</button>
            {#if config?.stt_model_path}
              <button class="s-btn" onclick={() => useDownloaded("stt")}>Use a download</button>
            {/if}
          </div>
          {#if folderError}
            <p class="s-status error mt-2">{folderError}</p>
          {/if}
        </details>
      {/if}
    {:else if stage === "llm"}
      <section class="sec">
        <h3 class="s-section">Server</h3>
        {#each serverList as url (url)}
          {@const state = store.reachable[url]}
          <div class="s-option server {url === activeServer ? 'active' : ''}">
            <button
              class="pick"
              role="radio"
              aria-checked={url === activeServer}
              disabled={connecting !== null}
              onclick={() => connect(url)}
            >
              <span class="radio"></span>
              <span class="flex min-w-0 flex-1 flex-col">
                <span class="name truncate" title={url}>{serverName(url)}</span>
                {#if connecting === url}
                  <span class="meta">Connecting</span>
                {:else if state === false}
                  <span class="meta unreachable">Unreachable</span>
                {/if}
              </span>
            </button>
            <span
              class="dot {state === true ? 'ok' : state === false ? 'down' : ''}"
              title={state === true ? "Reachable" : state === false ? "Unreachable" : "Checking"}
            ></span>
            {#if url && url !== activeServer}
              <button
                class="s-icon-btn"
                onclick={() => store.removeServer(url)}
                aria-label="Remove {serverName(url)}"
                title="Remove"
              >
                <Icon name="trash" size={13} />
              </button>
            {/if}
          </div>
        {/each}
        {#if adding}
          <div class="flex gap-1.5">
            <input
              use:focus
              bind:value={newServer}
              onkeydown={(e) => {
                if (e.key === "Enter") addServer();
                if (e.key === "Escape") adding = false;
              }}
              placeholder="http://host:11434"
              spellcheck="false"
              aria-label="Server address"
              class="s-field min-w-0 flex-1"
            />
            <button class="s-btn h-9" disabled={!newServer.trim()} onclick={addServer}>Add</button>
          </div>
        {:else}
          <button class="s-btn self-start" onclick={() => (adding = true)}>
            <Icon name="plus" size={13} />
            Add server
          </button>
        {/if}
        {#if serverError}
          <p class="s-status error">{serverError}</p>
        {/if}
      </section>
      <section class="sec">
        <h3 class="s-section">Model</h3>
        {#if modelMissing}
          <div class="s-option active">
            <span class="radio"></span>
            <span class="flex min-w-0 flex-col">
              <span class="name">{config?.llm_model}</span>
              <span class="meta">Not installed</span>
            </span>
          </div>
        {/if}
        {#each store.models as model}
          <button
            onclick={() => store.setModel(model.name).catch(fail("llm.set"))}
            class="s-option {model.name === config?.llm_model ? 'active' : ''}"
          >
            <span class="radio"></span>
            <span class="name min-w-0 flex-1 truncate">{model.name}</span>
            {#if model.vision}<span class="s-chip">vision</span>{/if}
            {#if model.tools}<span class="s-chip">tools</span>{/if}
          </button>
        {/each}
        {#if store.models.length === 0}
          <p class="s-status">Asking the server for its models</p>
        {/if}
      </section>
    {:else if stage === "persona"}
      <section class="sec">
        {#each store.personas as id}
          <button
            onclick={() => store.setPersona(id).catch(fail("persona.set"))}
            class="s-option {id === store.persona ? 'active' : ''}"
          >
            <span class="radio"></span>
            <span class="name">{store.personaNames[id] ?? id}</span>
          </button>
        {/each}
      </section>
      <button class="s-btn self-start" onclick={() => store.editPersona(store.persona)}>
        <Icon name="pencil" size={12} />
        Edit {config?.persona}
      </button>
    {:else if stage === "tools"}
      <section class="sec">
        {#each tools as tool}
          <div class="flex items-center justify-between gap-3 py-1">
            <div class="flex min-w-0 items-baseline gap-2">
              <span class="text-[12.5px] text-ink">{tool.name ?? title(tool.id)}</span>
              {#if !tool.available}<span class="s-status">Not installed</span>{/if}
            </div>
            <Switch
              label={tool.name ?? tool.id}
              checked={!!tool.enabled}
              disabled={!tool.available}
              onchange={(on) => store.setTool(tool.id, on).catch(fail("tools.set"))}
            />
          </div>
        {/each}
        {#if !modelHasTools}
          <p class="s-status">Not supported by this model</p>
        {/if}
      </section>
      {#if plugins.length}
        <section class="sec">
          <h3 class="s-section">Plugins</h3>
          {#each plugins as plugin}
            <div class="flex items-center justify-between gap-3 text-[12.5px]">
              <span class={plugin.active ? "text-ink" : "text-ink-3"}>{plugin.label}</span>
              {#if plugin.available === false}
                <span class="s-status">Not installed</span>
              {:else if plugin.source && plugin.source !== "built-in"}
                <span class="s-status">{plugin.source}</span>
              {/if}
            </div>
          {/each}
        </section>
      {/if}
    {:else if stage === "memory"}
      <section class="sec">
        <div class="flex items-center justify-between gap-3">
          <span class="text-[12.5px] text-ink">Remember conversations</span>
          <Switch
            label="Remember conversations"
            checked={!!config?.memory && !store.amnesia}
            disabled={!config?.memory}
            onchange={() => store.toggleAmnesia().catch(fail("memory.set_amnesia"))}
          />
        </div>
        {#if config?.memory && store.remembersAnything}
          {#if confirmingForget}
            <div class="confirm">
              <p>Forget everything Iris remembers? This cannot be undone.</p>
              <div class="flex justify-end gap-2">
                <button
                  class="s-btn"
                  onclick={() => (confirmingForget = false)}
                  disabled={forgetting}
                >
                  Cancel
                </button>
                <button class="s-btn solid-danger" onclick={forgetEverything} disabled={forgetting}>
                  Forget everything
                </button>
              </div>
            </div>
          {:else}
            <button class="s-btn danger forget self-start" onclick={() => (confirmingForget = true)}>
              Forget everything
            </button>
          {/if}
        {/if}
      </section>
      {#if config?.memory}
        <button class="s-btn self-start" onclick={() => store.togglePanel("memory")}>
          <Icon name="brain" size={13} />
          Open memory
        </button>
      {/if}
    {:else if stage === "voice"}
      <section class="sec">
        <div class="s-option active">
          <button
            class="s-play {store.previewing === `${config?.tts_engine}:${config?.tts_voice}` ? 'playing' : ''}"
            onclick={previewVoice}
            aria-label="Play"
          >
            <Icon name="play" size={11} />
          </button>
          <span class="flex min-w-0 flex-col">
            <span class="name">{voiceName}</span>
            <span class="meta">{config?.persona}</span>
          </span>
        </div>
        {#if voiceError}
          <p class="s-status error">{voiceError}</p>
        {/if}
        <button class="s-btn self-start" onclick={() => store.editPersona(store.persona)}>
          <Icon name="pencil" size={12} />
          Change {config?.persona}'s voice
        </button>
      </section>
      {#if speechEngines.length}
        <section class="sec">
          <h3 class="s-section">Engines</h3>
          {#each speechEngines as engine}
            <div class="flex items-center justify-between gap-3 text-[12.5px]">
              <span class={engine.active ? "text-ink" : "text-ink-2"}>{engine.label}</span>
              {#if engine.active}
                <span class="s-chip accent">In use</span>
              {:else if engine.available === false}
                <span class="s-status">Not installed</span>
              {/if}
            </div>
          {/each}
        </section>
      {/if}
      <details class="s-more">
        <summary>Voice model folder</summary>
        <div class="mt-3 flex items-center gap-2">
          <button class="s-btn" onclick={() => chooseFolder("tts")}>Choose…</button>
          {#if config?.tts_models_dir}
            <button class="s-btn" onclick={() => useDownloaded("tts")}>Use a download</button>
          {/if}
        </div>
        {#if folderError}
          <p class="s-status error mt-2">{folderError}</p>
        {/if}
      </details>
    {:else if stage === "output"}
      <section class="sec">
        <div class="flex items-center justify-between gap-3">
          <span class="text-[12.5px] text-ink">Speak replies</span>
          <Switch
            label="Speak replies"
            checked={store.speakReplies}
            onchange={() => store.toggleVoice().catch(fail("voice.set_output"))}
          />
        </div>
        {#if config?.barge_in_available}
          <div class="flex items-center justify-between gap-3">
            <span class="text-[12.5px] text-ink">Interrupt by voice</span>
            <Switch
              label="Interrupt by voice"
              checked={config?.barge_in ?? true}
              disabled={!store.speakReplies}
              onchange={(on) => store.setBargeIn(on).catch(fail("voice.set_barge_in"))}
            />
          </div>
        {/if}
      </section>
    {/if}
  </div>
</aside>

<style>
  .sec {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  /* A server: its row picks it, beside that whether it answers and a way to
     remove it */
  .server {
    gap: 0.5rem;
    padding: 0 0.5rem 0 0;
  }
  .server .pick {
    display: flex;
    min-width: 0;
    flex: 1;
    align-items: center;
    gap: 0.65rem;
    padding: 0.45rem 0 0.45rem 0.75rem;
    text-align: left;
  }
  .server .dot {
    height: 8px;
    width: 8px;
    flex-shrink: 0;
    border-radius: 9999px;
    border: 1.5px solid var(--color-ink-3);
  }
  .server .dot.ok {
    border-color: var(--color-ok);
    background: var(--color-ok);
  }
  .server .dot.down {
    border-color: var(--color-danger);
  }
  .unreachable {
    color: var(--color-danger);
  }
  /* Forgetting everything: quiet until asked, and asked first */
  .s-btn.forget {
    margin-top: 0.25rem;
    border-color: color-mix(in srgb, var(--color-danger) 35%, transparent);
    color: var(--color-danger);
  }
  .confirm {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    margin-top: 0.25rem;
    padding: 0.75rem 0.875rem;
    border-radius: 0.6rem;
    border: 1px solid color-mix(in srgb, var(--color-danger) 25%, transparent);
    background: color-mix(in srgb, var(--color-danger) 5%, transparent);
  }
  .confirm p {
    font-size: 12.5px;
    line-height: 1.5;
    color: var(--color-ink);
  }
  .s-btn.solid-danger {
    border-color: color-mix(in srgb, var(--color-danger) 45%, transparent);
    background: color-mix(in srgb, var(--color-danger) 16%, transparent);
    color: color-mix(in srgb, var(--color-danger) 55%, var(--color-ink));
  }
  .s-btn.solid-danger:hover:not(:disabled) {
    background: color-mix(in srgb, var(--color-danger) 24%, transparent);
    color: color-mix(in srgb, var(--color-danger) 55%, var(--color-ink));
  }
</style>
