<script lang="ts">
  import { rpc } from "../lib/bridge";
  import { store } from "../lib/state.svelte";

  let text = $state("");

  const paused = $derived(store.state === "paused");
  const starting = $derived(store.state === "starting" || store.state === "idle");

  async function togglePause() {
    if (paused) {
      await rpc("assistant.resume");
    } else {
      await rpc("assistant.pause");
    }
  }

  async function interrupt() {
    await rpc("assistant.interrupt");
  }

  async function send() {
    const value = text.trim();
    if (!value) return;
    text = "";
    await rpc("chat.send_text", { text: value });
  }

  function onKeydown(event: KeyboardEvent) {
    if (event.key === "Enter") send();
  }

  async function onPersonaChange(event: Event) {
    const name = (event.target as HTMLSelectElement).value;
    store.persona = name;
    try {
      await rpc("persona.set", { name });
    } catch (err) {
      console.error("[controls] persona.set failed:", err);
    }
  }

  async function onModelChange(event: Event) {
    const name = (event.target as HTMLSelectElement).value;
    try {
      await rpc("llm.set", { name });
    } catch (err) {
      console.error("[controls] llm.set failed:", err);
    }
  }

  const selectedMic = $derived(
    store.inputDevices.find((d) => d.name === store.config?.input_device)?.index ??
      -1
  );

  async function onMicChange(event: Event) {
    const index = Number((event.target as HTMLSelectElement).value);
    try {
      await rpc("audio.set_input", { index: index < 0 ? null : index });
    } catch (err) {
      console.error("[controls] audio.set_input failed:", err);
    }
  }
</script>

<div
  class="flex shrink-0 items-center gap-3 border-t border-neutral-800/80 px-4 py-2.5"
>
  <button
    onclick={togglePause}
    disabled={starting}
    class="flex h-8 w-8 items-center justify-center rounded-full border text-sm transition-colors {paused
      ? 'border-emerald-500/60 text-emerald-400 hover:bg-emerald-500/10'
      : 'border-neutral-700 text-neutral-400 hover:bg-neutral-800'} disabled:cursor-not-allowed disabled:opacity-30"
    title={paused ? "Resume listening" : "Pause listening"}
  >
    {paused ? "▶" : "⏸"}
  </button>

  <button
    onclick={interrupt}
    class="flex h-8 w-8 items-center justify-center rounded-full border border-neutral-700 text-sm text-neutral-400 transition-colors hover:bg-neutral-800"
    title="Interrupt (stop speaking)"
  >
    ⏭
  </button>

  <select
    value={store.persona}
    onchange={onPersonaChange}
    class="h-8 rounded-md border border-neutral-700 bg-neutral-900 px-2 font-mono text-xs text-neutral-300 outline-none"
    title="Persona"
  >
    {#each store.personas as name}
      <option value={name}>{name}</option>
    {/each}
  </select>

  <select
    value={store.config?.llm_model ?? ""}
    onchange={onModelChange}
    class="h-8 max-w-44 truncate rounded-md border border-neutral-700 bg-neutral-900 px-2 font-mono text-xs text-neutral-300 outline-none"
    title="LLM model"
  >
    {#if store.config && !store.models.some((m) => m.name === store.config?.llm_model)}
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

  <select
    value={selectedMic}
    onchange={onMicChange}
    class="h-8 max-w-44 truncate rounded-md border border-neutral-700 bg-neutral-900 px-2 font-mono text-xs text-neutral-300 outline-none"
    title="Microphone"
  >
    {#each store.inputDevices as device}
      <option value={device.index}>{device.name}</option>
    {/each}
  </select>

  <input
    bind:value={text}
    onkeydown={onKeydown}
    placeholder="type a message…"
    class="h-8 min-w-0 flex-1 rounded-md border border-neutral-800 bg-neutral-900/60 px-3 font-mono text-xs text-neutral-200 outline-none placeholder:text-neutral-600 focus:border-neutral-600"
  />
</div>
