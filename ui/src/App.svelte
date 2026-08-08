<script lang="ts">
  import { onMount } from "svelte";
  import Controls from "./components/Controls.svelte";
  import StatusZone from "./components/StatusZone.svelte";
  import Transcript from "./components/Transcript.svelte";
  import { store } from "./lib/state.svelte";

  onMount(() => {
    store.init();
  });
</script>

<main class="flex h-screen flex-col bg-[#050505] text-neutral-200">
  <!-- drag region for the hidden-inset title bar; doubles as config display -->
  <div class="drag-region flex h-8 shrink-0 items-center justify-center">
    {#if store.config}
      <span class="select-none font-mono text-[11px] text-neutral-600">
        ◈ {store.config.persona}
        <span class="text-neutral-700"> · </span>{store.config.llm_model}
        <span class="text-neutral-700"> · </span>whisper {store.config.whisper_model} ({store.config.whisper_device})
        <span class="text-neutral-700"> · </span>{store.config.tts_voice}
        <span class="text-neutral-700"> · </span>🎙 {store.config.input_device}
      </span>
    {/if}
  </div>
  <Transcript />
  <StatusZone />
  <Controls />
</main>

<style>
  .drag-region {
    -webkit-app-region: drag;
  }
</style>
