<script lang="ts">
  import { onMount } from "svelte";
  import Composer from "./components/Composer.svelte";
  import Icon from "./components/Icon.svelte";
  import Memory from "./components/Memory.svelte";
  import Settings from "./components/Settings.svelte";
  import StatusZone from "./components/StatusZone.svelte";
  import Transcript from "./components/Transcript.svelte";
  import { store } from "./lib/state.svelte";

  onMount(() => {
    store.init();
  });
</script>

<main class="flex h-screen flex-col bg-frame text-ink">
  <!-- Drag region for the hidden-inset title bar -->
  <header class="drag-region relative flex h-11 shrink-0 items-center justify-center">
    <span class="text-[13px] font-medium tracking-[0.08em] text-ink-3">iris</span>
    <div class="no-drag absolute right-3 flex items-center gap-1">
      <button
        onclick={() => store.togglePanel("memory")}
        class="flex h-7 w-7 items-center justify-center rounded-md transition-colors hover:bg-hover hover:text-ink {store.panel ===
        'memory'
          ? 'bg-hover text-ink'
          : 'text-ink-3'}"
        title="Memory"
      >
        <Icon name="brain" size={15} />
      </button>
      <button
        onclick={() => store.togglePanel("settings")}
        class="flex h-7 w-7 items-center justify-center rounded-md transition-colors hover:bg-hover hover:text-ink {store.panel ===
        'settings'
          ? 'bg-hover text-ink'
          : 'text-ink-3'}"
        title="Settings"
      >
        <Icon name="sliders" size={15} />
      </button>
    </div>
  </header>

  <div class="flex min-h-0 flex-1 gap-2 px-2 pb-2">
    <section
      class="relative flex min-w-0 flex-1 flex-col overflow-hidden rounded-[14px] border border-line bg-panel"
    >
      <Transcript />
      <Composer />
      <StatusZone />
    </section>
    {#if store.panel === "settings"}
      <Settings />
    {:else if store.panel === "memory"}
      <Memory />
    {/if}
  </div>
</main>

<style>
  .drag-region {
    -webkit-app-region: drag;
  }
  .no-drag {
    -webkit-app-region: no-drag;
  }
</style>
