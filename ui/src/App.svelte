<script lang="ts">
  import { onMount, untrack } from "svelte";
  import { fade } from "svelte/transition";
  import Composer from "./components/Composer.svelte";
  import Icon from "./components/Icon.svelte";
  import Memory from "./components/Memory.svelte";
  import NewThread from "./components/NewThread.svelte";
  import SettingsWindow from "./components/settings/SettingsWindow.svelte";
  import StatusZone from "./components/StatusZone.svelte";
  import SwitchConfirm from "./components/SwitchConfirm.svelte";
  import Threads from "./components/Threads.svelte";
  import Tour from "./components/Tour.svelte";
  import Transcript from "./components/Transcript.svelte";
  import Walkthrough from "./components/walkthrough/Walkthrough.svelte";
  import { connectConversation } from "./lib/orb.svelte";
  import { store } from "./lib/state.svelte";
  import { tour } from "./lib/tour.svelte";

  onMount(() => {
    store.init();
    // Presence is a window of its own; this one hands it the orb
    return connectConversation();
  });

  // The window's own buttons sit at the top left on macOS
  const mac = window.iris?.platform === "darwin";

  /** The tour taken at the end of the walkthrough starts once the
   *  conversation has faded in, and again at a launch until it is finished
   *  or skipped; run from Settings it shows at once. */
  let tourDelay = $state(0);
  $effect(() => {
    if (store.needsSetup || tour.active) return;
    untrack(() => {
      const byItself = tour.startsByItself();
      tourDelay = byItself ? 1000 : 0;
      if (byItself) tour.start();
    });
  });
</script>

<main class="flex h-screen flex-col bg-frame text-ink">
  <!-- Drag region for the hidden-inset title bar -->
  <header class="drag-region relative flex h-11 shrink-0 items-center justify-center">
    <button
      onclick={() => store.toggleSidebar()}
      class="no-drag absolute flex h-7 w-7 items-center justify-center rounded-md transition-colors hover:bg-hover hover:text-ink {mac
        ? 'left-[84px]'
        : 'left-3'} {store.sidebar ? 'bg-hover text-ink' : 'text-ink-3'} {store.needsSetup
        ? 'hidden'
        : ''}"
      title="Threads"
      data-tour="threads"
    >
      <Icon name="panel" size={15} />
    </button>
    <span class="text-[13px] font-medium tracking-[0.08em] text-ink-3">iris</span>
    <div class="no-drag absolute right-3 flex items-center gap-1 {store.needsSetup ? 'hidden' : ''}">
      <button
        onclick={() => store.setPresence(true)}
        class="flex h-7 w-7 items-center justify-center rounded-md text-ink-3 transition-colors hover:bg-hover hover:text-ink"
        title="Presence"
      >
        <Icon name="shrink" size={15} />
      </button>
      <button
        onclick={() => store.togglePanel("memory")}
        class="flex h-7 w-7 items-center justify-center rounded-md transition-colors hover:bg-hover hover:text-ink {store.panel ===
        'memory'
          ? 'bg-hover text-ink'
          : 'text-ink-3'}"
        title="Memory"
        data-tour="memory"
      >
        <Icon name="brain" size={15} />
      </button>
      <button
        onclick={() => store.toggleSettings()}
        class="flex h-7 w-7 items-center justify-center rounded-md transition-colors hover:bg-hover hover:text-ink {store.settingsPage
          ? 'bg-hover text-ink'
          : 'text-ink-3'}"
        title="Settings"
        data-tour="settings"
      >
        <Icon name="settings" size={15} />
      </button>
    </div>
  </header>

  <!-- One cell: the walkthrough fades out, then the conversation fades in -->
  <div class="stack min-h-0 flex-1">
    {#if store.needsSetup}
      <div class="flex min-h-0" out:fade={{ duration: 400 }}>
        <Walkthrough />
      </div>
    {:else}
      <div class="flex min-h-0 gap-2 px-2 pb-2" in:fade={{ duration: 600, delay: 400 }}>
        {#if store.sidebar && !store.settingsPage}
          <Threads />
        {/if}
        <!-- Memory beside the threads, left of the conversation -->
        {#if store.panel === "memory" && !store.settingsPage}
          <Memory />
        {/if}
        <!-- The settings window covers the conversation without unmounting it -->
        <section
          class="relative flex min-w-0 flex-1 flex-col overflow-hidden rounded-[14px] border border-line bg-panel {store.settingsPage
            ? 'hidden'
            : ''}"
        >
          <!-- Who the thread is with -->
          <div class="flex h-10 shrink-0 items-center px-4">
            {#if !store.choosing && store.config}
              <span class="thread-label text-[13px] text-ink-2"
                >{store.config.persona}{#if store.config.role_name}<span class="text-ink-3"
                    >&nbsp;· {store.config.role_name}</span
                  >{/if}</span
              >
            {/if}
          </div>
          {#if store.choosing}
            <NewThread />
          {:else}
            <Transcript />
          {/if}
          <Composer />
          <StatusZone />
          <SwitchConfirm />
        </section>
        {#if store.settingsPage}
          <SettingsWindow />
        {/if}
      </div>
    {/if}
  </div>

  {#if tour.active && !store.needsSetup}
    <Tour delay={tourDelay} />
  {/if}
</main>

<style>
  .drag-region {
    -webkit-app-region: drag;
  }
  .no-drag {
    -webkit-app-region: no-drag;
  }
  .stack {
    display: grid;
    grid-template-rows: minmax(0, 1fr);
  }
  .stack > :global(*) {
    grid-area: 1 / 1;
  }
</style>
