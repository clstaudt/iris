<script lang="ts">
  import { store, type SettingsPage } from "../../lib/state.svelte";
  import Icon from "../Icon.svelte";
  import Personas from "./Personas.svelte";
  import Pipeline from "./Pipeline.svelte";
  import StageInspector from "./StageInspector.svelte";

  const PAGES: { id: SettingsPage; label: string; icon: string }[] = [
    { id: "pipeline", label: "Pipeline", icon: "flow" },
    { id: "personas", label: "Personas", icon: "users" },
  ];

  const page = $derived(PAGES.find((p) => p.id === store.settingsPage) ?? PAGES[0]);
</script>

<div class="flex min-h-0 min-w-0 flex-1 gap-2">
  <nav
    class="flex w-[188px] shrink-0 flex-col gap-0.5 rounded-[14px] border border-line bg-panel p-2"
    aria-label="Settings"
  >
    {#each PAGES as item}
      <button
        onclick={() => store.openSettings(item.id)}
        aria-current={store.settingsPage === item.id ? "page" : undefined}
        class="nav {store.settingsPage === item.id ? 'active' : ''}"
      >
        <Icon name={item.icon} size={15} />
        {item.label}
        {#if item.id === "personas" && store.personas.length}
          <span class="count">{store.personas.length}</span>
        {/if}
      </button>
    {/each}
    <button onclick={() => store.openSettings(null)} class="nav back mt-auto">
      <Icon name="back" size={14} />
      Conversation
    </button>
  </nav>

  <section
    class="flex min-w-0 flex-1 flex-col overflow-hidden rounded-[14px] border border-line bg-panel"
  >
    <div class="flex h-12 shrink-0 items-center gap-3 border-b border-line px-5">
      <h2 class="text-[13.5px] font-semibold text-ink">{page.label}</h2>
    </div>
    <div class="min-h-0 flex-1 overflow-y-auto px-5 pb-8 pt-4">
      {#if page.id === "pipeline"}
        <Pipeline />
      {:else}
        <Personas />
      {/if}
    </div>
  </section>

  {#if page.id === "pipeline" && store.stage}
    {#key store.stage}
      <StageInspector stage={store.stage} />
    {/key}
  {/if}
</div>

<style>
  .nav {
    display: flex;
    height: 2.1rem;
    align-items: center;
    gap: 0.6rem;
    border-radius: 0.5rem;
    padding: 0 0.65rem;
    font-size: 13px;
    color: var(--color-ink-2);
    transition:
      background-color 150ms,
      color 150ms;
  }
  .nav:hover,
  .nav.active {
    background: var(--color-hover);
    color: var(--color-ink);
  }
  .nav.active :global(svg) {
    color: var(--color-accent);
  }
  .nav.back {
    font-size: 12.5px;
    color: var(--color-ink-3);
  }
  .count {
    margin-left: auto;
    font-size: 11px;
    color: var(--color-ink-3);
    font-variant-numeric: tabular-nums;
  }
</style>
