<script lang="ts">
  import { store, type ThreadInfo } from "../lib/state.svelte";
  import Icon from "./Icon.svelte";
  import PersonaRing from "./PersonaRing.svelte";

  /** The thread whose menu is open, and whether its forgetting is asked */
  let menuFor = $state<number | null>(null);
  let confirming = $state(false);

  const GROUPS = ["Today", "Yesterday", "Last week", "Older"];

  /** Which group a thread falls in, by the day it was last talked in. */
  function group(thread: ThreadInfo, now = new Date()): string {
    const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
    const day = 86_400_000;
    const at = thread.updated * 1000;
    if (at >= startOfToday) return "Today";
    if (at >= startOfToday - day) return "Yesterday";
    if (at >= startOfToday - 7 * day) return "Last week";
    return "Older";
  }

  const grouped = $derived(
    GROUPS.map((name) => ({ name, threads: store.threads.filter((t) => group(t) === name) })).filter(
      (g) => g.threads.length
    )
  );

  function openMenu(event: MouseEvent, id: number) {
    event.stopPropagation();
    confirming = false;
    menuFor = menuFor === id ? null : id;
  }

  async function forget(event: MouseEvent, id: number) {
    event.stopPropagation();
    if (!confirming) {
      confirming = true;
      return;
    }
    menuFor = null;
    confirming = false;
    await store.forgetThread(id);
  }

  function closeMenu() {
    menuFor = null;
    confirming = false;
  }
</script>

<svelte:window onclick={closeMenu} />

<aside
  class="flex w-[256px] shrink-0 flex-col overflow-hidden rounded-[14px] border border-line bg-panel"
  aria-label="Threads"
>
  <div class="px-2.5 pb-1.5 pt-2.5">
    <button onclick={() => store.newThread()} class="new">
      <Icon name="plus" size={15} />
      New thread
    </button>
  </div>
  <div class="min-h-0 flex-1 overflow-y-auto px-1.5 pb-2.5">
    {#each grouped as g (g.name)}
      <h3 class="s-section px-2.5 pb-1.5 pt-3.5">{g.name}</h3>
      {#each g.threads as thread (thread.id)}
        <div class="item {thread.id === store.openThread ? 'active' : ''}" data-thread={thread.id}>
          <button class="open" onclick={() => store.openThreadById(thread.id)}>
            <PersonaRing name={thread.persona_name} />
            <span class="flex min-w-0 flex-1 flex-col gap-0.5">
              <span class="title">{thread.title || "New thread"}</span>
              <span class="who">{thread.persona_name}</span>
            </span>
          </button>
          <button
            class="more {menuFor === thread.id ? 'shown' : ''}"
            onclick={(e) => openMenu(e, thread.id)}
            aria-label="More"
            title="More"
          >
            <Icon name="more" size={15} />
          </button>
          {#if menuFor === thread.id}
            <div class="menu">
              <button class={confirming ? "danger" : ""} onclick={(e) => forget(e, thread.id)}>
                <Icon name="trash" size={13} />
                {confirming ? "Confirm forget" : "Forget this thread"}
              </button>
            </div>
          {/if}
        </div>
      {/each}
    {/each}
  </div>
</aside>

<style>
  .new {
    display: flex;
    width: 100%;
    height: 34px;
    align-items: center;
    gap: 0.5rem;
    padding: 0 0.75rem;
    border-radius: 9px;
    border: 1px solid var(--color-line);
    background: var(--color-raised);
    font-size: 13px;
    color: var(--color-ink);
    transition: background-color 150ms;
  }
  .new:hover {
    background: var(--color-hover);
  }
  .item {
    position: relative;
    display: flex;
    align-items: center;
    border-radius: 9px;
    transition: background-color 150ms;
  }
  .item:hover {
    background: color-mix(in srgb, var(--color-hover) 60%, transparent);
  }
  .item.active {
    background: var(--color-hover);
  }
  .open {
    display: flex;
    min-width: 0;
    flex: 1;
    align-items: center;
    gap: 0.6rem;
    padding: 0.5rem 0 0.5rem 0.6rem;
    text-align: left;
  }
  .title {
    overflow: hidden;
    font-size: 13px;
    color: var(--color-ink);
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .who {
    overflow: hidden;
    font-size: 11.5px;
    color: var(--color-ink-3);
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .more {
    display: flex;
    height: 1.5rem;
    width: 1.5rem;
    flex-shrink: 0;
    margin: 0 0.35rem;
    align-items: center;
    justify-content: center;
    border-radius: 0.375rem;
    color: var(--color-ink-3);
    opacity: 0;
    transition:
      opacity 150ms,
      background-color 150ms,
      color 150ms;
  }
  .item:hover .more,
  .more.shown {
    opacity: 1;
  }
  .more:hover {
    background: var(--color-hover);
    color: var(--color-ink);
  }
  .menu {
    position: absolute;
    top: calc(100% - 4px);
    right: 6px;
    z-index: 10;
    padding: 4px;
    border-radius: 10px;
    border: 1px solid var(--color-line-strong);
    background: var(--color-raised);
    box-shadow: 0 16px 40px rgb(0 0 0 / 0.45);
  }
  .menu button {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.45rem 0.65rem;
    border-radius: 7px;
    font-size: 12.5px;
    color: var(--color-ink-2);
    white-space: nowrap;
    transition:
      background-color 150ms,
      color 150ms;
  }
  .menu button:hover {
    background: var(--color-hover);
    color: var(--color-ink);
  }
  .menu button.danger {
    color: var(--color-danger);
  }
  .menu button.danger:hover {
    background: color-mix(in srgb, var(--color-danger) 10%, transparent);
  }
</style>
