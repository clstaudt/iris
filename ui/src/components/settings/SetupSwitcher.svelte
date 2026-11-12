<script lang="ts">
  import { store } from "../../lib/state.svelte";
  import Icon from "../Icon.svelte";

  // The active setup follows every change made in the pipeline; switching
  // applies another one's choices

  let open = $state(false);
  let root: HTMLElement;
  let naming = $state<"new" | "rename" | null>(null);
  let name = $state("");

  const active = $derived(store.setups[store.activeSetup]);
  const ids = $derived(Object.keys(store.setups));

  function focus(node: HTMLInputElement) {
    node.focus();
    node.select();
  }

  function close() {
    open = false;
    naming = null;
  }

  function startNaming(kind: "new" | "rename") {
    naming = kind;
    name = kind === "rename" ? (active?.name ?? "") : "";
  }

  function commit() {
    if (naming === "new") store.newSetup(name);
    else if (naming === "rename") store.renameSetup(name);
    close();
  }

  async function choose(id: string) {
    close();
    if (id !== store.activeSetup) {
      await store.useSetup(id).catch((err) => console.error("[setups] switch failed:", err));
    }
  }

  async function remove() {
    close();
    await store.deleteSetup().catch((err) => console.error("[setups] delete failed:", err));
  }
</script>

<svelte:window
  onclick={(event) => {
    // composedPath, not closest(): a menu item replaced by its click (New
    // setup turning into the name field) is no longer in the document
    if (open && !event.composedPath().includes(root)) close();
  }}
/>

<div class="switcher" bind:this={root}>
  <button
    class="current"
    onclick={() => (open ? close() : (open = true))}
    aria-haspopup="menu"
    aria-expanded={open}
  >
    <span class="flex min-w-0 flex-1 flex-col text-left">
      <span class="key">Setup</span>
      <span class="truncate text-[13px] text-ink">{active?.name ?? ""}</span>
    </span>
    <Icon name="chevron" size={14} />
  </button>

  {#if open}
    <div class="menu" role="menu">
      {#each ids as id}
        <button
          role="menuitemradio"
          aria-checked={id === store.activeSetup}
          onclick={() => choose(id)}
        >
          <span class="flex w-3.5 shrink-0">
            {#if id === store.activeSetup}<Icon name="check" size={13} />{/if}
          </span>
          <span class="truncate">{store.setups[id].name}</span>
        </button>
      {/each}
      <hr />
      {#if naming}
        <input
          use:focus
          bind:value={name}
          placeholder="Name"
          spellcheck="false"
          aria-label={naming === "new" ? "Name of the new setup" : "New name"}
          onkeydown={(e) => {
            if (e.key === "Enter") commit();
            if (e.key === "Escape") naming = null;
          }}
          class="s-field !h-8"
        />
      {:else}
        <button role="menuitem" onclick={() => startNaming("new")}>
          <Icon name="plus" size={13} />
          New setup
        </button>
        <button role="menuitem" onclick={() => startNaming("rename")}>
          <Icon name="pencil" size={12} />
          Rename
        </button>
        {#if ids.length > 1}
          <button role="menuitem" class="danger" onclick={remove}>
            <Icon name="trash" size={13} />
            Delete
          </button>
        {/if}
      {/if}
    </div>
  {/if}
</div>

<style>
  .switcher {
    position: relative;
    margin-bottom: 0.5rem;
  }
  .current {
    display: flex;
    width: 100%;
    align-items: center;
    gap: 0.5rem;
    padding: 0.45rem 0.65rem;
    border-radius: 0.6rem;
    border: 1px solid var(--color-line);
    background: var(--color-raised);
    color: var(--color-ink-3);
    transition: border-color 150ms;
  }
  .current:hover {
    border-color: var(--color-line-strong);
  }
  .key {
    font-size: 10.5px;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--color-ink-3);
  }
  .menu {
    position: absolute;
    z-index: 20;
    left: 0;
    right: 0;
    top: calc(100% + 6px);
    display: flex;
    flex-direction: column;
    gap: 1px;
    padding: 0.3rem;
    border-radius: 0.6rem;
    border: 1px solid var(--color-line-strong);
    background: var(--color-raised);
    box-shadow: 0 12px 30px rgb(0 0 0 / 0.45);
  }
  .menu button {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.4rem 0.5rem;
    border-radius: 0.4rem;
    font-size: 12.5px;
    color: var(--color-ink-2);
    text-align: left;
  }
  .menu button:hover {
    background: var(--color-hover);
    color: var(--color-ink);
  }
  .menu button.danger:hover {
    color: var(--color-danger);
  }
  .menu hr {
    margin: 0.25rem 0.15rem;
    border: 0;
    border-top: 1px solid var(--color-line);
  }
</style>
