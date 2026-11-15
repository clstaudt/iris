<script lang="ts">
  import { onMount } from "svelte";
  import {
    PREFERRED_MODEL,
    serverName,
    walkthrough as w,
  } from "../../lib/walkthrough.svelte";
  import Icon from "../Icon.svelte";
  import Step from "./Step.svelte";

  const SERVER_CHECK_MS = 15_000;

  let adding = $state(false);
  let typed = $state("");

  // Whether the servers answer is asked again while the step is shown, so a
  // server started meanwhile shows up
  onMount(() => {
    const timer = setInterval(() => w.checkServers().catch(() => {}), SERVER_CHECK_MS);
    return () => clearInterval(timer);
  });

  const noServer = $derived(
    w.searched && w.connecting === null && w.server === null && !w.servers.some((s) => s.reachable)
  );

  function add() {
    if (!typed.trim()) return;
    w.addServer(typed);
    typed = "";
    adding = false;
  }

  function focus(node: HTMLInputElement) {
    node.focus();
  }
</script>

{#snippet serverRow(url: string, reachable: boolean | undefined)}
  <div class="s-option server {url === w.server ? 'active' : ''}">
    <button
      class="pick"
      role="radio"
      aria-checked={url === w.server}
      onclick={() => w.connect(url)}
    >
      <span class="radio"></span>
      <span class="flex min-w-0 flex-1 flex-col">
        <span class="name truncate" title={url || w.defaultUrl}>{serverName(url)}</span>
        {#if w.connecting === url}
          <span class="meta">Connecting</span>
        {:else if !url && w.defaultUrl}
          <span class="meta truncate">Ollama, {serverName(w.defaultUrl)}</span>
        {/if}
      </span>
    </button>
    <span
      class="dot {reachable === true ? 'ok' : reachable === false ? 'down' : ''}"
      title={reachable === true ? "Reachable" : reachable === false ? "Unreachable" : "Checking"}
    ></span>
  </div>
{/snippet}

<Step
  title="Language model"
  line="Iris thinks with a language model, on a server you choose."
>
  <section class="sec" aria-label="Server">
    <h3 class="s-section">Server</h3>
    {#each w.servers as server (server.url)}
      {@render serverRow(server.url, server.reachable)}
    {/each}
    {#if adding}
      <div class="flex gap-1.5">
        <input
          use:focus
          bind:value={typed}
          onkeydown={(e) => {
            if (e.key === "Enter") add();
            if (e.key === "Escape") adding = false;
          }}
          placeholder="http://host:11434"
          spellcheck="false"
          aria-label="Server address"
          class="s-field min-w-0 flex-1"
        />
        <button class="s-btn h-9" disabled={!typed.trim()} onclick={add}>Add</button>
      </div>
    {:else}
      <div class="flex items-center gap-2">
        <button class="s-btn" onclick={() => (adding = true)}>
          <Icon name="plus" size={13} />
          Add server
        </button>
        {#if noServer}
          <button class="s-btn" onclick={() => w.searchAgain()} disabled={w.searching}>
            Check again
          </button>
        {/if}
      </div>
    {/if}
    {#if w.serverError}
      <p class="s-status error">{w.serverError}</p>
    {:else if noServer}
      <p class="s-status">
        No server found. Install Ollama from ollama.com and run
        <code>ollama pull {PREFERRED_MODEL}</code>, or add a server you have.
      </p>
    {/if}
  </section>

  {#if w.server !== null}
    <section class="sec" aria-label="Model">
      <h3 class="s-section">Model</h3>
      {#each w.serverModels as model (model.name)}
        <button
          onclick={() => (w.model = model.name)}
          class="s-option {model.name === w.model ? 'active' : ''}"
          role="radio"
          aria-checked={model.name === w.model}
        >
          <span class="radio"></span>
          <span class="name min-w-0 flex-1 truncate">{model.name}</span>
          {#if model.vision}<span class="s-chip">vision</span>{/if}
          {#if model.tools}<span class="s-chip">tools</span>{/if}
        </button>
      {/each}
      {#if w.serverModels.length === 0}
        <p class="s-status">
          This server has no models yet. With Ollama: <code>ollama pull {PREFERRED_MODEL}</code>
        </p>
      {/if}
    </section>
  {/if}
</Step>

<style>
  .sec {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }
  code {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--color-ink-2);
  }
  /* A server: its row picks it, beside it whether it answers */
  .server {
    gap: 0.5rem;
    padding: 0 0.75rem 0 0;
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
</style>
