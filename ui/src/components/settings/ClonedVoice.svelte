<script lang="ts">
  import { store, type PersonaSettings, type VoiceInfo } from "../../lib/state.svelte";
  import { percent, running } from "../../lib/voices";
  import Icon from "../Icon.svelte";
  import JobLine from "../JobLine.svelte";
  import VoiceClone from "./VoiceClone.svelte";

  // A voice cloned here: its name, given in place; the languages made of its
  // sample, and the others to make now; where its files are; cloned again
  // from a new sample, keeping its name and personas; and deleted. A voice
  // personas speak with is deleted once the user agrees to them speaking
  // with Kyutai's default voice instead, which the card names.

  let {
    voice,
    focus = false,
    onplay,
  }: { voice: VoiceInfo; focus?: boolean; onplay: () => void } = $props();

  let name = $state("");
  let field = $state<HTMLInputElement | null>(null);
  let confirming = $state(false);
  let error = $state("");

  $effect(() => {
    name = voice.name;
  });
  // Cloned just now: its name is to be given
  $effect(() => {
    if (focus && field) {
      field.focus();
      field.select();
    }
  });

  const users = $derived(voice.personas ?? []);
  const languages = $derived(
    (store.voices?.languages ?? []).filter((l) => (voice.languages ?? []).includes(l.code))
  );
  const made = $derived(voice.cloned?.made ?? []);
  const fallback = $derived(
    store.voices?.engines.kyutai?.find((v) => v.id === store.voices?.fallback)?.name ?? "Eve"
  );
  const jobs = $derived(
    Object.values(store.jobs).filter((j) => j.voice === voice.id && j.state !== "ready")
  );
  /** As VoiceClone takes it: cloned again, the voice keeps its name */
  const again = $derived<PersonaSettings>({
    name: voice.name,
    engine: "kyutai",
    voice: voice.id,
    speed: 1,
    audio_filter: "",
    description: "",
  });

  function together(names: string[]) {
    return names.length <= 1 ? names.join("") : `${names.slice(0, -1).join(", ")} and ${names.at(-1)}`;
  }

  async function rename() {
    const wanted = name.trim();
    if (!wanted || wanted === voice.name) {
      name = voice.name;
      return;
    }
    error = (await store.renameVoice(voice.id, wanted)) ?? "";
    if (error) name = voice.name;
  }

  function keys(event: KeyboardEvent) {
    if (event.key === "Enter") field?.blur();
    if (event.key === "Escape") {
      name = voice.name;
      field?.blur();
    }
  }

  async function remove() {
    if (!confirming) {
      confirming = true;
      return;
    }
    error = (await store.deleteVoice(voice.id, users.length > 0)) ?? "";
    confirming = false;
  }
</script>

<article class="cloned" data-voice={voice.id}>
  <div class="head">
    <button
      class="s-play {store.previewing === `kyutai:${voice.id}` ? 'playing' : ''}"
      onclick={onplay}
      aria-label="Play {voice.name}"
    >
      <Icon name="play" size={10} />
    </button>
    <input
      bind:this={field}
      bind:value={name}
      onkeydown={keys}
      onblur={rename}
      class="title"
      aria-label="Name of the voice"
      spellcheck="false"
    />
    {#each users as persona (persona.id)}
      <button class="user" onclick={() => store.editPersona(persona.id)} title="Open {persona.name}">
        {persona.name}
      </button>
    {/each}
    {#if !(confirming && users.length)}
      <button
        class="s-btn danger"
        onclick={remove}
        onblur={() => {
          if (!users.length) confirming = false;
        }}
      >
        {confirming ? "Confirm delete" : "Delete"}
      </button>
    {/if}
  </div>

  {#if confirming && users.length}
    <div class="guard" role="alert">
      <span>
        {together(users.map((p) => p.name))}
        {users.length === 1 ? "speaks" : "speak"} with this voice, and will speak with {fallback}.
      </span>
      <span class="flex shrink-0 gap-1.5">
        <button class="s-btn" onclick={() => (confirming = false)}>Cancel</button>
        <button class="s-btn danger" onclick={remove}>Delete</button>
      </span>
    </div>
  {/if}
  {#if error}
    <p class="s-status error">{error}</p>
  {/if}

  <dl class="facts">
    <dt>Languages</dt>
    <dd class="langs">
      {#each languages as language (language.code)}
        {@const job = store.jobFor("kyutai", voice.id, language.code)}
        {#if made.includes(language.code)}
          <span class="lang made" data-language={language.code}>{language.name}</span>
        {:else if job && running(job)}
          <span class="lang busy tabular-nums" data-language={language.code}>
            {language.name}
            {percent(job) !== null && job.state === "downloading" ? `${percent(job)} %` : ""}
          </span>
        {:else}
          <button
            class="lang make"
            data-language={language.code}
            onclick={() => store.makeReady("kyutai", voice.id, [language.code])}
            title="Make {voice.name} for {language.name}"
          >
            <Icon name="plus" size={10} />
            {language.name}
          </button>
        {/if}
      {/each}
    </dd>
    {#if voice.cloned?.folder}
      <dt>Folder</dt>
      <dd class="path selectable" title={voice.cloned.folder}>{voice.cloned.folder}</dd>
    {/if}
  </dl>

  {#if jobs.length}
    <div class="jobs">
      {#each jobs as job (job.id)}
        <JobLine {job} quiet />
      {/each}
    </div>
  {/if}

  <VoiceClone draft={again} keep onchoose={() => {}} />
</article>

<style>
  .cloned {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    padding: 0.6rem 0.75rem 0.75rem 0.6rem;
    border-radius: 0.65rem;
    border: 1px solid var(--color-line);
    background: var(--color-raised);
  }
  .head {
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .title {
    min-width: 0;
    flex: 1;
    height: 1.9rem;
    padding: 0 0.45rem;
    border-radius: 0.45rem;
    border: 1px solid transparent;
    background: transparent;
    font-size: 13.5px;
    font-weight: 500;
    color: var(--color-ink);
    outline: none;
  }
  .title:hover,
  .title:focus {
    border-color: var(--color-line-strong);
    background: var(--color-panel);
  }
  .user {
    flex-shrink: 0;
    border-radius: 0.375rem;
    padding: 0 0.45rem;
    font-size: 11px;
    line-height: 1.25rem;
    color: var(--color-accent);
    background: color-mix(in srgb, var(--color-accent) 12%, transparent);
  }
  .user:hover {
    background: color-mix(in srgb, var(--color-accent) 20%, transparent);
  }
  .guard {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    padding: 0.5rem 0.6rem 0.5rem 0.75rem;
    border-radius: 0.5rem;
    background: color-mix(in srgb, var(--color-danger) 7%, transparent);
    font-size: 12px;
    color: var(--color-ink-2);
  }
  .facts {
    display: grid;
    grid-template-columns: max-content minmax(0, 1fr);
    align-items: baseline;
    gap: 0.45rem 0.9rem;
    padding-left: 2.1rem;
    font-size: 12px;
    color: var(--color-ink-2);
  }
  .facts dt {
    color: var(--color-ink-3);
  }
  .langs {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
  }
  .lang {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    height: 1.4rem;
    padding: 0 0.5rem;
    border-radius: 9999px;
    border: 1px solid var(--color-line);
    font-size: 11px;
    color: var(--color-ink-3);
  }
  .lang.made {
    border-color: color-mix(in srgb, var(--color-accent) 35%, transparent);
    color: var(--color-ink);
  }
  .lang.busy {
    border-color: color-mix(in srgb, var(--color-petal-cyan) 40%, transparent);
    color: var(--color-ink-2);
  }
  .lang.make:hover {
    border-color: var(--color-line-strong);
    color: var(--color-ink);
  }
  .path {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 11.5px;
    color: var(--color-ink-3);
  }
  .jobs {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
    padding-left: 2.1rem;
  }
  .cloned > :global(div:last-child) {
    padding-left: 2.1rem;
  }
</style>
