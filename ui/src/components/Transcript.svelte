<script lang="ts">
  import { tick } from "svelte";
  import { renderMarkdown } from "../lib/markdown";
  import { store } from "../lib/state.svelte";
  import Icon from "./Icon.svelte";

  const TOOLS: Record<string, { icon: string; label: string }> = {
    search: { icon: "search", label: "Searched the web" },
    browse: { icon: "globe", label: "Read a web page" },
    look: { icon: "camera", label: "Looked through the camera" },
    capture_screen: { icon: "monitor", label: "Looked at your screen" },
    recall: { icon: "brain", label: "Recalled" },
    search_memory: { icon: "search", label: "Searched memory" },
  };

  // Whole emoji sequences (ZWJ families, flags, skin tones) plus the space
  // before them. Text-style symbols such as (c) or arrows are left alone.
  const EMOJI =
    /[ \t]?(?:[\u{1F1E6}-\u{1F1FF}]{2}|(?:\p{Emoji_Presentation}|\p{Extended_Pictographic}\u{FE0F})(?:\u{200D}\p{Extended_Pictographic}\u{FE0F}?|\p{Emoji_Modifier})*)/gu;
  // Keycaps keep their digit: "1\uFE0F\u20E3" reads as "1"
  const KEYCAP = /\u{FE0F}?\u{20E3}/gu;

  /** Models like to add emoji; the UI shows none. */
  function withoutEmoji(text: string) {
    const cleaned = text.replace(EMOJI, "").replace(KEYCAP, "");
    // Space left behind by a leading emoji, but never real indentation
    return /^\s/.test(text) ? cleaned : cleaned.trimStart();
  }

  // Links open in the browser (main.ts routes window.open there), never in
  // the app window itself
  function onLinkClick(event: MouseEvent) {
    const link = (event.target as HTMLElement).closest("a");
    if (!link) return;
    event.preventDefault();
    if (/^https?:/.test(link.href)) window.open(link.href, "_blank");
  }

  let scrollEl: HTMLDivElement;
  let pinned = true;

  function onScroll() {
    const distance =
      scrollEl.scrollHeight - scrollEl.scrollTop - scrollEl.clientHeight;
    pinned = distance < 60;
  }

  // Auto-scroll to the newest line unless the user scrolled up
  $effect(() => {
    const last = store.messages[store.messages.length - 1];
    void last?.text;
    void store.messages.length;
    void store.pending.length;
    if (pinned) {
      tick().then(() => {
        scrollEl.scrollTop = scrollEl.scrollHeight;
      });
    }
  });
</script>

<div bind:this={scrollEl} onscroll={onScroll} class="min-h-0 flex-1 overflow-y-auto">
  <div
    class="mx-auto flex min-h-full max-w-[720px] flex-col justify-end gap-4 px-8 pb-4 pt-8 text-[14.5px] leading-relaxed"
  >
    {#if store.messages.length === 0 && store.pending.length === 0}
      <p class="pb-6 text-center text-[13px] text-ink-3">
        Speak, or start typing.
      </p>
    {/if}

    {#each store.messages as message}
      {#if message.role === "user"}
        <div class="row justify-end">
          <span class="gutter"></span>
          <div class="bubble user selectable whitespace-pre-wrap">{message.text}</div>
          <span class="gutter" title={message.source === "text" ? "Typed" : "Spoken"}>
            <Icon name={message.source === "text" ? "keyboard" : "mic"} size={13} />
          </span>
        </div>
      {:else if message.role === "assistant"}
        <div class="row">
          <span class="gutter" title="iris"><Icon name="orb" size={13} /></span>
          <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
          <div
            class="bubble assistant md selectable"
            class:streaming={message.streaming}
            onclick={onLinkClick}
          >
            {@html renderMarkdown(withoutEmoji(message.text))}
          </div>
          <span class="gutter"></span>
        </div>
      {:else if message.role === "tool"}
        {@const tool = TOOLS[message.tool ?? ""]}
        <div class="row">
          <span class="gutter"></span>
          <div
            class="flex min-w-0 items-center gap-2 rounded-full border border-line px-3 py-1 text-[12px] text-ink-3"
          >
            <Icon name={tool?.icon ?? "search"} size={12} class="shrink-0" />
            <span class="shrink-0">{tool?.label ?? message.tool}</span>
            {#if message.text}
              <span class="truncate text-ink-2">{message.text}</span>
            {/if}
          </div>
        </div>
      {:else if message.role === "system"}
        <div class="text-center text-[12px] text-ink-3">{message.text}</div>
      {:else}
        <div class="row">
          <span class="gutter text-danger"><Icon name="alert" size={13} /></span>
          <div class="bubble error selectable">{message.text}</div>
          <span class="gutter"></span>
        </div>
      {/if}
    {/each}

    {#each store.pending as turn (turn.id)}
      <div class="row justify-end opacity-55">
        <span class="gutter"></span>
        <div class="bubble pending whitespace-pre-wrap">{turn.text}</div>
        <span class="gutter" title="Waiting for the current reply">
          <Icon name="clock" size={13} />
        </span>
      </div>
    {/each}
  </div>
</div>

<style>
  /* One bubble family: same shape and padding, mirrored tails, the speaker
     marked by alignment, a gutter icon and a fill. */
  .row {
    display: flex;
    align-items: flex-end;
    gap: 0.5rem;
  }
  .gutter {
    display: flex;
    width: 1rem;
    flex-shrink: 0;
    justify-content: center;
    padding-bottom: 0.7rem;
    color: var(--color-ink-3);
  }
  .bubble {
    max-width: 85%;
    min-width: 0;
    border-radius: 1rem;
    border: 1px solid var(--color-line);
    padding: 0.6rem 1rem;
    overflow-wrap: anywhere;
  }
  .bubble.user {
    border-bottom-right-radius: 0.35rem;
    border-color: color-mix(in srgb, var(--color-accent) 18%, transparent);
    background: color-mix(in srgb, var(--color-accent) 9%, transparent);
    color: var(--color-ink);
  }
  .bubble.assistant {
    border-bottom-left-radius: 0.35rem;
    background: var(--color-raised);
    color: color-mix(in srgb, var(--color-ink) 92%, transparent);
  }
  .bubble.error {
    border-bottom-left-radius: 0.35rem;
    border-color: color-mix(in srgb, var(--color-danger) 16%, transparent);
    background: color-mix(in srgb, var(--color-danger) 6%, transparent);
    color: var(--color-danger);
    font-size: 13px;
  }
  .bubble.pending {
    border-bottom-right-radius: 0.35rem;
    border-style: dashed;
    border-color: var(--color-line-strong);
    color: var(--color-ink-2);
  }

  /* Markdown inside assistant bubbles */
  .md :global(> *:first-child) {
    margin-top: 0;
  }
  .md :global(> *:last-child) {
    margin-bottom: 0;
  }
  .md :global(p),
  .md :global(ul),
  .md :global(ol),
  .md :global(pre),
  .md :global(blockquote) {
    margin: 0.55em 0;
  }
  .md :global(h1),
  .md :global(h2),
  .md :global(h3),
  .md :global(h4) {
    margin: 0.8em 0 0.35em;
    font-size: 1em;
    font-weight: 600;
    color: var(--color-ink);
  }
  .md :global(strong) {
    font-weight: 600;
    color: var(--color-ink);
  }
  .md :global(ul) {
    list-style: disc;
    padding-left: 1.25em;
  }
  .md :global(ol) {
    list-style: decimal;
    padding-left: 1.4em;
  }
  .md :global(li) {
    margin: 0.2em 0;
  }
  .md :global(li::marker) {
    color: var(--color-ink-3);
  }
  .md :global(a) {
    color: var(--color-accent);
    text-decoration: underline;
    text-decoration-color: color-mix(in srgb, var(--color-accent) 40%, transparent);
    text-underline-offset: 2px;
  }
  .md :global(code) {
    font-family: var(--font-mono);
    font-size: 0.88em;
    border-radius: 0.3rem;
    background: rgb(255 255 255 / 0.06);
    padding: 0.1em 0.35em;
  }
  .md :global(pre) {
    overflow-x: auto;
    border-radius: 0.6rem;
    border: 1px solid var(--color-line);
    background: var(--color-frame);
    padding: 0.7rem 0.85rem;
  }
  .md :global(pre code) {
    background: none;
    padding: 0;
  }
  .md :global(blockquote) {
    border-left: 2px solid var(--color-line-strong);
    padding-left: 0.8em;
    color: var(--color-ink-2);
  }

  /* While a reply streams, a soft pulsing dot trails its last line */
  .md.streaming :global(> *:last-child::after) {
    content: "";
    display: inline-block;
    width: 7px;
    height: 7px;
    margin-left: 6px;
    border-radius: 50%;
    background: var(--color-accent);
    vertical-align: middle;
    animation: pulse 1.1s ease-in-out infinite;
  }
  @keyframes -global-pulse {
    50% {
      opacity: 0.25;
    }
  }
</style>
