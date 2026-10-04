<script lang="ts">
  import { onMount } from "svelte";
  import { store } from "../lib/state.svelte";
  import Icon from "./Icon.svelte";

  const MAX_HEIGHT = 168;

  let text = $state("");
  let focused = $state(false);
  let field: HTMLTextAreaElement;

  const hasDraft = $derived(text.trim().length > 0);

  // A focused draft keeps the mic closed: keyboard clatter must not become a
  // spoken turn, and the user has plainly chosen to type this one
  $effect(() => {
    store.setTyping(focused && hasDraft);
  });

  function resize() {
    field.style.height = "auto";
    field.style.height = `${Math.min(field.scrollHeight, MAX_HEIGHT)}px`;
  }

  async function send() {
    const value = text.trim();
    if (!value) return;
    text = "";
    queueMicrotask(resize);
    await store.send(value);
  }

  function onKeydown(event: KeyboardEvent) {
    // isComposing: Enter confirms an IME candidate (Japanese, Chinese)
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      send();
    } else if (event.key === "Escape") {
      field.blur();
    }
  }

  // Typing anywhere in the window lands in the composer
  onMount(() => {
    function onWindowKey(event: KeyboardEvent) {
      const target = event.target as HTMLElement | null;
      if (target && /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName)) return;
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      if (event.key.length === 1) field.focus();
    }
    window.addEventListener("keydown", onWindowKey);
    return () => window.removeEventListener("keydown", onWindowKey);
  });
</script>

<div class="relative z-10 shrink-0 px-6 pb-1">
  <div
    class="composer mx-auto max-w-[720px] rounded-2xl border bg-raised transition-colors {focused
      ? 'border-line-strong'
      : 'border-line'}"
  >
    <textarea
      bind:this={field}
      bind:value={text}
      oninput={resize}
      onkeydown={onKeydown}
      onfocus={() => (focused = true)}
      onblur={() => (focused = false)}
      rows="1"
      placeholder="Message iris"
      class="block w-full resize-none bg-transparent px-4 pb-1 pt-3.5 text-[14.5px] leading-relaxed text-ink outline-none placeholder:text-ink-3"
    ></textarea>

    <div class="flex items-center justify-between px-2.5 pb-2.5 pt-1">
      <div class="flex items-center gap-1">
        <button
          onclick={() => store.togglePause()}
          disabled={store.state === "starting"}
          class="toggle {store.paused ? 'resume' : ''}"
          title={store.paused ? "Resume the conversation" : "Pause the conversation"}
        >
          <Icon name={store.paused ? "play" : "pause"} size={14} />
          <span>{store.paused ? "Resume" : "Pause"}</span>
        </button>
        <span class="mx-1 h-4 w-px bg-line-strong"></span>
        <button
          onclick={() => store.toggleMic()}
          disabled={store.state === "starting"}
          class="toggle {store.micOn ? '' : 'off'}"
          title={store.micOn ? "Turn microphone off" : "Turn microphone on"}
        >
          <Icon name={store.micOn ? "mic" : "mic-off"} />
          <span>{store.micOn ? "Mic" : "Mic off"}</span>
        </button>
        <button
          onclick={() => store.toggleVoice()}
          disabled={store.state === "starting"}
          class="toggle {store.speakReplies ? '' : 'off'}"
          title={store.speakReplies ? "Reply in text only" : "Speak replies"}
        >
          <Icon name={store.speakReplies ? "volume" : "volume-off"} />
          <span>{store.speakReplies ? "Voice" : "Voice off"}</span>
        </button>
        {#if store.config?.memory}
          <button
            onclick={() => store.toggleAmnesia()}
            disabled={store.state === "starting"}
            class="toggle {store.amnesia ? 'resume' : ''}"
            title={store.amnesia
              ? "Amnesia mode is on: nothing is remembered"
              : "Amnesia mode: remember nothing of this conversation"}
          >
            <Icon name="brain-off" size={15} />
            <span>Amnesia</span>
          </button>
        {/if}
      </div>

      <div class="flex items-center gap-1.5">
        {#if store.busy}
          <button
            onclick={() => store.interrupt()}
            class="flex h-8 items-center gap-1.5 rounded-full border border-line-strong px-3 text-[12.5px] text-ink-2 transition-colors hover:bg-hover hover:text-ink"
            title="Stop the current reply"
          >
            <Icon name="stop" size={13} />
            Stop
          </button>
        {/if}
        <button
          onclick={send}
          disabled={!hasDraft}
          class="flex h-8 w-8 items-center justify-center rounded-full bg-accent text-frame transition-opacity hover:opacity-90 disabled:bg-hover disabled:text-ink-3"
          title="Send (Enter)"
        >
          <Icon name="send" size={16} stroke={2.25} />
        </button>
      </div>
    </div>
  </div>
</div>

<style>
  .toggle {
    display: flex;
    height: 2rem;
    align-items: center;
    gap: 0.4rem;
    border-radius: 9999px;
    padding: 0 0.7rem;
    font-size: 12.5px;
    color: var(--color-ink-2);
    transition:
      background-color 150ms,
      color 150ms;
  }
  .toggle:hover:not(:disabled) {
    background: var(--color-hover);
    color: var(--color-ink);
  }
  .toggle.resume {
    color: var(--color-accent);
    background: color-mix(in srgb, var(--color-accent) 12%, transparent);
  }
  .toggle.off {
    color: var(--color-danger);
    background: color-mix(in srgb, var(--color-danger) 10%, transparent);
  }
  /* Active states keep their color under the pointer */
  .toggle.resume:hover:not(:disabled) {
    color: var(--color-accent);
    background: color-mix(in srgb, var(--color-accent) 18%, transparent);
  }
  .toggle.off:hover:not(:disabled) {
    color: var(--color-danger);
    background: color-mix(in srgb, var(--color-danger) 16%, transparent);
  }
  .toggle:disabled {
    opacity: 0.4;
  }
</style>
