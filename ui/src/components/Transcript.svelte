<script lang="ts">
  import { tick } from "svelte";
  import { store } from "../lib/state.svelte";

  const TOOL_LABELS: Record<string, string> = {
    search: "searched",
    browse: "browsed",
    look: "looked",
    capture_screen: "captured screen",
  };

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
    if (pinned) {
      tick().then(() => {
        scrollEl.scrollTop = scrollEl.scrollHeight;
      });
    }
  });
</script>

<div
  bind:this={scrollEl}
  onscroll={onScroll}
  class="min-h-0 flex-1 overflow-y-auto px-6 pt-4"
>
  <div class="flex min-h-full flex-col justify-end gap-1.5 pb-3 font-mono text-[13px] leading-relaxed">
    {#each store.messages as message}
      {#if message.role === "user"}
        <div class="text-cyan-300">
          <span class="mr-2 select-none text-cyan-500">›</span>{message.text}
        </div>
      {:else if message.role === "assistant"}
        <div class="whitespace-pre-wrap text-emerald-300/90">
          {message.text}{#if message.streaming}<span class="cursor">▌</span>{/if}
        </div>
      {:else if message.role === "tool"}
        <div class="text-center text-[11px] text-neutral-500">
          ⌕ {TOOL_LABELS[message.tool ?? ""] ?? message.tool}{message.text
            ? `: ${message.text}`
            : ""}
        </div>
      {:else if message.role === "system"}
        <div class="text-center text-[11px] italic text-neutral-600">
          {message.text}
        </div>
      {:else}
        <div class="text-red-400">✗ {message.text}</div>
      {/if}
    {/each}
  </div>
</div>

<style>
  .cursor {
    animation: blink 1s steps(1) infinite;
  }
  @keyframes blink {
    50% {
      opacity: 0;
    }
  }
</style>
