<script lang="ts">
  import { store, type MemoryTopicDetail } from "../lib/state.svelte";
  import Icon from "./Icon.svelte";

  let openId = $state<number | null>(null);
  let detail = $state<MemoryTopicDetail | null>(null);
  /** Forgetting the thread is asked, to be confirmed */
  let confirming = $state(false);

  // Another thread: its own topics, none of them open
  $effect(() => {
    void store.openThread;
    openId = null;
    detail = null;
    confirming = false;
  });

  async function forgetThread() {
    if (store.openThread === null) return;
    confirming = false;
    openId = null;
    detail = null;
    await store.forgetThread(store.openThread);
  }

  function relativeDay(ts: number): string {
    const days = Math.floor((Date.now() / 1000 - ts) / 86400);
    if (days <= 0) return "today";
    if (days === 1) return "yesterday";
    return `${days} days ago`;
  }

  async function toggle(id: number) {
    if (openId === id) {
      openId = null;
      detail = null;
      return;
    }
    openId = id;
    detail = null;
    detail = await store.memoryTopic(id);
  }

  async function forget(id: number) {
    openId = null;
    detail = null;
    await store.forgetTopic(id);
  }
</script>

<aside
  class="flex w-[340px] shrink-0 flex-col overflow-hidden rounded-[14px] border border-line bg-panel"
  aria-label="Memory"
>
  <div class="flex h-12 shrink-0 items-center justify-between pl-5 pr-2.5">
    <h2 class="flex min-w-0 items-baseline gap-2 text-[13px] font-semibold text-ink">
      Memory
      <span class="truncate text-[12px] font-normal text-ink-3"
        >{store.thread?.title || "New thread"}</span
      >
    </h2>
    <button
      onclick={() => (store.panel = null)}
      class="flex h-7 w-7 items-center justify-center rounded-md text-ink-3 transition-colors hover:bg-hover hover:text-ink"
      title="Close memory"
    >
      <Icon name="close" size={15} />
    </button>
  </div>

  <div class="min-h-0 flex-1 overflow-y-auto pb-4">
    {#if store.memoryTopics.length === 0}
      <p class="px-5 pt-2 text-[12px] text-ink-3">Nothing remembered yet</p>
    {/if}

    {#each store.memoryTopics as topic (topic.id)}
      <div class="border-t border-line first:border-t-0">
        <button
          onclick={() => toggle(topic.id)}
          class="flex w-full flex-col gap-0.5 px-5 py-2.5 text-left transition-colors hover:bg-hover/60 {openId ===
          topic.id
            ? 'bg-hover/40'
            : ''}"
        >
          <div class="flex w-full items-baseline justify-between gap-3">
            <span class="truncate text-[13px] text-ink">{topic.title}</span>
            <span class="shrink-0 text-[11px] text-ink-3">{relativeDay(topic.updated)}</span>
          </div>
          <span class="line-clamp-2 text-[12px] leading-snug text-ink-3">{topic.gist}</span>
        </button>

        {#if openId === topic.id}
          <div class="selectable flex flex-col gap-3 px-5 pb-4 pt-1">
            {#if detail}
              <p class="text-[12.5px] leading-relaxed text-ink-2">{detail.summary}</p>
              <div class="flex flex-col gap-2 border-l border-line pl-3">
                {#each detail.turns as turn (turn.id)}
                  <div class="flex gap-2 text-[12px] leading-snug">
                    <span class="mt-[2px] shrink-0 text-ink-3">
                      <Icon
                        name={turn.role === "assistant"
                          ? "orb"
                          : turn.source === "text"
                            ? "keyboard"
                            : "mic"}
                        size={11}
                      />
                    </span>
                    <span class={turn.role === "assistant" ? "text-ink-2" : "text-ink"}>
                      {turn.text}
                    </span>
                  </div>
                {/each}
              </div>
              <button
                onclick={() => forget(topic.id)}
                class="self-start rounded-md px-2 py-1 text-[12px] text-danger/80 transition-colors hover:bg-danger/10 hover:text-danger"
              >
                Forget
              </button>
            {/if}
          </div>
        {/if}
      </div>
    {/each}
  </div>

  {#if store.openThread !== null}
    {#if confirming}
      <div class="confirm">
        <p>Forget this thread and everything Iris remembers of it?</p>
        <div class="flex justify-end gap-2">
          <button class="s-btn" onclick={() => (confirming = false)}>Cancel</button>
          <button class="s-btn forget" onclick={forgetThread}>Forget this thread</button>
        </div>
      </div>
    {:else}
      <div class="flex shrink-0 gap-2 border-t border-line px-3.5 py-3">
        <button class="s-btn" onclick={() => (confirming = true)}>
          <Icon name="trash" size={13} />
          Forget this thread
        </button>
      </div>
    {/if}
  {/if}
</aside>

<style>
  .confirm {
    display: flex;
    flex-shrink: 0;
    flex-direction: column;
    gap: 0.75rem;
    padding: 0.875rem 1rem;
    border-top: 1px solid color-mix(in srgb, var(--color-danger) 25%, transparent);
    background: color-mix(in srgb, var(--color-danger) 5%, transparent);
  }
  .confirm p {
    font-size: 13px;
    line-height: 1.5;
    color: var(--color-ink);
  }
  .forget {
    border-color: color-mix(in srgb, var(--color-danger) 45%, transparent);
    background: color-mix(in srgb, var(--color-danger) 16%, transparent);
    color: #f0b3b9;
  }
  .forget:hover:not(:disabled) {
    background: color-mix(in srgb, var(--color-danger) 24%, transparent);
    color: #f0b3b9;
  }
</style>
