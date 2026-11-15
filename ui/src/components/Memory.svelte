<script lang="ts">
  import { store, type MemoryTopicDetail } from "../lib/state.svelte";
  import Icon from "./Icon.svelte";

  let openId = $state<number | null>(null);
  let detail = $state<MemoryTopicDetail | null>(null);

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
>
  <div class="flex h-12 shrink-0 items-center justify-between pl-5 pr-2.5">
    <h2 class="text-[13px] font-semibold text-ink">Memory</h2>
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
</aside>
