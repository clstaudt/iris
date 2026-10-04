<script lang="ts">
  import type { AttachmentInfo } from "../lib/state.svelte";
  import Icon from "./Icon.svelte";

  interface Props {
    items: AttachmentInfo[];
    onOpen: (image: AttachmentInfo) => void;
  }

  let { items, onOpen }: Props = $props();

  const images = $derived(items.filter((a) => a.kind === "image" && a.preview));
  const documents = $derived(items.filter((a) => a.kind !== "image" || !a.preview));
</script>

{#if images.length > 0}
  <div class="images" class:single={images.length === 1}>
    {#each images as image}
      <button class="image" onclick={() => onOpen(image)} title={image.name}>
        <img src={image.preview} alt={image.name} />
      </button>
    {/each}
  </div>
{/if}
{#if documents.length > 0}
  <div class="flex flex-wrap justify-end gap-1.5">
    {#each documents as doc}
      <div class="document" title={doc.name}>
        <Icon name="file" size={13} class="shrink-0 text-ink-3" />
        <span class="truncate">{doc.name}</span>
      </div>
    {/each}
  </div>
{/if}

<style>
  .images {
    display: flex;
    flex-wrap: wrap;
    justify-content: flex-end;
    gap: 0.4rem;
  }
  .image {
    overflow: hidden;
    border: 1px solid var(--color-line);
    border-radius: 0.8rem;
    background: var(--color-raised);
    transition: border-color 150ms;
  }
  .image:hover {
    border-color: var(--color-line-strong);
  }
  .image img {
    display: block;
    width: 8.5rem;
    height: 8.5rem;
    object-fit: cover;
  }
  .single .image img {
    width: auto;
    height: auto;
    max-width: min(20rem, 100%);
    max-height: 16rem;
    object-fit: contain;
  }
  .document {
    display: flex;
    max-width: 16rem;
    align-items: center;
    gap: 0.45rem;
    border: 1px solid var(--color-line);
    border-radius: 0.7rem;
    background: var(--color-raised);
    padding: 0.45rem 0.8rem 0.45rem 0.65rem;
    font-size: 12.5px;
    color: var(--color-ink-2);
  }
</style>
