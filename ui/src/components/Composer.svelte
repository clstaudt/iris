<script lang="ts">
  import { onMount } from "svelte";
  import { releasePreviews, store, type DraftFile } from "../lib/state.svelte";
  import Icon from "./Icon.svelte";

  const MAX_HEIGHT = 168;

  let text = $state("");
  let files = $state<DraftFile[]>([]);
  let focused = $state(false);
  /** Files are being dragged over the window */
  let dragging = $state(false);
  let field: HTMLTextAreaElement;
  let picker: HTMLInputElement;

  const hasDraft = $derived(text.trim().length > 0 || files.length > 0);

  // A focused draft keeps the mic closed: keyboard clatter must not become a
  // spoken turn, and the user has plainly chosen to type this one
  $effect(() => {
    store.setTyping(focused && hasDraft);
  });

  function resize() {
    field.style.height = "auto";
    field.style.height = `${Math.min(field.scrollHeight, MAX_HEIGHT)}px`;
  }

  function addFiles(list: FileList | null) {
    for (const file of list ?? []) {
      const image = file.type.startsWith("image/");
      files.push({
        file,
        info: {
          name: file.name || (image ? "image" : "file"),
          kind: image ? "image" : "document",
          mime: file.type,
          preview: image ? URL.createObjectURL(file) : undefined,
        },
      });
    }
    field.focus();
  }

  function removeFile(index: number) {
    releasePreviews([files[index].info]);
    files.splice(index, 1);
  }

  async function send() {
    const value = text.trim();
    if (!hasDraft) return;
    const sending = files;
    text = "";
    files = [];
    queueMicrotask(resize);
    // Refused (a file iris cannot read): the draft comes back to fix
    if (!(await store.send(value, sending)) && !text && files.length === 0) {
      text = value;
      files = sending;
      queueMicrotask(resize);
    }
  }

  // Screenshots and copied images paste straight in as files
  function onPaste(event: ClipboardEvent) {
    const pasted = event.clipboardData?.files;
    if (pasted && pasted.length > 0) {
      event.preventDefault();
      addFiles(pasted);
    }
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

    // Files dropped anywhere in the window are attached
    let depth = 0;
    const carriesFiles = (event: DragEvent) =>
      event.dataTransfer?.types.includes("Files") ?? false;
    function onDragEnter(event: DragEvent) {
      if (!carriesFiles(event)) return;
      depth += 1;
      dragging = true;
    }
    function onDragLeave(event: DragEvent) {
      if (!carriesFiles(event)) return;
      depth = Math.max(0, depth - 1);
      dragging = depth > 0;
    }
    function onDragOver(event: DragEvent) {
      if (!carriesFiles(event)) return;
      event.preventDefault();
      event.dataTransfer!.dropEffect = "copy";
    }
    function onDrop(event: DragEvent) {
      if (!carriesFiles(event)) return;
      event.preventDefault();
      depth = 0;
      dragging = false;
      addFiles(event.dataTransfer!.files);
    }
    window.addEventListener("dragenter", onDragEnter);
    window.addEventListener("dragleave", onDragLeave);
    window.addEventListener("dragover", onDragOver);
    window.addEventListener("drop", onDrop);
    return () => {
      window.removeEventListener("keydown", onWindowKey);
      window.removeEventListener("dragenter", onDragEnter);
      window.removeEventListener("dragleave", onDragLeave);
      window.removeEventListener("dragover", onDragOver);
      window.removeEventListener("drop", onDrop);
      releasePreviews(files.map((f) => f.info));
    };
  });
</script>

<div class="relative z-10 shrink-0 px-6 pb-1">
  <div
    class="composer mx-auto max-w-[720px] rounded-2xl border bg-raised transition-colors {dragging
      ? 'border-accent'
      : focused
        ? 'border-line-strong'
        : 'border-line'}"
  >
    {#if files.length > 0}
      <div class="flex flex-wrap gap-2 px-3 pt-3">
        {#each files as draft, index (draft.info)}
          <div class="draft {draft.info.kind}" title={draft.info.name}>
            {#if draft.info.kind === "image"}
              <img src={draft.info.preview} alt={draft.info.name} />
            {:else}
              <Icon name="file" size={14} class="shrink-0 text-ink-3" />
              <span class="truncate">{draft.info.name}</span>
            {/if}
            <button class="remove" onclick={() => removeFile(index)} title="Remove">
              <Icon name="close" size={10} stroke={2.5} />
            </button>
          </div>
        {/each}
      </div>
    {/if}
    <textarea
      bind:this={field}
      bind:value={text}
      oninput={resize}
      onkeydown={onKeydown}
      onfocus={() => (focused = true)}
      onblur={() => (focused = false)}
      onpaste={onPaste}
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
            disabled={store.state === "starting" || !store.memoryOn}
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
          onclick={() => picker.click()}
          class="flex h-8 w-8 items-center justify-center rounded-full text-ink-2 transition-colors hover:bg-hover hover:text-ink"
          title="Attach files"
        >
          <Icon name="paperclip" size={16} />
        </button>
        <input
          bind:this={picker}
          type="file"
          multiple
          class="hidden"
          onchange={() => {
            addFiles(picker.files);
            picker.value = "";
          }}
        />
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

  /* Files waiting to be sent */
  .draft {
    position: relative;
    display: flex;
    align-items: center;
    border: 1px solid var(--color-line-strong);
    border-radius: 0.6rem;
    background: var(--color-panel);
  }
  .draft.image {
    width: 3.5rem;
    height: 3.5rem;
  }
  .draft.image img {
    width: 100%;
    height: 100%;
    border-radius: inherit;
    object-fit: cover;
  }
  .draft.document {
    height: 3.5rem;
    max-width: 14rem;
    gap: 0.45rem;
    padding: 0 0.85rem 0 0.7rem;
    font-size: 12.5px;
    color: var(--color-ink-2);
  }
  .remove {
    position: absolute;
    top: -0.4rem;
    right: -0.4rem;
    display: flex;
    width: 1.15rem;
    height: 1.15rem;
    align-items: center;
    justify-content: center;
    border: 1px solid var(--color-line-strong);
    border-radius: 9999px;
    background: var(--color-raised);
    color: var(--color-ink-2);
    opacity: 0;
    transition: opacity 150ms;
  }
  .draft:hover .remove,
  .remove:focus-visible {
    opacity: 1;
  }
  .remove:hover {
    color: var(--color-ink);
  }
</style>
