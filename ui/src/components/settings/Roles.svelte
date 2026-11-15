<script lang="ts">
  import { onMount } from "svelte";
  import { store, type RoleInfo } from "../../lib/state.svelte";
  import Icon from "../Icon.svelte";

  // A role is a focus a thread can be given: a name and instructions that
  // follow the persona's in the prompt. A thread keeps the role it was
  // started with; an edited focus applies from its next turn.

  let draft = $state<RoleInfo | null>(null);
  let saved = $state("");
  let error = $state("");
  let saving = $state(false);
  let confirmDelete = $state(false);

  const editable = (role: RoleInfo) => JSON.stringify({ name: role.name, focus: role.focus });

  onMount(() => {
    store.loadRoles().catch(() => {});
  });

  // The role chosen, as saved, or a new one; at first the first role
  $effect(() => {
    if (store.editingRole === null && store.roles.length) store.editingRole = store.roles[0].id!;
    const id = store.editingRole ?? "";
    const role = id ? store.roles.find((r) => r.id === id) : null;
    if (id && !role) return; // the list is still on its way
    if (draft && (draft.id ?? "") === id) return;
    draft = role ? { ...role } : { name: "", focus: "" };
    saved = editable(draft);
    error = "";
    confirmDelete = false;
  });

  const isNew = $derived(!draft?.id);
  const changed = $derived(draft !== null && editable(draft) !== saved);
  const canSave = $derived(!!draft && (changed || isNew) && !!draft.name.trim() && !saving);

  async function save() {
    if (!draft || !canSave) return;
    saving = true;
    error = "";
    try {
      const failed = await store.saveRole($state.snapshot(draft));
      if (failed) {
        error = failed;
      } else {
        const role = store.roles.find((r) => r.id === store.editingRole);
        draft = role ? { ...role } : draft;
        saved = editable(draft);
      }
    } catch (err) {
      error = String(err);
    } finally {
      saving = false;
    }
  }

  function revert() {
    if (!draft) return;
    const role = draft.id ? store.roles.find((r) => r.id === draft!.id) : null;
    if (!role) {
      store.editingRole = store.roles[0]?.id ?? "";
      return;
    }
    draft = { ...role };
  }

  async function remove() {
    if (!draft?.id) return;
    if (!confirmDelete) {
      confirmDelete = true;
      return;
    }
    const id = draft.id;
    await store.deleteRole(id);
    // A shipped role is as shipped again; the user's own is gone
    const role = store.roles.find((r) => r.id === id);
    draft = null;
    store.editingRole = role?.id ?? store.roles[0]?.id ?? "";
  }
</script>

<div class="grid max-w-[980px] grid-cols-[260px_minmax(0,1fr)] items-start gap-6">
  <div class="flex flex-col gap-1" data-tour="roles">
    {#each store.roles as role (role.id)}
      <button
        onclick={() => (store.editingRole = role.id!)}
        aria-current={store.editingRole === role.id ? "true" : undefined}
        class="role {store.editingRole === role.id ? 'active' : ''}"
      >
        <span class="name">{role.name}</span>
        <span class="focus">{role.focus}</span>
      </button>
    {/each}
    <button
      onclick={() => (store.editingRole = "")}
      class="s-btn mt-1 self-start {store.editingRole === '' ? 'text-ink' : ''}"
    >
      <Icon name="plus" size={13} />
      New role
    </button>
  </div>

  {#if draft}
    <div class="flex min-w-0 flex-col gap-6">
      <section class="flex flex-col gap-2">
        <label class="s-section" for="role-name">Name</label>
        <input id="role-name" bind:value={draft.name} spellcheck="false" class="s-field" />
      </section>
      <section class="flex flex-col gap-2">
        <label class="s-section" for="role-focus">Focus</label>
        <textarea id="role-focus" bind:value={draft.focus} spellcheck="false" rows="7" class="s-field"
        ></textarea>
      </section>
      <div class="flex items-center justify-between gap-2 border-t border-line pt-4">
        <div class="min-w-0">
          {#if error}
            <p class="s-status error truncate" title={error}>{error}</p>
          {:else if draft.id && (!draft.shipped || draft.edited)}
            <button class="s-btn danger" onclick={remove} onblur={() => (confirmDelete = false)}>
              {#if draft.shipped}
                {confirmDelete ? "Confirm reset" : "Reset"}
              {:else}
                {confirmDelete ? "Confirm delete" : "Delete"}
              {/if}
            </button>
          {/if}
        </div>
        <div class="flex shrink-0 gap-1.5">
          <button class="s-btn" disabled={!changed && !isNew} onclick={revert}>Cancel</button>
          <button class="s-btn primary" disabled={!canSave} onclick={save}>
            {saving ? "Saving" : "Save"}
          </button>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .role {
    display: flex;
    width: 100%;
    flex-direction: column;
    gap: 3px;
    padding: 0.6rem 0.75rem;
    border-radius: 0.65rem;
    text-align: left;
    transition: background-color 150ms;
  }
  .role:hover {
    background: color-mix(in srgb, var(--color-hover) 60%, transparent);
  }
  .role.active {
    background: var(--color-hover);
  }
  .name {
    font-size: 13px;
    color: var(--color-ink);
  }
  .focus {
    display: -webkit-box;
    overflow: hidden;
    -webkit-box-orient: vertical;
    -webkit-line-clamp: 2;
    line-clamp: 2;
    font-size: 12px;
    line-height: 1.4;
    color: var(--color-ink-3);
  }
</style>
