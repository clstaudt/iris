<script lang="ts">
  import LanguageChoice from "./LanguageChoice.svelte";

  // Iris's languages, which every persona's voice is made ready for, with
  // the progress of what that downloads or makes.
</script>

<div class="flex max-w-[980px] flex-col gap-8">
  <section class="flex flex-col gap-3" aria-label="Languages">
    <h3 class="s-section">Languages</h3>
    <LanguageChoice />
  </section>
</div>
