<script lang="ts">
  interface Props {
    checked: boolean;
    disabled?: boolean;
    label: string;
    onchange: (checked: boolean) => void;
  }

  let { checked, disabled = false, label, onchange }: Props = $props();
</script>

<button
  role="switch"
  aria-checked={checked}
  aria-label={label}
  {disabled}
  onclick={() => onchange(!checked)}
  class="relative h-[18px] w-8 shrink-0 rounded-full transition-colors duration-200 disabled:opacity-40 {checked
    ? 'bg-accent'
    : 'bg-hover'}"
>
  <span
    class="absolute left-[2px] top-[2px] h-[14px] w-[14px] rounded-full bg-ink shadow transition-transform duration-200 {checked
      ? 'translate-x-[14px]'
      : ''}"
  ></span>
</button>
