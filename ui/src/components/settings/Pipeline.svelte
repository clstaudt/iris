<script lang="ts">
  import { onMount } from "svelte";
  import { store } from "../../lib/state.svelte";
  import Icon from "../Icon.svelte";

  /** One stage of the chain, or something the language model draws on
   *  (a branch). */
  interface Stage {
    id: string;
    icon: string;
    label: string;
    value: string;
    meta?: string;
    off?: boolean;
    branch?: boolean;
  }

  interface Row {
    key: string;
    color: string;
    head?: string;
    stage?: Stage;
    /** The last of a run of branches */
    end?: boolean;
  }

  onMount(() => {
    if (!store.voices) store.loadVoices().catch(() => {});
  });

  const config = $derived(store.config);

  const title = (text: string) => text.charAt(0).toUpperCase() + text.slice(1);

  const voiceName = $derived(
    store.voices?.engines[config?.tts_engine ?? ""]?.find((v) => v.id === config?.tts_voice)
      ?.name ??
      config?.tts_voice ??
      ""
  );
  const speechEngine = $derived(
    store.voices?.engine_info?.find((e) => e.id === config?.tts_engine)?.name ??
      config?.engines?.tts?.find((e) => e.id === config?.tts_engine)?.label ??
      title(config?.tts_engine ?? "")
  );

  /** "Whisper large-v3-turbo · MLX", a model folder, or another engine. */
  const recognition = $derived.by(() => {
    if (!config) return { value: "", meta: "" };
    const engine = config.recognition_engine ?? "whisper";
    if (engine !== "whisper") {
      const label = config.engines?.stt?.find((e) => e.id === engine)?.label ?? title(engine);
      return { value: label, meta: "" };
    }
    const backend =
      config.whisper_engine === "mlx" ? "MLX" : `faster-whisper · ${config.whisper_device}`;
    if (config.stt_model_path) {
      const folder = config.stt_model_path.split(/[\\/]/).filter(Boolean).pop() ?? "";
      return { value: folder, meta: backend };
    }
    return { value: `Whisper ${config.whisper_model}`, meta: backend };
  });

  /** The server's address without its scheme. */
  const server = $derived.by(() => {
    try {
      return new URL(config?.ollama_url ?? "").host;
    } catch {
      return config?.ollama_url ?? "";
    }
  });

  const toolsOn = $derived(
    Object.values(config?.tools ?? {})
      .filter((t) => t.enabled && t.available)
      .map((t) => t.name ?? "")
  );

  const sections = $derived.by(() => {
    if (!config) return [];
    return [
      {
        id: "hear",
        label: "Hear",
        color: "var(--color-petal-cyan)",
        stages: [
          { id: "mic", icon: "mic", label: "Microphone", value: config.input_device },
          {
            id: "wake",
            icon: "wave",
            label: "Wake word",
            value: config.wake_word || "Off",
            off: !config.wake_word,
          },
          {
            id: "turn",
            icon: "timer",
            label: "Turn end",
            value:
              config.turn_detection === "silence"
                ? `Silence of ${config.silence_duration ?? 3.5} s`
                : "Smart Turn",
          },
          {
            id: "stt",
            icon: "ear",
            label: "Recognition",
            value: recognition.value,
            meta: recognition.meta,
          },
        ],
      },
      {
        id: "think",
        label: "Think",
        color: "var(--color-accent)",
        stages: [
          { id: "llm", icon: "cpu", label: "Language model", value: config.llm_model, meta: server },
          { id: "persona", icon: "user", label: "Persona", value: config.persona, branch: true },
          {
            id: "tools",
            icon: "wrench",
            label: "Tools",
            value: toolsOn.join(", ") || "None",
            off: toolsOn.length === 0,
            branch: true,
          },
          {
            id: "memory",
            icon: "brain",
            label: "Memory",
            value: !config.memory ? "Off" : config.amnesia ? "Paused" : "On",
            off: !config.memory || config.amnesia,
            branch: true,
          },
        ],
      },
      {
        id: "speak",
        label: "Speak",
        color: "var(--color-petal-warm)",
        stages: [
          {
            id: "voice",
            icon: "message",
            label: "Voice",
            value: `${speechEngine} · ${voiceName}`,
            meta: config.persona,
          },
          {
            id: "output",
            icon: "volume",
            label: "Output",
            value: store.speakReplies ? "Spoken" : "Text only",
            meta:
              store.speakReplies && config.barge_in && config.barge_in_available
                ? "interrupt by voice"
                : undefined,
            off: !store.speakReplies,
          },
        ],
      },
    ] as { id: string; label: string; color: string; stages: Stage[] }[];
  });

  const rows = $derived<Row[]>(
    sections.flatMap((section) => [
      { key: section.id, color: section.color, head: section.label },
      ...section.stages.map((stage, i, list) => ({
        key: stage.id,
        color: section.color,
        stage,
        end: stage.branch && !list[i + 1]?.branch,
      })),
    ])
  );

  function open(id: string) {
    store.stage = store.stage === id ? null : id;
  }
</script>

{#if !config}
  <p class="s-status">Loading models</p>
{:else}
  <div class="chain">
    {#each rows as row, i (row.key)}
      <div class="row">
        <span
          class="rail {i === 0 ? 'first' : ''} {i === rows.length - 1 ? 'last' : ''}"
          style="--c: {row.color}"
        >
          {#if row.head}
            <span class="dot sect"></span>
          {:else if !row.stage?.branch}
            <span class="dot"></span>
          {/if}
        </span>
        {#if row.head}
          <h3 class="s-section head">{row.head}</h3>
        {:else if row.stage}
          <button
            onclick={() => open(row.stage!.id)}
            aria-expanded={store.stage === row.stage.id}
            class="stage {row.stage.branch ? 'branch' : ''} {row.end ? 'end' : ''} {row.stage.off
              ? 'off'
              : ''} {store.stage === row.stage.id ? 'active' : ''}"
          >
            <Icon name={row.stage.icon} size={row.stage.branch ? 14 : 16} />
            <span class="label">{row.stage.label}</span>
            <span class="value">
              {row.stage.value}{#if row.stage.meta}<span class="meta">&nbsp;· {row.stage.meta}</span>{/if}
            </span>
          </button>
        {/if}
      </div>
    {/each}
  </div>
{/if}

<style>
  .chain {
    display: flex;
    max-width: 760px;
    flex-direction: column;
  }
  .row {
    display: grid;
    grid-template-columns: 30px minmax(0, 1fr);
  }
  .rail {
    position: relative;
  }
  .rail::before {
    content: "";
    position: absolute;
    left: 14px;
    top: 0;
    bottom: 0;
    width: 2px;
    background: var(--c);
    opacity: 0.4;
  }
  .rail.first::before {
    top: 50%;
  }
  .rail.last::before {
    bottom: 50%;
  }
  .dot {
    position: absolute;
    left: 9px;
    top: 50%;
    width: 12px;
    height: 12px;
    margin-top: -6px;
    border-radius: 9999px;
    border: 2px solid var(--c);
    background: var(--color-panel);
  }
  .dot.sect {
    left: 10px;
    width: 10px;
    height: 10px;
    margin-top: -5px;
    border: 0;
    background: var(--c);
  }
  .head {
    display: flex;
    min-height: 2.5rem;
    align-items: center;
    padding-left: 0.75rem;
  }
  .stage {
    position: relative;
    display: grid;
    width: 100%;
    min-height: 3.1rem;
    grid-template-columns: 18px 112px minmax(0, 1fr);
    align-items: center;
    gap: 0.75rem;
    margin: 1px 0;
    padding: 0 0.75rem;
    border-radius: 0.6rem;
    border: 1px solid transparent;
    color: var(--color-ink-2);
    text-align: left;
    transition:
      background-color 150ms,
      border-color 150ms;
  }
  .stage:hover {
    background: var(--color-raised);
  }
  .stage.active {
    border-color: color-mix(in srgb, var(--color-accent) 50%, transparent);
    background: var(--color-raised);
  }
  .label {
    font-size: 12.5px;
    color: var(--color-ink-2);
    white-space: nowrap;
  }
  .value {
    overflow: hidden;
    font-size: 13px;
    color: var(--color-ink);
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .meta {
    font-size: 12px;
    color: var(--color-ink-3);
  }
  .stage.off .value {
    color: var(--color-ink-3);
  }
  /* What the language model draws on, branching off the line */
  .stage.branch {
    min-height: 2.4rem;
    grid-template-columns: 16px 94px minmax(0, 1fr);
    padding-left: 2.5rem;
  }
  .stage.branch .value {
    font-size: 12.5px;
    color: var(--color-ink-2);
  }
  .stage.branch::before {
    content: "";
    position: absolute;
    left: 1.25rem;
    top: -2px;
    height: 50%;
    width: 0.75rem;
    border-left: 1.5px solid var(--color-line-strong);
    border-bottom: 1.5px solid var(--color-line-strong);
    border-bottom-left-radius: 7px;
  }
  .stage.branch:not(.end)::after {
    content: "";
    position: absolute;
    left: 1.25rem;
    top: 0;
    bottom: 0;
    border-left: 1.5px solid var(--color-line-strong);
  }
</style>
