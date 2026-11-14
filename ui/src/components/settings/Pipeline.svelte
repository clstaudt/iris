<script lang="ts">
  import { onMount } from "svelte";
  import { pipelineSection, type PipelineStage } from "../../lib/pipeline";
  import { store, type TurnTiming } from "../../lib/state.svelte";
  import PipelineChain from "../PipelineChain.svelte";

  type Timed = keyof TurnTiming["stages"];

  /** A stage, and the part of a turn's time it takes */
  type Stage = PipelineStage & { timed?: Timed };

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

  /** The persona of the thread governs the words and the voice */
  const personaTag = $derived({
    persona: config?.persona ?? "",
    open: () => store.editPersona(store.persona),
  });

  /** The thread governs the persona: it is shown in the conversation */
  function toConversation() {
    store.openSettings(null);
  }

  const sections = $derived.by(() => {
    if (!config) return [];
    return [
      pipelineSection<Stage>("hear", [
        { id: "mic", icon: "mic", label: "Microphone", value: config.input_device },
        {
          id: "wake",
          icon: "wave",
          label: "Wake word",
          value: config.wake_word || "Off",
          off: !config.wake_word,
          tag: personaTag,
        },
        {
          id: "turn",
          icon: "timer",
          label: "Turn end",
          timed: "turn_end",
          value:
            config.turn_detection === "silence"
              ? `Silence of ${config.silence_duration ?? 3.5} s`
              : "Smart Turn",
        },
        {
          id: "stt",
          icon: "ear",
          label: "Recognition",
          timed: "recognition",
          value: recognition.value,
          meta: recognition.meta,
        },
      ]),
      pipelineSection<Stage>("think", [
        {
          id: "llm",
          icon: "cpu",
          label: "Language model",
          value: config.llm_model,
          meta: server,
          timed: "reply",
        },
        {
          id: "persona",
          icon: "user",
          label: "Persona",
          value: config.persona,
          branch: true,
          tag: { thread: store.thread?.title || "New thread", open: toConversation },
        },
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
      ]),
      pipelineSection<Stage>("speak", [
        {
          id: "voice",
          icon: "message",
          label: "Voice",
          timed: "voice",
          value: `${speechEngine} · ${voiceName}`,
          tag: personaTag,
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
      ]),
    ];
  });

  /** What runs each timed stage now, as the core names it in a timing. */
  const components = $derived<Record<Timed, string>>({
    turn_end: config?.turn_detection ?? "smart_turn",
    recognition: `${config?.recognition_engine ?? "whisper"}:${config?.whisper_model ?? ""}`,
    reply: config?.llm_model ?? "",
    voice: `${config?.tts_engine ?? ""}:${config?.tts_voice ?? ""}`,
  });

  /** A stage's latest measurement taken with what runs it now; a model
   *  swapped since is not measured yet. */
  function measured(stage: Timed): number | null {
    for (let i = store.timings.length - 1; i >= 0; i--) {
      const timing = store.timings[i];
      const ms = timing.stages[stage];
      if (ms !== null && timing.components[stage] === components[stage]) return ms;
    }
    return null;
  }

  /** Each stage with its latest measurement, if it takes part of a turn's time */
  const timed = $derived(
    sections.map((section) => ({
      ...section,
      stages: section.stages.map(({ timed, ...stage }) =>
        timed ? { ...stage, ms: measured(timed) } : stage
      ),
    }))
  );

  function open(id: string) {
    if (id === "persona") return toConversation();
    store.stage = store.stage === id ? null : id;
  }
</script>

{#if !config}
  <p class="s-status">Loading models</p>
{:else}
  <PipelineChain sections={timed} timings active={store.stage} onselect={open} />
{/if}
