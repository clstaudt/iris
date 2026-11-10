"""Benchmark kokoro-onnx vs kokoro-cli for TTS performance."""

import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import ollama
import pandas as pd
import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from scipy import stats
from typing_extensions import Annotated

from iris.tts import KokoroGenerator

app = typer.Typer(help="Benchmark kokoro-onnx vs kokoro-cli for TTS performance")
console = Console()


def generate_test_texts(
    target_word_counts: list[int] | None = None,
    model: str = "qwen3:1.7b",
) -> list[str]:
    """Generate test texts using Ollama LLM."""
    if target_word_counts is None:
        target_word_counts = [5, 10, 30, 60]

    console.print(f"\n[dim]Generating test texts using {model}...[/dim]")

    # Fallback texts in case generation fails
    fallbacks = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "In the heart of a bustling city, where skyscrapers touched the clouds, "
        "there lived a small bookstore that time seemed to have forgotten.",
        "Artificial intelligence is transforming how we interact with technology. "
        "Machine learning algorithms can recognize patterns, make predictions, and "
        "generate creative content with remarkable accuracy and speed.",
    ]

    texts = []

    def get_fallback(index: int) -> str:
        return fallbacks[min(index, len(fallbacks) - 1)]

    for i, target_words in enumerate(target_word_counts):
        try:
            response = ollama.generate(
                model=model,
                prompt=f"Generate exactly {target_words} words of natural English text.",
                options={"temperature": 0.7},
            )

            text = response["response"].strip()

            # If empty or too short, use fallback
            if not text or len(text) < 10:
                console.print(f"  [yellow]• Generated text too short, using fallback[/yellow]")
                text = get_fallback(i)

            word_count = len(text.split())
            texts.append(text)
            console.print(f"  [dim]• Generated: {word_count} words[/dim]")

        except Exception as e:
            console.print(f"  [yellow]• Warning: Generation failed ({e}), using fallback[/yellow]")
            texts.append(get_fallback(i))

    return texts


def benchmark_onnx(text: str, generator: KokoroGenerator) -> float:
    """Benchmark kokoro-onnx using existing KokoroGenerator."""
    start = time.perf_counter()

    # Use the existing generate method
    audio_file = generator.generate(text)

    # Clean up temp file
    if audio_file:
        Path(audio_file).unlink(missing_ok=True)

    return time.perf_counter() - start


def benchmark_cli(
    text: str,
    cli_path: str,
    voice: str = "af_bella",
    model_path: Path | None = None,
    voices_path: Path | None = None,
) -> float:
    """Benchmark kokoro-cli tool."""
    start = time.perf_counter()

    # Create temp files for input text and output audio
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as text_file:
        text_file.write(text)
        text_file_path = text_file.name

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_file:
        audio_file_path = audio_file.name

    try:
        # Build command with model paths if provided
        cmd = [cli_path, text_file_path, audio_file_path, "--voice", voice]
        if model_path:
            cmd.extend(["--model", str(model_path)])
        if voices_path:
            cmd.extend(["--voices", str(voices_path)])

        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=30,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"kokoro-tts CLI failed with exit code {result.returncode}\n"
                f"stderr: {result.stderr}\n"
                f"stdout: {result.stdout}"
            )
    finally:
        # Clean up temp files
        Path(text_file_path).unlink(missing_ok=True)
        Path(audio_file_path).unlink(missing_ok=True)

    return time.perf_counter() - start


def find_cli_tool() -> str | None:
    """Find kokoro-tts CLI in PATH or uv tools directory."""
    if cli_path := shutil.which("kokoro-tts"):
        return cli_path

    # Check uv tools directory
    uv_tools = Path.home() / ".local" / "bin" / "kokoro-tts"
    return str(uv_tools) if uv_tools.exists() else None


def run_benchmarks(
    texts: list[str],
    iterations: int,
    warmup: int,
    voice: str,
    speed: float,
    onnx_only: bool = False,
    cli_only: bool = False,
) -> pd.DataFrame:
    """Run benchmarks and return results as DataFrame."""
    generator = None
    cli_path = None
    model_path = None
    voices_path = None

    # Setup ONNX
    if not cli_only:
        generator = KokoroGenerator(voice=voice, speed=speed)
        model_path = generator.model_path
        voices_path = generator.voices_path
        console.print("[dim]Warming up ONNX...[/dim]")
        for _ in range(warmup):
            benchmark_onnx(texts[0], generator)

    # Setup CLI
    if not onnx_only:
        cli_path = find_cli_tool()
        if not cli_path:
            if cli_only:
                raise RuntimeError(
                    "kokoro-tts CLI not found but --cli-only was specified.\n"
                    "Install with: uv tool install kokoro-tts"
                )
        elif cli_path:
            # If we don't have model paths yet (cli_only mode), get them
            if model_path is None:
                temp_gen = KokoroGenerator(voice=voice, speed=speed)
                model_path = temp_gen.model_path
                voices_path = temp_gen.voices_path

            console.print("[dim]Warming up CLI...[/dim]")
            for _ in range(warmup):
                benchmark_cli(texts[0], cli_path, voice, model_path, voices_path)

    # Build method list
    methods = []
    if not cli_only and generator:
        methods.append(("onnx", lambda t: benchmark_onnx(t, generator)))
    if not onnx_only and cli_path:
        methods.append(
            ("cli", lambda t: benchmark_cli(t, cli_path, voice, model_path, voices_path))
        )

    # Run benchmarks with progress tracking
    results = []
    total = len(texts) * len(methods) * iterations

    console.print()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("Benchmarking...", total=total)

        for text_idx, text in enumerate(texts):
            text_preview = f"{text[:50]}..." if len(text) > 50 else text
            word_count = len(text.split())
            for method_name, bench_func in methods:
                for i in range(iterations):
                    results.append(
                        {
                            "method": method_name,
                            "text_id": text_idx,
                            "text": text_preview,
                            "word_count": word_count,
                            "iteration": i,
                            "time_sec": bench_func(text),
                        }
                    )
                    progress.update(
                        task,
                        advance=1,
                        description=f"[cyan]{method_name}[/cyan] | text {text_idx + 1} ({word_count} words) | {i + 1}/{iterations}",
                    )

    return pd.DataFrame(results)


def analyze_results(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Analyze benchmark results."""
    # Summary statistics - group by method and text_id only (word_count is redundant)
    summary = (
        df.groupby(["method", "text_id"])["time_sec"]
        .agg(
            [
                ("n", "count"),
                ("mean", "mean"),
                ("median", "median"),
                ("std", "std"),
                ("min", "min"),
                ("max", "max"),
            ]
        )
        .reset_index()
    )

    # Add word_count from the first occurrence
    word_counts = df.groupby("text_id")["word_count"].first().reset_index()
    summary = summary.merge(word_counts, on="text_id")

    # Add confidence intervals
    def ci_95(x):
        n = len(x)
        sem = stats.sem(x)
        ci = stats.t.interval(0.95, n - 1, loc=x.mean(), scale=sem)
        return pd.Series({"ci_lower": ci[0], "ci_upper": ci[1]})

    ci_df = df.groupby(["method", "text_id"])["time_sec"].apply(ci_95).reset_index()
    summary = summary.merge(ci_df, on=["method", "text_id"])

    # Comparisons
    comparisons = []
    for text_id in df["text_id"].unique():
        subset = df[df["text_id"] == text_id]
        methods = subset["method"].unique()
        word_count = subset["word_count"].iloc[0]

        if len(methods) == 2:
            times_a = subset[subset["method"] == methods[0]]["time_sec"].values
            times_b = subset[subset["method"] == methods[1]]["time_sec"].values

            _, p_value = stats.mannwhitneyu(times_a, times_b, alternative="two-sided")

            mean_a = times_a.mean()
            mean_b = times_b.mean()
            speedup = mean_b / mean_a

            pooled_std = np.sqrt((times_a.var() + times_b.var()) / 2)
            cohens_d = abs(mean_a - mean_b) / pooled_std if pooled_std > 0 else 0

            comparisons.append(
                {
                    "text_id": text_id,
                    "word_count": word_count,
                    "method_a": methods[0],
                    "method_b": methods[1],
                    "speedup": speedup,
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                    "cohens_d": cohens_d,
                }
            )

    return summary, pd.DataFrame(comparisons)


def format_time(sec: float) -> str:
    """Format time with appropriate units."""
    if sec < 0.001:
        return f"{sec * 1_000_000:.1f} µs"
    elif sec < 1:
        return f"{sec * 1000:.1f} ms"
    else:
        return f"{sec:.3f} s"


def print_results(summary: pd.DataFrame, comparisons: pd.DataFrame, texts: list[str]):
    """Print benchmark results."""
    console.print()
    console.print("[bold cyan]" + "═" * 70 + "[/bold cyan]")
    console.print("[bold cyan]  Benchmark Results[/bold cyan]")
    console.print("[bold cyan]" + "═" * 70 + "[/bold cyan]")

    # Show test texts
    console.print("\n[bold]Test Texts:[/bold]")
    for i, text in enumerate(texts, 1):
        preview = f"{text[:60]}..." if len(text) > 60 else text
        word_count = len(text.split())
        console.print(f"  {i}. [{word_count} words] {preview}")

    # Summary table
    table = Table(title="\n📊 Summary Statistics", show_header=True, header_style="bold cyan")
    table.add_column("Method", style="cyan")
    table.add_column("Text", justify="right")
    table.add_column("Words", justify="right")
    table.add_column("N", justify="right")
    table.add_column("Mean", justify="right")
    table.add_column("Median", justify="right")
    table.add_column("Std", justify="right")

    for _, row in summary.iterrows():
        table.add_row(
            row["method"],
            str(row["text_id"] + 1),
            str(row["word_count"]),
            str(row["n"]),
            format_time(row["mean"]),
            format_time(row["median"]),
            format_time(row["std"]),
        )

    console.print()
    console.print(table)

    # Comparisons table
    if not comparisons.empty:
        table = Table(
            title="\n⚔️  Performance Comparison",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Text", justify="right")
        table.add_column("Words", justify="right")
        table.add_column("Winner", style="bold green")
        table.add_column("Speedup", justify="right")
        table.add_column("P-value", justify="right")
        table.add_column("Sig", justify="center")

        for _, row in comparisons.iterrows():
            winner = row["method_a"] if row["speedup"] > 1 else row["method_b"]
            speedup_val = row["speedup"] if row["speedup"] > 1 else 1 / row["speedup"]
            sig_icon = "✓" if row["significant"] else "✗"
            sig_style = "green" if row["significant"] else "dim"

            table.add_row(
                str(row["text_id"] + 1),
                str(row["word_count"]),
                winner,
                f"{speedup_val:.2f}×",
                f"{row['p_value']:.4f}",
                f"[{sig_style}]{sig_icon}[/{sig_style}]",
            )

        console.print()
        console.print(table)

        # Conclusion
        sig_comparisons = comparisons[comparisons["significant"]]
        if not sig_comparisons.empty:
            onnx_wins = (sig_comparisons["speedup"] > 1).sum()
            cli_wins = (sig_comparisons["speedup"] < 1).sum()

            console.print()
            console.print("[bold cyan]🏆 Conclusion:[/bold cyan]")
            if onnx_wins > cli_wins:
                avg_speedup = sig_comparisons[sig_comparisons["speedup"] > 1]["speedup"].mean()
                console.print(
                    f"  • [bold green]onnx[/bold green] is faster (significant in {onnx_wins}/{len(comparisons)} cases)"
                )
                console.print(f"  • Average speedup: [bold]{avg_speedup:.2f}×[/bold]")
            elif cli_wins > onnx_wins:
                avg_speedup = (
                    1 / sig_comparisons[sig_comparisons["speedup"] < 1]["speedup"]
                ).mean()
                console.print(
                    f"  • [bold green]cli[/bold green] is faster (significant in {cli_wins}/{len(comparisons)} cases)"
                )
                console.print(f"  • Average speedup: [bold]{avg_speedup:.2f}×[/bold]")
            else:
                console.print("  • [yellow]Performance is comparable[/yellow]")
        else:
            console.print()
            console.print("[bold cyan]🏆 Conclusion:[/bold cyan]")
            console.print("  • [yellow]No statistically significant differences found[/yellow]")

    console.print()
    console.print("[bold cyan]" + "═" * 70 + "[/bold cyan]")
    console.print()


@app.command()
def main(
    iterations: Annotated[
        int, typer.Option("-n", "--iterations", help="Number of iterations per test")
    ] = 10,
    warmup: Annotated[int, typer.Option("-w", "--warmup", help="Number of warmup iterations")] = 2,
    texts: Annotated[
        list[str],
        typer.Option("--text", help="Custom test text (can be used multiple times)"),
    ] = None,
    word_counts: Annotated[
        list[int],
        typer.Option("--words", help="Target word counts for generated texts"),
    ] = None,
    llm: Annotated[str, typer.Option(help="Ollama model for text generation")] = "qwen3:1.7b",
    voice: Annotated[str, typer.Option(help="Voice to use for TTS")] = "af_bella",
    speed: Annotated[float, typer.Option(help="Speech speed multiplier")] = 1.0,
    onnx_only: Annotated[
        bool, typer.Option(help="Benchmark only kokoro-onnx (skip comparison)")
    ] = False,
    cli_only: Annotated[bool, typer.Option(help="Benchmark only kokoro-cli")] = False,
    csv: Annotated[Path, typer.Option(help="Save results to CSV file")] = None,
):
    """
    Comparative benchmark: kokoro-onnx vs kokoro-cli for TTS performance.

    This tool compares the Python library (kokoro-onnx) against the CLI tool
    to determine which is faster for speech generation.

    By default, generates test texts using an LLM at various word counts.
    Use --text multiple times to provide custom texts instead.
    """
    console.print()
    console.print("[bold cyan]═" * 70 + "[/bold cyan]")
    console.print("[bold cyan]  Kokoro TTS Performance Benchmark[/bold cyan]")
    console.print("[bold cyan]═" * 70 + "[/bold cyan]")

    # Check for CLI tool FIRST, before doing any work
    if not onnx_only and not cli_only:
        if not find_cli_tool():
            console.print()
            console.print("[bold yellow]⚠ Warning: kokoro-tts CLI not found![/bold yellow]")
            console.print()
            console.print("This benchmark compares kokoro-onnx vs kokoro-cli performance.")
            console.print("Without the CLI tool, only ONNX can be benchmarked.")
            console.print()
            console.print("To install the CLI tool:")
            console.print("  [cyan]uv tool install kokoro-tts[/cyan]")
            console.print()
            console.print("Or run with [cyan]--onnx-only[/cyan] to skip this check.")
            console.print()

            try:
                response = (
                    console.input("[bold]Continue with ONNX-only benchmark? [y/N]: [/bold]")
                    .strip()
                    .lower()
                )
                if response not in ("y", "yes"):
                    console.print("[dim]Benchmark cancelled.[/dim]")
                    sys.exit(0)
                onnx_only = True
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Benchmark cancelled.[/dim]")
                sys.exit(0)

    # Generate or use provided texts
    test_texts = list(texts) if texts else generate_test_texts(word_counts, llm)
    if texts:
        console.print(f"\n[dim]Using {len(test_texts)} provided texts[/dim]")

    console.print(
        f"[dim]Iterations: {iterations}, Warmup: {warmup}, Voice: {voice}, Speed: {speed}[/dim]"
    )

    # Run benchmarks
    df = run_benchmarks(
        texts=test_texts,
        iterations=iterations,
        warmup=warmup,
        voice=voice,
        speed=speed,
        onnx_only=onnx_only,
        cli_only=cli_only,
    )

    # Analyze
    summary, comparisons = analyze_results(df)

    # Display
    print_results(summary, comparisons, test_texts)

    # Save if requested
    if csv:
        df.to_csv(csv, index=False)
        console.print(f"[dim]Raw results saved to {csv}[/dim]\n")


if __name__ == "__main__":
    app()
