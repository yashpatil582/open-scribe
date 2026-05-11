"""Run open-scribe over a benchmark dataset and compute eval metrics.

Examples:

    # Whole-note F1 on first 3 PriMock57 examples, using gold transcripts (isolates note-gen quality)
    python eval/run_eval.py --dataset primock57 \
        --data-dir datasets/primock57 --limit 3 --from-gold-transcript

    # Full ASR + note-gen on PriMock57 (slow; requires LFS-pulled audio)
    python eval/run_eval.py --dataset primock57 --data-dir datasets/primock57 --limit 3

    # ACI-Bench
    python eval/run_eval.py --dataset acibench --data-dir datasets/aci-bench --limit 5

LLM backend is configured by env vars; see ``src/open_scribe/llm.py``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean

import click
from rich.console import Console
from rich.table import Table

from open_scribe.asr import Transcript, Turn, transcribe
from open_scribe.fhir_emit import soap_to_text
from open_scribe.note_gen import generate_note

from .datasets import Example, load_acibench, load_primock57
from .metrics import section_f1, whole_note_f1

console = Console()

LOADERS = {
    "primock57": load_primock57,
    "acibench": load_acibench,
}


def _build_input_transcript(
    example: Example,
    asr_model: str,
    from_gold_transcript: bool,
) -> Transcript:
    if from_gold_transcript and example.gold_transcript is not None:
        return example.gold_transcript

    if not example.audio_paths:
        # Fall back to a single-blob transcript built from the gold transcript text
        # (e.g. ACI-Bench has no audio at all).
        if example.gold_transcript is not None:
            return example.gold_transcript
        raise RuntimeError(
            f"example {example.id} has no audio and no gold transcript — cannot run"
        )

    # Multi-file audio (PriMock57 has separate doctor + patient WAVs). We
    # transcribe each, label by file, then merge by timestamp.
    turns: list[Turn] = []
    for path in example.audio_paths:
        speaker = "doctor" if "doctor" in path.name else "patient"
        partial = transcribe(path, model_size=asr_model)
        for t in partial.turns:
            turns.append(Turn(speaker=speaker, start=t.start, end=t.end, text=t.text))
    turns.sort(key=lambda t: t.start)
    return Transcript(turns=turns, language="en", duration=max((t.end for t in turns), default=0.0))


@click.command()
@click.option("--dataset", type=click.Choice(list(LOADERS)), required=True)
@click.option("--data-dir", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option("--limit", type=int, default=None, help="Eval only the first N examples")
@click.option("--from-gold-transcript", is_flag=True,
              help="Skip ASR, use the dataset's gold transcript (isolates note-gen quality)")
@click.option("--asr-model", default="base.en", show_default=True)
@click.option("--out", "out_path", type=click.Path(dir_okay=False, path_type=Path), default=None,
              help="Where to write the results JSON (default: eval/results/<dataset>.json)")
@click.option("--resume", is_flag=True,
              help="Skip examples already completed in the output file (merges into it)")
def main(
    dataset: str,
    data_dir: Path,
    limit: int | None,
    from_gold_transcript: bool,
    asr_model: str,
    out_path: Path | None,
    resume: bool,
) -> None:
    loader = LOADERS[dataset]
    examples = list(loader(data_dir, limit=limit))
    if not examples:
        console.print(f"[red]no examples found in {data_dir} for dataset {dataset}[/]")
        return

    out_path = out_path or (Path(__file__).parent / "results" / f"{dataset}.json")

    previous: list[dict] = []
    if resume and out_path.exists():
        prior = json.loads(out_path.read_text())
        previous = [e for e in prior.get("per_example", []) if "whole_note" in e]
        done_ids = {e["id"] for e in previous}
        skipped = len(examples) - len([e for e in examples if e.id not in done_ids])
        examples = [e for e in examples if e.id not in done_ids]
        console.print(f"[dim]resume: {skipped} already done, {len(examples)} remaining[/]")

    console.rule(f"[bold]eval: {dataset} ({len(examples)} examples)")

    per_example = list(previous)
    t0 = time.time()
    for ex in examples:
        try:
            transcript = _build_input_transcript(ex, asr_model, from_gold_transcript)
            note = generate_note(transcript)
            pred_text = soap_to_text(note)
            metrics_whole = whole_note_f1(pred_text, ex.gold_note)
            console.log(
                f"[bold]{ex.id}[/]  whole-note F1 = {metrics_whole['f1']:.3f}  "
                f"(P={metrics_whole['precision']:.3f} R={metrics_whole['recall']:.3f})"
            )
            per_example.append({
                "id": ex.id,
                "whole_note": metrics_whole,
                "predicted": note.model_dump(),
                "predicted_text": pred_text,
                "gold_note": ex.gold_note,
            })
        except Exception as e:
            console.log(f"[red]{ex.id} failed[/]: {e}")
            per_example.append({"id": ex.id, "error": str(e)})

    elapsed = time.time() - t0
    successes = [r for r in per_example if "whole_note" in r]
    aggregate = {
        "n": len(successes),
        "n_failed": len(per_example) - len(successes),
        "whole_note_f1_mean": mean(r["whole_note"]["f1"] for r in successes) if successes else None,
        "whole_note_precision_mean": mean(r["whole_note"]["precision"] for r in successes) if successes else None,
        "whole_note_recall_mean": mean(r["whole_note"]["recall"] for r in successes) if successes else None,
        "elapsed_sec": round(elapsed, 1),
    }

    table = Table(title=f"{dataset} — aggregate")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for k, v in aggregate.items():
        table.add_row(k, f"{v:.3f}" if isinstance(v, float) else str(v))
    console.print(table)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"aggregate": aggregate, "per_example": per_example}, indent=2))
    console.log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
