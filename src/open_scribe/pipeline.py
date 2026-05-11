"""End-to-end pipeline: audio -> FHIR DocumentReference.

Orchestrates ASR -> note generation -> coding -> FHIR emission.
"""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console

from open_scribe.asr import transcribe

console = Console()


@click.command()
@click.option("--audio", "audio_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--out", "out_path", type=click.Path(dir_okay=False), default="out.json", show_default=True)
@click.option("--model", "model_size", default="base.en", show_default=True)
@click.option("--diarize", is_flag=True)
def main(audio_path: str, out_path: str, model_size: str, diarize: bool) -> None:
    """Run the full pipeline. Stages not yet implemented will be skipped with a notice."""

    console.rule("[bold]Stage 1: ASR")
    transcript = transcribe(audio_path, model_size=model_size, diarize=diarize)
    console.log(f"transcribed {len(transcript.turns)} turns ({transcript.duration:.1f}s)")

    console.rule("[bold]Stage 2: Note generation")
    note = None
    try:
        from open_scribe.note_gen import generate_note

        note = generate_note(transcript)
        console.log("SOAP note generated")
        for section in ("subjective", "objective", "assessment", "plan"):
            text = getattr(note, section)
            preview = text[:120] + ("…" if len(text) > 120 else "")
            console.print(f"  [bold]{section.upper():<10}[/] {preview or '[dim](empty)[/]'}")
    except NotImplementedError as e:
        console.log(f"[yellow]skipped[/]: {e}")
    except Exception as e:
        console.log(f"[red]note generation failed[/]: {e}")

    console.rule("[bold]Stage 3: Code suggestion")
    codes = None
    if note is not None:
        try:
            from open_scribe.coder import suggest_codes

            codes = suggest_codes(note)
            console.log(f"suggested {len(codes)} codes")
            for c in codes:
                console.print(
                    f"  [bold]{c.code}[/] ({c.system}, conf={c.confidence:.2f})  "
                    f"{c.description}"
                )
        except NotImplementedError as e:
            console.log(f"[yellow]skipped[/]: {e}")
        except Exception as e:
            console.log(f"[red]code suggestion failed[/]: {e}")

    console.rule("[bold]Stage 4: FHIR emission")
    if note is not None:
        try:
            from datetime import datetime, timezone

            from fhir.resources.R4B.period import Period

            from open_scribe.fhir_emit import to_document_reference

            now = datetime.now(timezone.utc)
            period = Period(
                start=now.replace(microsecond=0).isoformat(),
                end=now.replace(microsecond=0).isoformat(),
            )
            bundle_dict = to_document_reference(note, codes=codes, encounter_period=period)
            Path(out_path).write_text(json.dumps(bundle_dict, indent=2))
            console.log(f"wrote FHIR Bundle to {out_path}")
            return
        except NotImplementedError as e:
            console.log(f"[yellow]skipped[/]: {e}")
        except Exception as e:
            console.log(f"[red]FHIR emission failed[/]: {e}")

    transcript_out = Path(out_path).with_suffix(".transcript.json")
    transcript_out.write_text(json.dumps(
        {
            "language": transcript.language,
            "duration": transcript.duration,
            "turns": [
                {"speaker": t.speaker, "start": t.start, "end": t.end, "text": t.text}
                for t in transcript.turns
            ],
        },
        indent=2,
    ))
    console.log(f"wrote transcript to {transcript_out} (downstream stages not yet implemented)")


if __name__ == "__main__":
    main()
