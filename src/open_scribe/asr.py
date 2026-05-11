"""ASR stage: audio -> transcript with optional speaker diarization.

Uses faster-whisper for transcription. Diarization is optional and requires
the `diarize` extra plus a HuggingFace token with access to
`pyannote/speaker-diarization-3.1`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import click
from rich.console import Console

console = Console()


@dataclass
class Turn:
    speaker: str | None
    start: float
    end: float
    text: str


@dataclass
class Transcript:
    turns: list[Turn] = field(default_factory=list)
    language: str | None = None
    duration: float | None = None

    @property
    def text(self) -> str:
        return "\n".join(
            f"[{t.speaker or 'spk?'}] {t.text}".strip() for t in self.turns
        )


def transcribe(
    audio_path: str | Path,
    model_size: str = "base.en",
    diarize: bool = False,
    hf_token: str | None = None,
) -> Transcript:
    """Transcribe an audio file. Diarization is opt-in.

    `model_size` accepts any faster-whisper model ID (e.g. "tiny.en",
    "base.en", "small.en", "medium.en", "large-v3"). For real eval runs use
    "large-v3"; "base.en" is fine for the day-1 smoke test on CPU.
    """
    from faster_whisper import WhisperModel

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    console.log(f"loading whisper model: {model_size}")
    model = WhisperModel(model_size, device="auto", compute_type="auto")

    console.log(f"transcribing: {audio_path.name}")
    segments_iter, info = model.transcribe(
        str(audio_path),
        beam_size=5,
        vad_filter=True,
        word_timestamps=False,
    )
    segments = list(segments_iter)

    if diarize:
        speaker_labels = _diarize(audio_path, hf_token=hf_token)
        turns = _assign_speakers(segments, speaker_labels)
    else:
        turns = [
            Turn(speaker=None, start=s.start, end=s.end, text=s.text.strip())
            for s in segments
        ]

    return Transcript(turns=turns, language=info.language, duration=info.duration)


def _diarize(audio_path: Path, hf_token: str | None) -> list[tuple[float, float, str]]:
    """Run pyannote speaker diarization. Returns list of (start, end, speaker_label)."""
    try:
        from pyannote.audio import Pipeline
    except ImportError as e:
        raise RuntimeError(
            "diarization requires the 'diarize' extra: pip install -e '.[diarize]'"
        ) from e

    token = hf_token or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "diarization requires HF_TOKEN env var (and accepting "
            "pyannote/speaker-diarization-3.1 license on HuggingFace)"
        )

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token,
    )
    diarization = pipeline(str(audio_path))
    return [
        (turn.start, turn.end, label)
        for turn, _, label in diarization.itertracks(yield_label=True)
    ]


def _assign_speakers(whisper_segments, speaker_labels) -> list[Turn]:
    """Assign each whisper segment a speaker by max temporal overlap."""
    turns: list[Turn] = []
    for seg in whisper_segments:
        best_label = None
        best_overlap = 0.0
        for s_start, s_end, label in speaker_labels:
            overlap = max(0.0, min(seg.end, s_end) - max(seg.start, s_start))
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = label
        turns.append(
            Turn(speaker=best_label, start=seg.start, end=seg.end, text=seg.text.strip())
        )
    return turns


@click.command()
@click.argument("audio_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--model", "model_size", default="base.en", show_default=True,
              help="faster-whisper model id (tiny.en/base.en/small.en/medium.en/large-v3)")
@click.option("--diarize", is_flag=True, help="run pyannote speaker diarization")
@click.option("--out", "out_path", type=click.Path(dir_okay=False), default=None,
              help="optional path to write the transcript JSON")
def main(audio_path: str, model_size: str, diarize: bool, out_path: str | None) -> None:
    """Transcribe an audio file and print the transcript."""
    import json

    transcript = transcribe(audio_path, model_size=model_size, diarize=diarize)

    console.print(
        f"\n[bold]Language[/]: {transcript.language}  "
        f"[bold]Duration[/]: {transcript.duration:.1f}s  "
        f"[bold]Turns[/]: {len(transcript.turns)}\n"
    )
    for t in transcript.turns:
        spk = f"[{t.speaker}] " if t.speaker else ""
        console.print(f"{t.start:6.2f}-{t.end:6.2f}  {spk}{t.text}")

    if out_path:
        payload = {
            "language": transcript.language,
            "duration": transcript.duration,
            "turns": [
                {"speaker": t.speaker, "start": t.start, "end": t.end, "text": t.text}
                for t in transcript.turns
            ],
        }
        Path(out_path).write_text(json.dumps(payload, indent=2))
        console.log(f"wrote {out_path}")


if __name__ == "__main__":
    main()
