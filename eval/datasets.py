"""Dataset loaders for medical-scribe benchmarks.

Each loader yields ``Example`` records with optional audio path, gold transcript,
and gold note text. Notes are returned as plain text — datasets vary in whether
they structure notes as SOAP (ACI-Bench) or free-form (PriMock57).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from open_scribe.asr import Transcript, Turn


@dataclass
class Example:
    id: str
    gold_note: str
    gold_transcript: Transcript | None = None
    audio_paths: list[Path] | None = None


def parse_textgrid(path: Path, speaker: str) -> list[Turn]:
    """Parse a Praat TextGrid file into a list of Turns.

    PriMock57 uses one IntervalTier per file (Doctor or Patient). We only care
    about intervals with non-empty text. Tags like <UNSURE>, <UNIN/> are kept
    inline — they're part of the gold transcript.
    """
    content = path.read_text(encoding="utf-8")
    turns: list[Turn] = []
    # Match each "intervals [N]:" block with xmin, xmax, text fields.
    pattern = re.compile(
        r"intervals\s*\[\d+\]:\s*"
        r"xmin\s*=\s*([\d.]+)\s*"
        r"xmax\s*=\s*([\d.]+)\s*"
        r'text\s*=\s*"((?:[^"\\]|\\.)*)"',
        re.MULTILINE,
    )
    for match in pattern.finditer(content):
        xmin, xmax, text = match.group(1), match.group(2), match.group(3)
        text = text.strip()
        if not text:
            continue
        turns.append(Turn(speaker=speaker, start=float(xmin), end=float(xmax), text=text))
    return turns


def _merge_consultation(doctor_grid: Path, patient_grid: Path) -> Transcript:
    """Merge doctor + patient TextGrid utterances into a single time-ordered transcript."""
    turns = parse_textgrid(doctor_grid, "doctor") + parse_textgrid(patient_grid, "patient")
    turns.sort(key=lambda t: t.start)
    duration = max((t.end for t in turns), default=0.0)
    return Transcript(turns=turns, language="en", duration=duration)


def load_primock57(data_dir: Path, limit: int | None = None) -> Iterator[Example]:
    """Yield PriMock57 examples.

    Expects ``data_dir`` to contain ``notes/``, ``transcripts/``, and ``audio/``
    as in the upstream repo layout.
    """
    notes_dir = data_dir / "notes"
    transcripts_dir = data_dir / "transcripts"
    audio_dir = data_dir / "audio"

    note_files = sorted(notes_dir.glob("day*_consultation*.json"))
    if limit:
        note_files = note_files[:limit]

    for note_path in note_files:
        record = json.loads(note_path.read_text(encoding="utf-8"))
        cid = note_path.stem  # e.g. "day1_consultation01"

        doctor_grid = transcripts_dir / f"{cid}_doctor.TextGrid"
        patient_grid = transcripts_dir / f"{cid}_patient.TextGrid"
        transcript = None
        if doctor_grid.exists() and patient_grid.exists():
            transcript = _merge_consultation(doctor_grid, patient_grid)

        audio_paths = [
            audio_dir / f"{cid}_doctor.wav",
            audio_dir / f"{cid}_patient.wav",
        ]
        audio_paths = [p for p in audio_paths if p.exists() and p.stat().st_size > 1024]

        yield Example(
            id=cid,
            gold_note=record["note"],
            gold_transcript=transcript,
            audio_paths=audio_paths or None,
        )


def _parse_acibench_dialogue(dialogue_text: str) -> list[Turn]:
    """Split an ACI-Bench dialogue string into Turns.

    Dialogue format: each line is ``[speaker] utterance`` (e.g. ``[doctor] hi``).
    Multi-line utterances continue until the next ``[speaker]`` marker.
    """
    turns: list[Turn] = []
    current_speaker: str | None = None
    current_text: list[str] = []

    def _flush() -> None:
        if current_text:
            joined = " ".join(s.strip() for s in current_text).strip()
            if joined:
                turns.append(Turn(speaker=current_speaker, start=0.0, end=0.0, text=joined))

    for line in dialogue_text.splitlines():
        m = re.match(r"^\[([^\]]+)\]\s*(.*)$", line)
        if m:
            _flush()
            current_speaker = m.group(1).strip().lower()
            current_text = [m.group(2)]
        else:
            current_text.append(line)
    _flush()
    return turns


def load_acibench(
    data_dir: Path,
    limit: int | None = None,
    split: str = "clinicalnlp_taskB_test1",
) -> Iterator[Example]:
    """Yield ACI-Bench examples from a CSV split.

    Expects the upstream layout from https://github.com/wyim/aci-bench:
    ``data/challenge_data/<split>.csv`` with columns
    ``dataset, encounter_id, dialogue, note``.
    """
    import csv

    csv_path = data_dir / "data" / "challenge_data" / f"{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"ACI-Bench split not found at {csv_path}. "
            "Clone https://github.com/wyim/aci-bench and pass --data-dir <clone-path>."
        )

    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            if limit is not None and count >= limit:
                break
            count += 1
            turns = _parse_acibench_dialogue(row["dialogue"])
            transcript = Transcript(turns=turns, language="en", duration=0.0)
            yield Example(
                id=row["encounter_id"],
                gold_note=row["note"],
                gold_transcript=transcript,
            )
