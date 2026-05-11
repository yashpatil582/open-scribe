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


def load_acibench(data_dir: Path, limit: int | None = None) -> Iterator[Example]:
    """Yield ACI-Bench examples.

    Expects the upstream layout from https://github.com/wyim/aci-bench:
    a ``data/challenge_data/`` directory with ``src_experiment/`` (transcripts)
    and ``tgt_experiment/`` (notes) split into train/valid/test. We pull from
    the test split for held-out eval.
    """
    test_src = data_dir / "data" / "challenge_data" / "src_experiment" / "test"
    test_tgt = data_dir / "data" / "challenge_data" / "tgt_experiment" / "test"

    if not test_src.exists() or not test_tgt.exists():
        raise FileNotFoundError(
            f"ACI-Bench layout not found under {data_dir}. "
            "Clone https://github.com/wyim/aci-bench and pass --data-dir <clone-path>."
        )

    transcript_files = sorted(test_src.glob("*.txt"))
    if limit:
        transcript_files = transcript_files[:limit]

    for src_path in transcript_files:
        cid = src_path.stem
        tgt_path = test_tgt / src_path.name
        if not tgt_path.exists():
            continue
        transcript_text = src_path.read_text(encoding="utf-8")
        note_text = tgt_path.read_text(encoding="utf-8")
        transcript = Transcript(
            turns=[Turn(speaker=None, start=0.0, end=0.0, text=transcript_text)],
            language="en",
            duration=0.0,
        )
        yield Example(id=cid, gold_note=note_text, gold_transcript=transcript)
