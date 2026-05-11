"""Note generation: transcript -> structured SOAP note via an LLM with JSON-schema output."""

from __future__ import annotations

import click
from pydantic import BaseModel, Field
from rich.console import Console

from open_scribe.asr import Transcript, transcribe
from open_scribe.llm import LLMConfig, make_client

console = Console()


class SoapNote(BaseModel):
    """SOAP note: the standard structured clinical encounter document.

    Each section is plain prose. Empty strings are allowed (e.g. an Objective
    section may be empty for a telehealth encounter with no exam findings),
    but the LLM should be prompted to fill them whenever the transcript supports it.
    """

    subjective: str = Field(
        description="Patient's reported symptoms, history, and concerns in their own words"
    )
    objective: str = Field(
        description="Observable findings: vitals, exam findings, lab/imaging results"
    )
    assessment: str = Field(
        description="Clinician's diagnostic interpretation and differential"
    )
    plan: str = Field(
        description="Treatment plan: medications, procedures, follow-up, patient instructions"
    )


SYSTEM_PROMPT = """You are a careful medical scribe assistant.

Given a transcript of a clinician-patient encounter, write a SOAP note that
faithfully reflects what was discussed. Rules:

- Use only information present in the transcript. Do not invent vitals, findings, or diagnoses.
- If a section has no supporting content in the transcript, return an empty string for that section. Do not write "Not discussed" or filler phrases.
- Subjective: chief complaint, HPI, ROS, relevant history — in the clinician's voice, not as quoted dialogue.
- Objective: vitals and physical exam findings and any reviewed labs/imaging that are explicitly stated.
- Assessment: the clinician's stated impression / working diagnosis. If multiple, list them.
- Plan: medications prescribed, tests ordered, procedures, follow-up, patient education.
- Keep each section concise: prefer clinical density over verbosity.
"""


def generate_note(
    transcript: Transcript,
    config: LLMConfig | None = None,
    max_retries: int = 2,
) -> SoapNote:
    """Convert a transcript into a structured SOAP note using the configured LLM."""
    client, cfg = make_client(config)
    console.log(f"note_gen: backend={cfg.label} model={cfg.model}")

    return client.chat.completions.create(
        model=cfg.model,
        response_model=SoapNote,
        max_retries=max_retries,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Transcript:\n\n{transcript.text}"},
        ],
    )


@click.command()
@click.argument("audio_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--asr-model", default="base.en", show_default=True)
def main(audio_path: str, asr_model: str) -> None:
    """Transcribe an audio file and generate a SOAP note. For quick manual testing."""
    transcript = transcribe(audio_path, model_size=asr_model)
    note = generate_note(transcript)
    console.print_json(note.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
