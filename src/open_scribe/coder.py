"""Medical coding stage: SOAP note -> ICD-10-CM and CPT code suggestions.

LLM proposes candidate codes over the Assessment + Plan sections; each
suggestion is validated against a bundled lookup of ~150 common ICD-10-CM
and ~25 CPT codes. Hallucinated codes are dropped.

For production use, swap ``icd10_common`` for the full CMS ICD-10-CM tabular
list (~73k codes) and the AMA-licensed CPT codebook.
"""

from __future__ import annotations

import click
from pydantic import BaseModel, Field
from rich.console import Console

from open_scribe.icd10_common import CPT, ICD10_CM, lookup
from open_scribe.llm import LLMConfig, make_client
from open_scribe.note_gen import SoapNote

console = Console()


class CodeSuggestion(BaseModel):
    code: str
    system: str = Field(description="'icd-10-cm' or 'cpt'")
    description: str
    rationale: str = Field(description="Span from the note that supports this code")
    confidence: float = Field(ge=0.0, le=1.0)


class _CodeSuggestionList(BaseModel):
    suggestions: list[CodeSuggestion]


SYSTEM_PROMPT = """You are a careful medical coder.

Given a SOAP note, propose the most likely ICD-10-CM diagnosis codes (for the
Assessment) and CPT codes (for the visit type and any procedures in the Plan).

Rules:
- Only suggest codes you are confident exist in the standard ICD-10-CM / CPT codebooks.
- Always include a short ``rationale`` quoting or paraphrasing the supporting span from the note.
- ``confidence`` reflects how strongly the note supports the code (0.0-1.0).
- Sort suggestions by confidence descending.
- Return at most 6 suggestions total.
- If the note is empty or non-clinical, return an empty list.
"""


def suggest_codes(
    note: SoapNote,
    config: LLMConfig | None = None,
    max_retries: int = 2,
) -> list[CodeSuggestion]:
    """Suggest ICD-10-CM and CPT codes for a SOAP note.

    Codes not present in the bundled lookup are dropped. The description is
    overwritten with the canonical lookup description (so the LLM can't fudge
    a wrong description onto a real code).
    """
    if not (note.assessment.strip() or note.plan.strip()):
        return []

    client, cfg = make_client(config)
    console.log(f"coder: backend={cfg.label} model={cfg.model}")

    user_msg = (
        "Assessment:\n" + note.assessment.strip() + "\n\n"
        "Plan:\n" + note.plan.strip() + "\n\n"
        "Subjective (context):\n" + note.subjective.strip()
    )

    result = client.chat.completions.create(
        model=cfg.model,
        response_model=_CodeSuggestionList,
        max_retries=max_retries,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )

    validated: list[CodeSuggestion] = []
    for s in result.suggestions:
        canonical = lookup(s.code)
        if canonical is None:
            continue
        system, description = canonical
        validated.append(
            CodeSuggestion(
                code=s.code.upper() if system == "icd-10-cm" else s.code,
                system=system,
                description=description,
                rationale=s.rationale,
                confidence=s.confidence,
            )
        )

    validated.sort(key=lambda s: s.confidence, reverse=True)
    return validated


@click.command()
def main() -> None:
    """Print bundled lookup sizes — sanity check that the data file loaded."""
    console.print(f"ICD-10-CM codes bundled: {len(ICD10_CM)}")
    console.print(f"CPT codes bundled:       {len(CPT)}")


if __name__ == "__main__":
    main()
