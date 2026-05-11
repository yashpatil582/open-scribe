"""LLM client factory.

Single place to configure which LLM backend the rest of the pipeline talks to.
Anything OpenAI-compatible works — Ollama, Groq, vLLM, OpenAI, Together, etc.

Default is local Ollama. To use a hosted backend, set OPEN_SCRIBE_BASE_URL +
OPEN_SCRIBE_API_KEY (+ optionally OPEN_SCRIBE_MODEL). Example for Groq:

    export OPEN_SCRIBE_BASE_URL=https://api.groq.com/openai/v1
    export OPEN_SCRIBE_API_KEY=$GROQ_API_KEY
    export OPEN_SCRIBE_MODEL=llama-3.1-8b-instant

The local-Ollama default is deliberate: zero data egress matches the
HIPAA-mindful framing of this project.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import instructor
from openai import OpenAI


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    label: str


def resolve_config() -> LLMConfig:
    if base := os.environ.get("OPEN_SCRIBE_BASE_URL"):
        return LLMConfig(
            base_url=base,
            api_key=os.environ.get("OPEN_SCRIBE_API_KEY", "sk-unused"),
            model=os.environ.get("OPEN_SCRIBE_MODEL", "gpt-4o-mini"),
            label="custom",
        )
    return LLMConfig(
        base_url=os.environ.get("OLLAMA_HOST", "http://localhost:11434") + "/v1",
        api_key="ollama",
        model=os.environ.get("OPEN_SCRIBE_MODEL", "llama3.1:latest"),
        label="ollama",
    )


def make_client(config: LLMConfig | None = None) -> tuple[instructor.Instructor, LLMConfig]:
    """Return an instructor-wrapped OpenAI-compatible client + the resolved config."""
    config = config or resolve_config()
    raw = OpenAI(base_url=config.base_url, api_key=config.api_key)
    return instructor.from_openai(raw, mode=instructor.Mode.JSON), config
