# open-scribe

[![CI](https://github.com/yashpatil582/open-scribe/actions/workflows/ci.yml/badge.svg)](https://github.com/yashpatil582/open-scribe/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Open-source, FHIR-native medical scribe baseline.

**Pipeline:** encounter audio → SOAP note → ICD-10 / CPT suggestions → FHIR `DocumentReference`, with an eval harness reporting section-level F1 against gold notes.

This is a public, reproducible baseline for ambient clinical AI systems (the same shape as commercial products like Commure Scribe, Suki, DeepScribe). All data used is public — no PHI.

## Status

Full v1 pipeline complete: ASR + SOAP generation + ICD-10/CPT coding + FHIR R4B Bundle emission, with a real eval harness reporting whole-note F1 on PriMock57. See `Eval results` below. Remaining work is mostly polish + scaling to a paid LLM tier.

## Quickstart

```bash
cd open-scribe
pip install -e ".[notes]"          # ASR + LLM note generation

# Point the LLM at any OpenAI-compatible endpoint. Default: local Ollama.
# Groq example (fast, free tier, no PHI in this repo so external is fine for the public demo):
export OPEN_SCRIBE_BASE_URL=https://api.groq.com/openai/v1
export OPEN_SCRIBE_API_KEY=$GROQ_API_KEY
export OPEN_SCRIBE_MODEL=llama-3.3-70b-versatile

# Run the full pipeline: audio -> SOAP -> FHIR Bundle (Patient + Encounter + DocumentReference)
python -m open_scribe.pipeline --audio path/to/encounter.wav --out bundle.json

# Or just transcribe
python -m open_scribe.asr path/to/encounter.wav
```

For the eval harness (Day 8-9, in progress):

```bash
pip install -e ".[eval]"
```

For speaker diarization (clinician vs. patient turns) — requires HuggingFace token + accepting `pyannote/speaker-diarization-3.1` license:

```bash
pip install -e ".[diarize]"
export HF_TOKEN=hf_...
```

## Datasets

| Dataset       | Audio | Gold notes | Access                 | Use                         |
|---------------|-------|------------|------------------------|-----------------------------|
| PriMock57     | yes   | yes        | open download          | primary eval                |
| ACI-Bench     | yes   | yes        | open download          | secondary eval, generalization |
| MTSamples     | no    | yes        | open                   | text-only eval               |
| MIMIC-IV-Note | no    | yes        | PhysioNet credentialed | optional fine-tuning        |

See `eval/datasets.md` (TODO) for download scripts.

## Eval results

Methodology:

- **Whole-note F1**: token-level F1 between predicted SOAP note (rendered as text) and the gold note. Token-level multiset overlap; punctuation stripped; case-folded.
- **Setup**: Llama 3.3 70B via Groq, default prompt (`src/open_scribe/note_gen.py`), gold transcripts (no ASR error introduced).

| Dataset    | n         | Whole-note F1 | Precision | Recall | Notes                                                |
|------------|-----------|--------------:|----------:|-------:|------------------------------------------------------|
| PriMock57  | **57 / 57** |     **0.279** | 0.266     | 0.315  | Full split; gold notes are dense clinician shorthand |
| ACI-Bench  | 35 / 40   |     **0.447** | 0.709     | 0.338  | Test split (`clinicalnlp_taskB_test1`)              |

> **Interpreting the gap.** ACI-Bench scores ~1.6× higher because its gold notes are full English prose (CHIEF COMPLAINT, HISTORY OF PRESENT ILLNESS sections) that match the LLM's natural output style. PriMock57's gold uses dense clinician shorthand (`3/7 hx of diarrhea`, `LLQ pain`) — the model produces clinically equivalent content but rarely shares surface tokens. The metric is honest; the system isn't broken.
>
> Precision > recall on ACI-Bench tells you the LLM's notes are mostly-correct content (high precision = few extraneous tokens) but shorter than gold (low recall = missing some sections). A longer system prompt with explicit section requirements would close the gap.

> **How the full PriMock57 run was completed on the free tier.** Groq's free-tier `llama-3.3-70b-versatile` caps at ~100K tokens-per-day per *organization* (not per key — generating new keys in the same org doesn't help). The `eval/run_eval.py --resume` flag merges new examples into the existing results file, so the full 57 was completed across three same-day runs as the rolling token bucket refilled. Total elapsed: ~12 minutes of wall-clock LLM time spread over ~30 minutes of polling. Alternatively: paid Groq Dev tier or any other OpenAI-compatible endpoint via `OPEN_SCRIBE_BASE_URL`.

**On the number.** F1 ≈ 0.25 is honest, not spectacular. Why this number, and why it doesn't mean the system is broken:

1. PriMock57 gold notes are *extremely* dense clinical shorthand — e.g. `"3/7 hx of diarrhea, mainly watery. No blood in stool. Opening bowels x6/day. Associated LLQ pain - crampy, intermittent..."`. They use abbreviations the LLM doesn't produce (`3/7` = "for three days", `LLQ` = "left lower quadrant", `hx` = "history").
2. The LLM produces full English prose. Even a *clinically equivalent* SOAP note won't share many surface tokens with the gold.
3. Best PriMock57 examples score F1 ≈ 0.31; worst score F1 ≈ 0.15. Range is reasonable; nothing is catastrophically broken.
4. For comparison, Commure publishes F1 ≈ 0.8–0.9, but that's against *Commure-internal* gold notes designed for their model, scored on their metric — not a directly comparable number.

**What would move the number:**

- Match the gold's *style* via few-shot examples in the prompt — abbreviations, telegraphic format.
- Use a section-aware metric (when paired with a structured-note dataset like ACI-Bench).
- Switch to clinical-NER overlap or ROUGE-L instead of token F1.
- Fine-tune a small model on the gold notes.

**Reproducing:**

```bash
# Gold-transcript eval, all 57 (needs Groq Dev tier or any paid backend)
python -m eval.run_eval --dataset primock57 --data-dir datasets/primock57 --from-gold-transcript

# Quick smoke (3 examples, free tier OK)
python -m eval.run_eval --dataset primock57 --data-dir datasets/primock57 \
    --from-gold-transcript --limit 3
```

## Architecture

```
audio.wav
   │
   ▼  asr.py        faster-whisper + (optional) pyannote diarization
[diarized transcript: turn-tagged]
   │
   ▼  note_gen.py   structured-output LLM, JSON schema = SOAP sections
{ subjective, objective, assessment, plan }
   │
   ▼  coder.py      LLM proposes ICD-10 / CPT, validated against code tables
{ suggested_codes: [...] }
   │
   ▼  fhir_emit.py  fhir.resources -> DocumentReference + Encounter
DocumentReference.json (FHIR R4)
```

## Roadmap

- [x] Day 1: Repo scaffold, pyproject, ASR module wired to faster-whisper
- [x] Day 2: ASR CLI, optional pyannote diarization wired
- [x] Day 3–4: Note generation with structured output (SOAP JSON schema) via `instructor`
- [x] Day 5: FHIR R4B Bundle emission (Patient + Encounter + DocumentReference), JSON-validated
- [x] Day 6–7: pipeline.py end-to-end; demo notebook executed with outputs persisted
- [x] Day 8–9: Eval harness on PriMock57 — **57/57 examples**, whole-note F1 = 0.279
- [x] Day 10: ICD-10 / CPT suggestion stage with bundled validation lookup
- [x] Day 11: ACI-Bench eval — 35/40 examples, whole-note F1 = 0.447
- [ ] Day 12–13: README polish, demo recording, push public

## Why this exists

A reference implementation of the ambient-scribe pipeline that anyone can run, evaluate, and extend — without access to proprietary clinical data.

## License

Apache 2.0.
