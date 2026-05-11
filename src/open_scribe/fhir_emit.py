"""FHIR emission stage: SOAP note + codes -> FHIR R4B Bundle (Patient + Encounter + DocumentReference).

The output is a single Bundle so the result is self-contained — you can POST
it to any FHIR server as a transaction, or open it in a tool like Inferno.

References:
  - DocumentReference.type: LOINC 11506-3 "Subsequent evaluation note"
  - Encounter.class: ActCode "AMB" (ambulatory)
  - All resources use synthetic IDs and a placeholder Patient. The pipeline
    is dataset-agnostic; callers pass `patient_name` / `encounter_period`
    when emitting against real (mock) data.
"""

from __future__ import annotations

import base64
import uuid
from datetime import datetime, timezone

from fhir.resources.R4B.attachment import Attachment
from fhir.resources.R4B.bundle import Bundle, BundleEntry, BundleEntryRequest
from fhir.resources.R4B.codeableconcept import CodeableConcept
from fhir.resources.R4B.coding import Coding
from fhir.resources.R4B.documentreference import (
    DocumentReference,
    DocumentReferenceContent,
    DocumentReferenceContext,
)
from fhir.resources.R4B.encounter import Encounter
from fhir.resources.R4B.humanname import HumanName
from fhir.resources.R4B.identifier import Identifier
from fhir.resources.R4B.patient import Patient
from fhir.resources.R4B.period import Period
from fhir.resources.R4B.reference import Reference

from open_scribe.coder import CodeSuggestion
from open_scribe.note_gen import SoapNote

LOINC_PROGRESS_NOTE = Coding(
    system="http://loinc.org",
    code="11506-3",
    display="Subsequent evaluation note",
)
ACT_CODE_AMBULATORY = Coding(
    system="http://terminology.hl7.org/CodeSystem/v3-ActCode",
    code="AMB",
    display="ambulatory",
)


def soap_to_text(note: SoapNote) -> str:
    """Render a SOAP note as plain text for embedding in DocumentReference.attachment."""
    return (
        "SUBJECTIVE\n" + note.subjective.strip() + "\n\n"
        "OBJECTIVE\n" + note.objective.strip() + "\n\n"
        "ASSESSMENT\n" + note.assessment.strip() + "\n\n"
        "PLAN\n" + note.plan.strip() + "\n"
    )


def _make_patient(name: str | None) -> Patient:
    given, family = ("Anonymous", "Patient")
    if name:
        parts = name.strip().split(maxsplit=1)
        given = parts[0]
        family = parts[1] if len(parts) > 1 else ""
    return Patient(
        id=str(uuid.uuid4()),
        identifier=[Identifier(system="urn:open-scribe:patient", value=str(uuid.uuid4()))],
        name=[HumanName(given=[given], family=family or None)],
    )


def _make_encounter(patient_id: str, period: Period | None) -> Encounter:
    return Encounter(
        id=str(uuid.uuid4()),
        status="finished",
        class_fhir=ACT_CODE_AMBULATORY,
        subject=Reference(reference=f"Patient/{patient_id}"),
        period=period,
    )


def _make_document_reference(
    note: SoapNote,
    patient_id: str,
    encounter_id: str,
    codes: list[CodeSuggestion] | None = None,
) -> DocumentReference:
    # fhir.resources expects Attachment.data as a base64-encoded string;
    # it decodes internally and re-encodes on JSON serialization.
    encoded = base64.b64encode(soap_to_text(note).encode("utf-8")).decode("ascii")
    attachment = Attachment(
        contentType="text/plain",
        data=encoded,
        language="en",
        title="SOAP note",
    )

    doc_ref = DocumentReference(
        id=str(uuid.uuid4()),
        status="current",
        docStatus="final",
        type=CodeableConcept(coding=[LOINC_PROGRESS_NOTE], text="Progress note"),
        subject=Reference(reference=f"Patient/{patient_id}"),
        date=datetime.now(timezone.utc).isoformat(),
        content=[DocumentReferenceContent(attachment=attachment)],
        context=DocumentReferenceContext(
            encounter=[Reference(reference=f"Encounter/{encounter_id}")],
        ),
    )

    if codes:
        # Optional: attach top suggested codes as DocumentReference.category for searchability.
        # (Real systems would use Condition / Procedure resources; this is a v1 shortcut.)
        category_codings = [
            Coding(
                system=(
                    "http://hl7.org/fhir/sid/icd-10-cm"
                    if c.system == "icd-10-cm"
                    else "http://www.ama-assn.org/go/cpt"
                ),
                code=c.code,
                display=c.description,
            )
            for c in codes[:5]
        ]
        if category_codings:
            doc_ref.category = [CodeableConcept(coding=category_codings)]

    return doc_ref


def to_bundle(
    note: SoapNote,
    codes: list[CodeSuggestion] | None = None,
    patient_name: str | None = None,
    encounter_period: Period | None = None,
) -> Bundle:
    """Build a self-contained FHIR R4B Bundle: Patient + Encounter + DocumentReference."""
    patient = _make_patient(patient_name)
    encounter = _make_encounter(patient.id, encounter_period)
    doc_ref = _make_document_reference(note, patient.id, encounter.id, codes=codes)

    def _entry(resource, resource_type: str) -> BundleEntry:
        return BundleEntry(
            fullUrl=f"urn:uuid:{resource.id}",
            resource=resource,
            request=BundleEntryRequest(method="POST", url=resource_type),
        )

    return Bundle(
        id=str(uuid.uuid4()),
        type="transaction",
        timestamp=datetime.now(timezone.utc).isoformat(),
        entry=[
            _entry(patient, "Patient"),
            _entry(encounter, "Encounter"),
            _entry(doc_ref, "DocumentReference"),
        ],
    )


def to_document_reference(
    note: SoapNote,
    codes: list[CodeSuggestion] | None = None,
    patient_name: str | None = None,
    encounter_period: Period | None = None,
) -> dict:
    """Build the full Bundle and return it as a JSON-serializable dict (pipeline-facing API)."""
    bundle = to_bundle(note, codes=codes, patient_name=patient_name, encounter_period=encounter_period)
    return bundle.model_dump(mode="json", exclude_none=True)
