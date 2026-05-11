"""Smoke tests that don't require model downloads or network calls."""

import base64
import json

from fhir.resources.R4B.bundle import Bundle

from open_scribe.fhir_emit import soap_to_text, to_bundle, to_document_reference
from open_scribe.note_gen import SoapNote


SAMPLE_NOTE = SoapNote(
    subjective="patient reports cough x 3 days",
    objective="t 38.1, lungs clear",
    assessment="acute viral URI",
    plan="rest, fluids, return if worsening",
)


def test_soap_to_text_renders_all_sections():
    text = soap_to_text(SAMPLE_NOTE)
    assert "SUBJECTIVE" in text
    assert "OBJECTIVE" in text
    assert "ASSESSMENT" in text
    assert "PLAN" in text
    assert "patient reports cough" in text


def test_fhir_bundle_has_expected_resources():
    bundle = to_bundle(SAMPLE_NOTE)
    assert bundle.type == "transaction"
    assert len(bundle.entry) == 3
    types = [e.resource.__resource_type__ for e in bundle.entry]
    assert types == ["Patient", "Encounter", "DocumentReference"]


def test_fhir_bundle_embeds_soap_as_attachment():
    """In-memory, fhir.resources stores Attachment.data as the *decoded* bytes."""
    bundle = to_bundle(SAMPLE_NOTE)
    attachment = bundle.entry[2].resource.content[0].attachment
    assert attachment.contentType == "text/plain"
    text = attachment.data.decode("utf-8")
    assert "SUBJECTIVE" in text
    assert "acute viral URI" in text


def test_fhir_bundle_json_attachment_is_base64():
    """When serialized to JSON, the attachment must be base64-encoded per FHIR spec."""
    bundle_dict = to_document_reference(SAMPLE_NOTE)
    encoded_data = bundle_dict["entry"][2]["resource"]["content"][0]["attachment"]["data"]
    decoded = base64.b64decode(encoded_data).decode("utf-8")
    assert "SUBJECTIVE" in decoded
    assert "acute viral URI" in decoded


def test_fhir_bundle_round_trips_through_json_validation():
    """JSON-dump the bundle and re-validate it. Catches any invalid FHIR R4B output."""
    bundle_dict = to_document_reference(SAMPLE_NOTE)
    Bundle.model_validate(json.loads(json.dumps(bundle_dict)))


def test_fhir_bundle_links_encounter_and_patient_correctly():
    bundle = to_bundle(SAMPLE_NOTE)
    patient = bundle.entry[0].resource
    encounter = bundle.entry[1].resource
    doc_ref = bundle.entry[2].resource

    assert encounter.subject.reference == f"Patient/{patient.id}"
    assert doc_ref.subject.reference == f"Patient/{patient.id}"
    assert doc_ref.context.encounter[0].reference == f"Encounter/{encounter.id}"


def test_icd10_lookup_recognises_common_codes():
    from open_scribe.icd10_common import is_valid_cpt, is_valid_icd10, lookup

    assert is_valid_icd10("J06.9")
    assert is_valid_icd10("j06.9")          # case-insensitive
    assert not is_valid_icd10("ZZZ.99")     # nonsense -> rejected
    assert is_valid_cpt("99213")
    assert not is_valid_cpt("00000")

    system, desc = lookup("J06.9")
    assert system == "icd-10-cm"
    assert "upper respiratory" in desc.lower()
