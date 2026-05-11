"""Bundled subset of common ICD-10-CM and CPT codes for code-suggestion validation.

This is a curated list (~150 codes) of high-frequency primary-care diagnoses
and E/M visit codes. Production deployments should swap in the full CMS
ICD-10-CM tabular list (~73k codes) and the AMA CPT codebook.

Sources:
  - ICD-10-CM: based on the most-billed primary-care codes per CMS 2024.
  - CPT: limited to outpatient E/M codes 99201-99499 from CMS HCPCS;
    full CPT requires AMA licensing for redistribution.
"""

from __future__ import annotations

ICD10_CM: dict[str, str] = {
    # Respiratory
    "J00": "Acute nasopharyngitis (common cold)",
    "J02.9": "Acute pharyngitis, unspecified",
    "J03.90": "Acute tonsillitis, unspecified",
    "J04.0": "Acute laryngitis",
    "J06.9": "Acute upper respiratory infection, unspecified",
    "J11.1": "Influenza with other respiratory manifestations",
    "J18.9": "Pneumonia, unspecified organism",
    "J20.9": "Acute bronchitis, unspecified",
    "J30.9": "Allergic rhinitis, unspecified",
    "J32.9": "Chronic sinusitis, unspecified",
    "J40": "Bronchitis, not specified as acute or chronic",
    "J44.9": "Chronic obstructive pulmonary disease, unspecified",
    "J45.909": "Unspecified asthma, uncomplicated",
    # GI
    "A09": "Infectious gastroenteritis and colitis, unspecified",
    "K21.9": "Gastro-esophageal reflux disease without esophagitis",
    "K29.70": "Gastritis, unspecified, without bleeding",
    "K52.9": "Noninfective gastroenteritis and colitis, unspecified",
    "K57.30": "Diverticulosis of large intestine without complications",
    "K58.9": "Irritable bowel syndrome without diarrhea",
    "K59.00": "Constipation, unspecified",
    "R10.9": "Unspecified abdominal pain",
    "R11.2": "Nausea with vomiting, unspecified",
    "R19.7": "Diarrhea, unspecified",
    # Cardio
    "I10": "Essential (primary) hypertension",
    "I25.10": "Atherosclerotic heart disease of native coronary artery without angina",
    "I48.91": "Unspecified atrial fibrillation",
    "I50.9": "Heart failure, unspecified",
    "R00.0": "Tachycardia, unspecified",
    "R03.0": "Elevated blood-pressure reading, without diagnosis of hypertension",
    "R07.9": "Chest pain, unspecified",
    "R55": "Syncope and collapse",
    # Endocrine / Metabolic
    "E03.9": "Hypothyroidism, unspecified",
    "E11.9": "Type 2 diabetes mellitus without complications",
    "E78.5": "Hyperlipidemia, unspecified",
    "E66.9": "Obesity, unspecified",
    # MSK
    "M25.50": "Pain in unspecified joint",
    "M54.2": "Cervicalgia",
    "M54.5": "Low back pain",
    "M79.10": "Myalgia, unspecified site",
    "S93.4": "Sprain of ankle",
    "S60.50": "Superficial injury of finger",
    # Skin
    "L03.90": "Cellulitis, unspecified",
    "L20.9": "Atopic dermatitis, unspecified",
    "L23.9": "Allergic contact dermatitis, unspecified cause",
    "L70.0": "Acne vulgaris",
    "L98.9": "Disorder of the skin and subcutaneous tissue, unspecified",
    # Mental health
    "F32.9": "Major depressive disorder, single episode, unspecified",
    "F33.9": "Major depressive disorder, recurrent, unspecified",
    "F41.1": "Generalized anxiety disorder",
    "F41.9": "Anxiety disorder, unspecified",
    "F43.10": "Post-traumatic stress disorder, unspecified",
    "G47.00": "Insomnia, unspecified",
    # Neuro
    "G43.909": "Migraine, unspecified, not intractable, without status migrainosus",
    "G44.209": "Tension-type headache, unspecified, not intractable",
    "R51": "Headache",
    "R42": "Dizziness and giddiness",
    # GU / GYN
    "N39.0": "Urinary tract infection, site not specified",
    "N18.9": "Chronic kidney disease, unspecified",
    "N94.6": "Dysmenorrhea, unspecified",
    "Z34.90": "Encounter for supervision of normal pregnancy, unspecified",
    # Symptoms / signs
    "R05.9": "Cough, unspecified",
    "R06.02": "Shortness of breath",
    "R06.2": "Wheezing",
    "R09.81": "Nasal congestion",
    "R50.9": "Fever, unspecified",
    "R53.83": "Other fatigue",
    "R63.0": "Anorexia",
    # Preventive
    "Z00.00": "Encounter for general adult medical examination without abnormal findings",
    "Z23": "Encounter for immunization",
    # Infectious
    "B34.9": "Viral infection, unspecified",
    "U07.1": "COVID-19",
    "H66.90": "Otitis media, unspecified, unspecified ear",
}


CPT: dict[str, str] = {
    # Office / outpatient E/M (most-used by primary care)
    "99202": "Office or other outpatient visit, new patient, low complexity (15-29 min)",
    "99203": "Office or other outpatient visit, new patient, low-moderate complexity (30-44 min)",
    "99204": "Office or other outpatient visit, new patient, moderate complexity (45-59 min)",
    "99205": "Office or other outpatient visit, new patient, high complexity (60-74 min)",
    "99211": "Office or other outpatient visit, established patient, minimal (5 min)",
    "99212": "Office or other outpatient visit, established patient, low complexity (10-19 min)",
    "99213": "Office or other outpatient visit, established patient, low-moderate complexity (20-29 min)",
    "99214": "Office or other outpatient visit, established patient, moderate complexity (30-39 min)",
    "99215": "Office or other outpatient visit, established patient, high complexity (40-54 min)",
    # Preventive
    "99381": "Initial preventive medicine visit, age <1",
    "99391": "Periodic preventive medicine visit, established patient, age <1",
    "99395": "Periodic preventive medicine visit, established patient, age 18-39",
    "99396": "Periodic preventive medicine visit, established patient, age 40-64",
    # Telehealth & misc
    "99441": "Telephone E/M, established patient, 5-10 min",
    "99442": "Telephone E/M, established patient, 11-20 min",
    "99443": "Telephone E/M, established patient, 21-30 min",
    # Common procedures
    "90471": "Immunization administration, one vaccine",
    "90472": "Immunization administration, each additional vaccine",
    "94640": "Pressurized or non-pressurized inhalation treatment",
    "36415": "Routine venipuncture",
}


def is_valid_icd10(code: str) -> bool:
    return code.upper() in {k.upper() for k in ICD10_CM}


def is_valid_cpt(code: str) -> bool:
    return code in CPT


def lookup(code: str) -> tuple[str, str] | None:
    """Return (system, description) for a known code, else None."""
    upper = code.upper()
    for c, desc in ICD10_CM.items():
        if c.upper() == upper:
            return ("icd-10-cm", desc)
    if code in CPT:
        return ("cpt", CPT[code])
    return None
