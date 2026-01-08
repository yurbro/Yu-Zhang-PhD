TRIAGE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "priority": {"type": "string", "enum": ["high", "medium", "low"]},
        "likely_franz": {"type": "string", "enum": ["yes", "no", "uncertain"]},
        "likely_study_type": {"type": "string", "enum": ["IVPT", "IVRT", "both", "uncertain"]},
        "likely_has_numeric_endpoint": {"type": "string", "enum": ["yes", "no", "uncertain"]},
        "likely_has_formulation_info": {"type": "string", "enum": ["yes", "no", "uncertain"]},
        "likely_barrier_category": {"type": "string", "enum": ["skin", "synthetic_membrane", "both", "uncertain"]},
        "what_to_check_in_fulltext": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 5
        },
        "exclude_reason_if_low": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0}
    },
    "required": [
        "priority",
        "likely_franz",
        "likely_study_type",
        "likely_has_numeric_endpoint",
        "likely_has_formulation_info",
        "likely_barrier_category",
        "what_to_check_in_fulltext",
        "exclude_reason_if_low",
        "confidence"
    ]
}
