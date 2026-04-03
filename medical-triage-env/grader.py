from models import IncidentState
from typing import List


# Per-condition expected clinical actions
EXPECTED = {
    "STEMI": {
        "level": 1,
        "tests": ["ECG"],
        "treat": ["Aspirin", "Nitroglycerin", "Heparin"],
        "ward": "Cardiology",
    },
    "Sepsis": {
        "level": 2,
        "tests": ["Blood Test"],
        # Any broad-spectrum antibiotic except contraindicated ones is acceptable
        "treat": ["Vancomycin", "Ceftriaxone", "Meropenem", "Piperacillin", "Fluids", "IV Fluids"],
        "ward": "ICU",
    },
    "Ankle Sprain": {
        "level": 5,
        "tests": ["X-Ray"],
        "treat": [],
        "ward": None,
    },
    "Hemorrhagic Shock": {
        "level": 1,
        "tests": ["CT Scan"],
        "treat": ["Blood Transfusion", "IV Fluids", "Fluids", "Transfusion"],
        "ward": "Surgery",
    },
    "Status Asthmaticus": {
        "level": 1,
        "tests": [],
        "treat": ["Albuterol", "Salbutamol", "Epinephrine", "Steroids", "Oxygen"],
        "ward": "ICU",
    },
    "Stroke": {
        "level": 2,
        "tests": ["CT Scan"],
        "treat": ["tPA", "Alteplase", "Aspirin"],
        "ward": "Neurology",
    },
    "Opioid Overdose": {
        "level": 1,
        "tests": ["Tox Screen"],
        "treat": ["Naloxone"],
        "ward": "ICU",
    },
}


def grade(state: IncidentState, all_patients_history: list) -> float:
    """
    Deterministic grader. Computes a 0.0–1.0 score for the episode.

    Scoring breakdown per patient (1.0 base):
      +0.25 — Correct triage level assigned
      +0.25 — Required diagnostic test ordered
      +0.25 — Appropriate treatment selected (any accepted drug)
      +0.25 — Correct disposition (ward or discharge)

    Penalties:
      -0.10 per unnecessary test ordered
      -0.40 per fatal drug interaction
      +Efficiency bonus (up to +0.15 max) if all patients handled early

    Final score is clamped to [0.0, 1.0].
    """
    score = 0.0
    max_score = 0.0
    unnecessary_penalty = 0.0
    outcomes_achieved = 0

    for p in all_patients_history:
        max_score += 1.0
        p_score = 0.0
        cond = p.hidden_condition
        exp = EXPECTED.get(cond, {})

        # --- Triage (0.25) ---
        if p.triage_level == exp.get("level"):
            p_score += 0.25

        # --- Diagnostic test (0.25) ---
        required_tests = exp.get("tests", [])
        if not required_tests:
            p_score += 0.25  # No required test → free points
        else:
            if any(t in p.tests_ordered for t in required_tests):
                p_score += 0.25

        # Unnecessary test penalty (-0.10 each)
        for t in p.tests_ordered:
            if required_tests and t not in required_tests:
                unnecessary_penalty += 0.10

        # --- Treatment (0.25) ---
        accepted_treats = exp.get("treat", [])
        if not accepted_treats:
            p_score += 0.25  # No required treatment → free points
        else:
            if any(t in p.treatments_given for t in accepted_treats):
                p_score += 0.25

        # --- Disposition (0.25) ---
        target_ward = exp.get("ward")
        if target_ward:
            if p.admitted_ward == target_ward:
                p_score += 0.25
                outcomes_achieved += 1
        else:
            # Dischargeable conditions (e.g., ankle sprain)
            if p.discharged and not p.admitted_ward:
                p_score += 0.25
                outcomes_achieved += 1

        score += p_score

    # Fatal interaction penalty
    score -= 0.40 * len(state.fatal_errors)

    # Unnecessary test penalty
    score -= unnecessary_penalty

    # Efficiency bonus: up to +0.15 if all patients fully handled
    n_patients = len(all_patients_history)
    if max_score > 0 and outcomes_achieved == n_patients and not state.fatal_errors:
        steps_used = state.current_step
        efficiency = max(0.0, (state.max_steps - steps_used) / state.max_steps)
        score += efficiency * 0.15 * max_score

    # Normalize and clamp
    if max_score > 0:
        final = score / max_score
    else:
        final = 0.0

    return round(max(0.0, min(1.0, final)), 4)


def grade_task(task_id: str, state: IncidentState, all_patients_history: list) -> float:
    """
    Grade a specific task by ID. Thin wrapper around grade() that logs context.
    Returns a float in [0.0, 1.0].
    """
    return grade(state, all_patients_history)
