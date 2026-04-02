from models import IncidentState

def grade(state: IncidentState, all_patients_history: list) -> float:
    """Deterministic Grader with Efficiency and Outcome Tracking."""
    score = 0.0
    max_score = 0.0
    
    # Expected treatments mapping
    EXPECTED = {
        "STEMI": {"level": 1, "test": "ECG", "treat": "Aspirin", "ward": "Cardiology"},
        "Sepsis": {"level": 2, "test": "Blood Test", "ward": "ICU"},
        "Ankle Sprain": {"level": 5, "test": "X-Ray"},
        "Hemorrhagic Shock": {"level": 1, "test": "CT Scan", "ward": "Surgery"},
        "Status Asthmaticus": {"level": 1, "ward": "ICU"},
        "Stroke": {"level": 2, "test": "CT Scan", "ward": "Neurology"},
        "Opioid Overdose": {"level": 1, "test": "Tox Screen", "treat": "Naloxone", "ward": "ICU"}
    }
    
    unnecessary_tests_penalty = 0.0
    outcomes_achieved = 0

    for p in all_patients_history:
        max_score += 1.0 # Base 1.0 per patient handling
        p_score = 0.0
        
        cond = p.hidden_condition
        exp = EXPECTED.get(cond, {})
        
        # 1. Did they triage correctly? (+0.3)
        if p.triage_level == exp.get("level", None):
            p_score += 0.3
            
        # 2. Diagnostics
        if "test" in exp:
            if exp["test"] in p.tests_ordered:
                p_score += 0.3
        else:
            p_score += 0.3 # Free points if no specific required test
            
        # Penalize unnecessary tests (-0.05 per test ordered that wasn't expected)
        for t in p.tests_ordered:
            if t != exp.get("test"):
                unnecessary_tests_penalty += 0.05
            
        # 3. Treatment
        if "treat" in exp:
            if exp["treat"] in p.treatments_given:
                p_score += 0.2
        else:
            p_score += 0.2
            
        # 4. Correct disposition (Admit/Discharge) (+0.2)
        if "ward" in exp:
            if p.admitted_ward == exp["ward"]:
                p_score += 0.2
                outcomes_achieved += 1
        else:
            if p.discharged and not p.admitted_ward:
                p_score += 0.2
                outcomes_achieved += 1
                
        score += p_score
        
    # Penalty for fatal errors (drug interactions)
    if state.fatal_errors:
        score -= 0.5 * len(state.fatal_errors)
        
    score -= unnecessary_tests_penalty

    # Efficiency Multiplier: (Max - Current Steps) / Max = efficiency bonus
    # Up to 20% bonus for finishing perfectly in 0 steps (theoretical)
    if max_score > 0 and outcomes_achieved == len(all_patients_history) and len(state.fatal_errors) == 0:
        efficiency = max(0, state.max_steps - state.current_step) / state.max_steps
        score += efficiency * 0.2 * max_score

    if max_score > 0:
        final = score / max_score
    else:
        final = 0.0
        
    return max(0.0, min(1.0, final))
