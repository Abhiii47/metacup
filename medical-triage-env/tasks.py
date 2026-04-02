import copy
from typing import Dict, Any

EVALUATIONS_DB = {
    "ECG": {"P-101": "ST elevation in leads II, III, aVF (Inferior STEMI).", "P-102": "Normal Sinus Rhythm.", "P-103": "Sinus Tachycardia."},
    "Blood Test": {"P-101": "Troponin highly elevated.", "P-102": "WBC 18.0 (High), Lactate 4.5 (High).", "P-103": "Normal limits."},
    "X-Ray": {"P-103": "No fracture, significant soft tissue swelling."},
    "CT Scan": {"P-107": "Acute ischemic stroke in territory of MCA.", "P-104": "Massive internal bleeding detected."},
    "Tox Screen": {"P-108": "Positive for Opioids and Benzodiazepines."}
}

INTERACTIONS_DB = {
    "Penicillin Allergy": ["Penicillin", "Amoxicillin"],
    "Opioid Overdose": ["Morphine", "Fentanyl", "Oxycodone"],
    "Hemorrhagic Shock": ["Aspirin", "Heparin", "Warfarin"] # Blood thinners are fatal
}

SCENARIOS: Dict[str, Any] = {
    "easy": {
        "max_steps": 15,
        "patients": [
            {
                "id": "P-101",
                "age": 65,
                "vitals": {"HR": "110", "BP": "150/90", "O2": "94%", "Temp": "37.1"},
                "symptoms": ["Crushing chest pain", "Diaphoresis", "Left arm radiation"],
                "history": ["Hypertension", "Smoker"],
                "hidden_condition": "STEMI"
            }
        ],
        "beds": {"Bed_1": None, "Bed_2": None}
    },
    "medium": {
        "max_steps": 20,
        "patients": [
            {
                "id": "P-102",
                "age": 78,
                "vitals": {"HR": "125", "BP": "85/50", "O2": "92%", "Temp": "39.2"},
                "symptoms": ["Confusion", "Fever", "Chills", "Decreased urination"],
                "history": ["UTI recurrences", "Penicillin Allergy"],
                "hidden_condition": "Sepsis"
            },
            {
                "id": "P-108",
                "age": 28,
                "vitals": {"HR": "40", "BP": "90/50", "O2": "82%", "Temp": "36.2"},
                "symptoms": ["Pinpoint pupils", "Unresponsive", "Respiratory depression"],
                "history": ["Substance Abuse"],
                "hidden_condition": "Opioid Overdose"
            }
        ],
        "beds": {"Bed_1": None, "Bed_2": None}
    },
    "hard": {
        "max_steps": 25,
        "patients": [
             {
                "id": "P-104",
                "age": 45,
                "vitals": {"HR": "140", "BP": "70/40", "O2": "88%", "Temp": "36.5"},
                "symptoms": ["Unresponsive", "Massive trauma from MVA"],
                "history": ["Unknown"],
                "hidden_condition": "Hemorrhagic Shock"
            },
            {
                "id": "P-107",
                "age": 62,
                "vitals": {"HR": "85", "BP": "190/110", "O2": "96%", "Temp": "37.4"},
                "symptoms": ["Facial droop", "Slurred speech", "Left arm weakness"],
                "history": ["Hypertension"],
                "hidden_condition": "Stroke"
            },
            {
                "id": "P-105",
                "age": 9,
                "vitals": {"HR": "130", "BP": "100/60", "O2": "90%", "Temp": "37.5"},
                "symptoms": ["Severe wheezing", "Accessory muscle use", "Can't speak in full sentences"],
                "history": ["Asthma"],
                "hidden_condition": "Status Asthmaticus"
            }
        ],
        # NOTE: 3 patients, 2 beds — agent MUST prioritize triage order. P-104 is most critical.
        "beds": {"Bed_1": None, "Bed_2": None}
    }
}

def get_scenario(difficulty: str) -> Dict[str, Any]:
    return copy.deepcopy(SCENARIOS.get(difficulty.lower(), SCENARIOS["easy"]))
