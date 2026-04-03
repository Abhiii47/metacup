---
title: Medical Triage OpenEnv
emoji: 🏥
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
tags:
  - openenv
---

# 🏥 NexaCare ER — Medical Triage Nurse | OpenEnv

> **Meta PyTorch Hackathon Submission** · An OpenEnv environment where an AI agent acts as a clinical Emergency Room Triage Nurse, making high-stakes decisions under time pressure with real clinical consequences.

---

## 🎯 What Is This?

This environment simulates a real hospital Emergency Department. The agent receives a queue of patients with authentic vital signs, symptoms, and medical histories, and must:

1. **Assess** patients to gather clinical information
2. **Order** the correct diagnostic tests (ECG, Blood Test, CT Scan, Tox Screen)
3. **Triage** each patient to the correct ESI level (1 = Resuscitation → 5 = Non-Urgent)
4. **Treat** patients with appropriate medications — **avoiding fatal drug interactions**
5. **Admit** to the correct ward (Cardiology, ICU, Neurology, Surgery) or discharge home

A **deterministic grader** computes a score from **0.0 to 1.0** based on clinical accuracy.

---

## 🩺 Why Medical Triage?

- **Real-world gap**: No existing OpenEnv environment models clinical decision-making under uncertainty
- **High stakes**: Wrong actions (e.g. Penicillin to an allergic patient) have immediate, irreversible consequences
- **Partial observability**: Hidden conditions must be discovered through diagnostic testing
- **Time pressure**: Untreated critical patients deteriorate step-by-step (vitals worsen)
- **Excellent RL training signal**: Dense reward shaping + deterministic terminal scoring

### Why Medical Triage Matters for RL
This environment serves as an excellent sandbox for localized Reinforcement Learning. It features strong partial observability (hidden conditions requirements), strict time pressure, dense step-wise reward shaping, and severe penalties for unsafe actions (fatal drug interactions). By training an RL policy in this simulation, researchers can develop models that prioritize safe clinical decision-making protocols and learn generalization over rote token prediction.

---

## 📐 Observation Space

| Field | Type | Description |
|---|---|---|
| `episode_id` | string | Unique episode identifier |
| `queue_summary` | list[dict] | Patients waiting: id, vitals, top 2 symptoms |
| `active_beds_summary` | dict | Map of bed → patient state (vitals, triage, tests, treatments) |
| `alerts` | list[string] | Last 5 system alerts (critical vitals, fatal errors) |
| `current_step` | int | Step counter since last reset |
| `max_steps` | int | Maximum steps for this difficulty |
| `action_feedback` | string | Natural language result of last action |

---

## ⚡ Action Space

| `action_type` | `patient_id` | `target` | Description |
|---|---|---|---|
| `assess` | required | — | Reveal full symptoms, vitals, history |
| `order_test` | required | `ECG` \| `Blood Test` \| `CT Scan` \| `X-Ray` \| `Tox Screen` | Run a diagnostic |
| `triage` | required | `1`–`5` (string) | Assign ESI triage level |
| `treat` | required | Drug name (e.g. `Aspirin`, `Naloxone`, `IV Fluids`) | Administer treatment |
| `admit` | required | `Cardiology` \| `ICU` \| `Neurology` \| `Surgery` \| `General` | Admit to ward |
| `discharge` | required | — | Discharge patient home |
| `wait` | — | — | Pass this turn (−0.01 penalty) |

---

## 🎯 Tasks

### Task 1 — Easy 🟢: STEMI Triage
- **Patient**: P-101, 65M — crushing chest pain, diaphoresis, left arm radiation
- **Hidden condition**: STEMI (ST-elevation myocardial infarction)
- **Max steps**: 15
- **Ideal pathway**: assess → order ECG → triage level 1 → treat Aspirin → admit Cardiology
- **Expected score range**: 0.70 – 1.0

### Task 2 — Medium 🟡: Sepsis + Opioid Overdose
- **Patients**: P-102 (78F, Sepsis, **Penicillin Allergy**) + P-108 (28M, Opioid Overdose)
- **Trap**: Giving Penicillin/Amoxicillin to P-102 is a **fatal interaction** (−0.40)
- **Max steps**: 20
- **Ideal pathway**: Use Vancomycin/Ceftriaxone for P-102 (ICU) + Naloxone for P-108 (ICU)
- **Expected score range**: 0.40 – 0.80

### Task 3 — Hard 🔴: Mass Casualty
- **Patients**: P-104 (Hemorrhagic Shock), P-107 (Stroke), P-105 (9yo Status Asthmaticus)
- **Resource constraint**: 3 critical patients, 2 beds — **prioritization is required**
- **Traps**: Blood thinners (Aspirin/Heparin) on P-104 are fatal; wrong ward kills prognosis
- **Max steps**: 25
- **Ideal pathway**: Prioritize P-104 (Level 1, Surgery) → P-107 (Level 2, CT Scan, Neurology) → P-105 (Level 1, ICU)
- **Expected score range**: 0.20 – 0.65

---

## 🏆 Reward Function

### Step-level signals (partial progress for RL):
| Action | Reward |
|---|---|
| `assess` | +0.03 |
| `order_test` | +0.01 |
| `triage` | +0.03 |
| `admit` / `discharge` | +0.05 |
| `treat` (with fatal interaction) | −0.15 |
| `wait` | −0.01 |

### Terminal reward (final step = deterministic grade):
| Component | Weight |
|---|---|
| Correct triage level | +0.25 |
| Required diagnostic test | +0.25 |
| Appropriate treatment | +0.25 |
| Correct ward disposition | +0.25 |
| **Fatal drug interaction** | **−0.40** per event |
| Unnecessary tests | −0.10 each |
| Efficiency bonus (finish early) | +0.15 max |

---

## 📊 Baseline Scores

Run with `gpt-4o-mini` (temperature=0):

| Task | Difficulty | Baseline Score |
|---|---|---|
| STEMI Triage | 🟢 Easy | **0.8750** |
| Sepsis + OD | 🟡 Medium | **0.6500** |
| Mass Casualty | 🔴 Hard | **0.5200** |

---

## 🚀 Quick Start

### Option 1: Docker (recommended)
```bash
cd medical-triage-env
docker build -t medical-triage-env .
docker run -p 7860:7860 medical-triage-env
```

Then open: `http://localhost:7860/ui` for the live dashboard.

### Option 2: Local (without Docker)
```bash
cd medical-triage-env
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

---

## 🤖 Running the Inference Script

```bash
# Required environment variables
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your-api-key"

# Optional: point at a running env (default: HF Space URL)
export ENV_BASE_URL="http://localhost:7860"

# Run baseline inference against all 3 tasks
python inference.py
```

The script emits structured logs:
```
[START] task=STEMI Triage env=medical-triage-env model=gpt-4o-mini
[STEP] step=1 action={"action_type": "assess", "patient_id": "P-101"} reward=0.0300 done=false error=null
[STEP] step=2 action={"action_type": "order_test", ...} reward=0.0100 done=false error=null
...
[END] success=true steps=8 score=0.8750 rewards=[0.03, 0.01, ...]
```

---

## 🧪 Running Tests

```bash
cd medical-triage-env
python tests/test_env.py
```

Covers:
- Perfect STEMI pathway → score ≥ 0.80
- Fatal Penicillin interaction → penalized
- Naloxone for OD → positive step rewards
- `state()` method returns correct structure
- All tasks produce scores in [0.0, 1.0]
- Hemorrhagic Shock blood thinner penalty

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/reset` | Start new episode `{"difficulty": "easy"\|"medium"\|"hard"}` |
| `POST` | `/step` | Take action `{"action_type": "...", "patient_id": "...", "target": "..."}` |
| `GET` | `/state` | Current episode state (OpenEnv state() contract) |
| `GET` | `/tasks` | List all 3 tasks with metadata |
| `GET` | `/health` | Health check |
| `GET` | `/ui` | Live clinical dashboard |

---

## 📁 Project Structure

```
metacup/
├── inference.py               ← Baseline inference script (spec-compliant)
└── medical-triage-env/
    ├── models.py              ← Pydantic models: Action, Observation, TriageState
    ├── tasks.py               ← 3 scenario definitions + drug interaction database
    ├── simulator.py           ← State transition engine with vitals deterioration
    ├── grader.py              ← Deterministic 0.0–1.0 scorer
    ├── client.py              ← HTTP client for the environment
    ├── openenv.yaml           ← OpenEnv manifest (tags, tasks, action/obs schema)
    ├── requirements.txt       ← Python dependencies
    ├── Dockerfile             ← Container definition
    ├── tests/
    │   └── test_env.py        ← 6 unit tests
    └── server/
        ├── env.py             ← MedicalTriageEnv (reset/step/state)
        ├── app.py             ← FastAPI server + live dashboard + all endpoints
        └── requirements.txt   ← Server-side dependencies
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|---|---|---|
| `API_BASE_URL` | `https://api.openai.com/v1` | LLM API endpoint |
| `MODEL_NAME` | `gpt-4o-mini` | Model identifier for inference |
| `HF_TOKEN` | — | Hugging Face / OpenAI API key |
| `ENV_BASE_URL` | HF Space URL | Medical triage environment URL |
| `ENV_DIFFICULTY` | `easy` | Default difficulty for /reset |

---

## 🏥 Drug Interaction Database

The environment enforces real clinical contraindications:

| Condition | **Contraindicated Drugs** |
|---|---|
| Penicillin Allergy | Penicillin, Amoxicillin |
| Opioid Overdose | Morphine, Fentanyl, Oxycodone |
| Hemorrhagic Shock | Aspirin, Heparin, Warfarin |

Administering any contraindicated drug incurs a **−0.40 penalty** and marks the patient as unstable.

---

## 📜 License

MIT License — See [LICENSE](LICENSE)
