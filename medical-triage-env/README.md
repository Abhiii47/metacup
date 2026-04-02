---
title: Medical Triage OpenEnv
emoji: 🏥
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---

# 🏥 Medical Triage Nurse — OpenEnv

> **Meta PyTorch Hackathon submission.** An OpenEnv environment where an AI agent acts as a clinical ER Triage Nurse, managing emergency patients under time pressure.

---

## What It Does

The agent receives a queue of patients with real vital signs, symptoms, and medical histories. It must:

1. **Assess** patients and order the right diagnostic tests
2. **Triage** each patient to the correct emergency level (1–5)
3. **Treat** correctly — avoiding fatal drug interactions
4. **Admit** or discharge to the correct ward

A final score between **0.0 and 1.0** is computed deterministically by the grader.

---

## Action Space

| Action | Parameters | Description |
|---|---|---|
| `assess` | `patient_id` | Review symptoms, vitals, history |
| `order_test` | `patient_id`, `target` (ECG/Blood Test/CT Scan/X-Ray/Tox Screen) | Diagnostic tests |
| `treat` | `patient_id`, `target` (drug name) | Administer treatment |
| `triage` | `patient_id`, `target` (1–5) | Assign triage level |
| `admit` | `patient_id`, `target` (ward name) | Admit to hospital ward |
| `discharge` | `patient_id` | Discharge patient home |
| `wait` | — | Pass turn |

---

## Task Difficulties

| # | Difficulty | Scenario | Root Cause | Max Steps |
|---|---|---|---|---|
| 1 | 🟢 Easy | Single patient: classic heart attack | STEMI | 15 |
| 2 | 🟡 Medium | Sepsis + Opioid Overdose; Penicillin allergy trap | Sepsis / Drug Interaction | 20 |
| 3 | 🔴 Hard | Mass casualty: Hemorrhagic Shock, Stroke, Asthmatic child | Prioritization | 25 |

---

## Reward Shaping

| Signal | Points |
|---|---|
| Correct triage level | +0.30 |
| Correct diagnostic test ordered | +0.30 |
| Correct treatment given | +0.20 |
| Correct ward disposition | +0.20 |
| Efficiency bonus (finish faster) | up to +0.20 |
| Fatal drug interaction | −0.50 per event |
| Unnecessary test ordered | −0.05 per test |

---

## Running Locally

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Running the PyTorch Policy Agent
```bash
python inference.py --agent pytorch
```

### Running the LLM Agent (requires OPENAI_API_KEY)
```bash
export OPENAI_API_KEY=sk-...
python inference.py --agent llm
```

### Running Tests
```bash
python tests/test_env.py
```

### Environment Variables
| Variable | Default | Description |
|---|---|---|
| `ENV_DIFFICULTY` | `easy` | Sets default scenario difficulty |
| `OPENAI_API_KEY` | — | Required for LLM agent mode |

---

## Project Structure

```
medical-triage-env/
├── models.py              ← Pydantic Action, Observation, State
├── tasks.py               ← 3 scenario definitions + drug interaction DB
├── simulator.py           ← State transition engine
├── grader.py              ← Deterministic 0.0–1.0 scorer
├── inference.py           ← PyTorch + LLM baseline agents
├── client.py              ← HTTP client for the environment
├── openenv.yaml           ← OpenEnv manifest
├── server/
│   ├── env.py             ← MedicalTriageEnv (reset/step/state)
│   ├── app.py             ← FastAPI server + live dashboard UI
│   └── Dockerfile
└── tests/test_env.py      ← Unit tests covering Easy + Fatal Error
```
