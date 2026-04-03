"""
inference.py — Medical Triage Nurse OpenEnv
Meta PyTorch Hackathon Submission

Baseline inference script that runs an LLM agent against all 3 tasks
and emits structured [START] / [STEP] / [END] logs for automated evaluation.

Required environment variables:
  API_BASE_URL   — LLM endpoint base URL (e.g. https://api.openai.com/v1)
  MODEL_NAME     — Model identifier (e.g. gpt-4o-mini)
  HF_TOKEN       — API key / Hugging Face token

Optional environment variables:
  ENV_BASE_URL   — Medical Triage env base URL (defaults to localhost:7860)
  MAX_STEPS      — Override max steps per task (default uses task max)
"""

import os
import sys
import json
import time
import asyncio
from typing import List, Optional

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration — read from environment variables
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_KEY: str      = os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))

ENV_BASE_URL: str = os.environ.get(
    "ENV_BASE_URL",
    "https://openenv-medical-triage-env.hf.space"
)

BENCHMARK: str  = "medical-triage-env"
TEMPERATURE: float = 0.0
MAX_TOKENS: int    = 512

TASKS = [
    {
        "id": "easy",
        "name": "STEMI Triage",
        "difficulty": "easy",
        "max_steps": 15,
        "max_total_reward": 1.0,
        "success_threshold": 0.60,
    },
    {
        "id": "medium",
        "name": "Sepsis + Opioid Overdose",
        "difficulty": "medium",
        "max_steps": 20,
        "max_total_reward": 1.0,
        "success_threshold": 0.45,
    },
    {
        "id": "hard",
        "name": "Mass Casualty — Hemorrhagic Shock, Stroke, Asthmatic Child",
        "difficulty": "hard",
        "max_steps": 25,
        "max_total_reward": 1.0,
        "success_threshold": 0.30,
    },
]

# ---------------------------------------------------------------------------
# Structured logging — exact [START] / [STEP] / [END] format
# ---------------------------------------------------------------------------

def log_start(*, task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(*, step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_str = f" error={error}" if error else " error=null"
    print(
        f"[STEP] step={step} action={json.dumps(action)} "
        f"reward={reward:.4f} done={str(done).lower()}{error_str}",
        flush=True,
    )


def log_end(*, success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = json.dumps([round(r, 4) for r in rewards])
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.4f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# System & user prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert emergency room triage nurse making rapid clinical decisions.

Your available actions:
  - assess      : patient_id required. Review all patient info.
  - order_test  : patient_id + target required. Targets: ECG, Blood Test, CT Scan, X-Ray, Tox Screen
  - triage      : patient_id + target required. Target: 1=Resuscitation, 2=Emergent, 3=Urgent, 4=Less Urgent, 5=Non-Urgent
  - treat       : patient_id + target required. Drug/treatment name (e.g. Aspirin, Naloxone, IV Fluids, Albuterol)
  - admit       : patient_id + target required. Ward: Cardiology, ICU, Neurology, Surgery, General
  - discharge   : patient_id required. Send patient home.
  - wait        : No parameters. Pass this turn.

CRITICAL RULES:
  - Never give Penicillin/Amoxicillin to patients with Penicillin Allergy (fatal)
  - Never give Morphine/Fentanyl/Oxycodone to Opioid Overdose patients (fatal)
  - Never give Aspirin/Heparin/Warfarin to Hemorrhagic Shock patients (fatal)

Respond ONLY with a valid JSON object:
{"action_type": "...", "patient_id": "P-XXX", "target": "..."}
For wait: {"action_type": "wait"}
No markdown, no explanation."""


def build_prompt(step: int, obs: dict, last_reward: float, history: List[str]) -> str:
    queue = obs.get("queue_summary", [])
    beds = obs.get("active_beds_summary", {})
    alerts = obs.get("alerts", [])
    feedback = obs.get("action_feedback", "")

    queue_str = json.dumps(queue, indent=2) if queue else "Empty"
    beds_str = json.dumps(beds, indent=2)

    hist_str = "\n".join(history[-8:]) if history else "None yet"

    return f"""=== Step {step} ===
Last Feedback: {feedback}
Last Reward: {last_reward:+.4f}

WAITING QUEUE:
{queue_str}

ACTIVE BEDS:
{beds_str}

ALERTS (last 5):
{chr(10).join(alerts) if alerts else 'None'}

RECENT ACTIONS:
{hist_str}

Choose your next action as JSON:"""


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def get_action(client: OpenAI, step: int, obs: dict, last_reward: float, history: List[str]) -> dict:
    prompt = build_prompt(step, obs, last_reward, history)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
        content = (response.choices[0].message.content or "").strip()
        # Extract JSON from the response
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(content[start:end])
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", flush=True)
    return {"action_type": "wait"}


# ---------------------------------------------------------------------------
# Single-task runner
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, http: httpx.Client, task: dict) -> float:
    task_id    = task["id"]
    task_name  = task["name"]
    max_steps  = task["max_steps"]
    success_th = task["success_threshold"]

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    history:  List[str]  = []
    rewards:  List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_reward = 0.0

    try:
        # Reset the environment for this task
        resp = http.post(f"{ENV_BASE_URL}/reset", json={"difficulty": task_id}, timeout=30)
        resp.raise_for_status()
        obs = resp.json()

        for step in range(1, max_steps + 1):
            done = obs.get("done", False)
            if done:
                break

            action = get_action(client, step, obs, last_reward, history)
            action_str = json.dumps(action)

            error_msg = None
            try:
                step_resp = http.post(f"{ENV_BASE_URL}/step", json=action, timeout=30)
                step_resp.raise_for_status()
                obs = step_resp.json()
            except Exception as e:
                error_msg = str(e)
                print(f"[DEBUG] Step {step} HTTP error: {e}", flush=True)
                obs = {"done": True, "reward": 0.0, "current_step": step, "action_feedback": "error"}

            reward = float(obs.get("reward", 0.0))
            done   = bool(obs.get("done", False))
            last_reward = reward
            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=error_msg)
            history.append(
                f"Step {step}: {action.get('action_type')}({action.get('patient_id','')}"
                f" {action.get('target','')}) → reward={reward:+.4f}"
            )

            if done:
                break

        # The final reward from the env IS the graded score [0,1]
        score = rewards[-1] if rewards else 0.0
        score = max(0.0, min(1.0, score))
        success = score >= success_th

    except Exception as exc:
        print(f"[DEBUG] Task {task_id} failed: {exc}", flush=True)

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return score


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    if not API_KEY:
        print("[ERROR] HF_TOKEN or OPENAI_API_KEY environment variable not set.", flush=True)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    print(f"[INFO] Connecting to environment at: {ENV_BASE_URL}", flush=True)
    print(f"[INFO] Using model: {MODEL_NAME}", flush=True)
    print(f"[INFO] Running {len(TASKS)} tasks...\n", flush=True)

    all_scores: List[float] = []

    with httpx.Client() as http:
        # Verify env is reachable
        try:
            health = http.get(f"{ENV_BASE_URL}/health", timeout=10)
            print(f"[INFO] Health check: {health.json()}", flush=True)
        except Exception as e:
            print(f"[WARN] Health check failed: {e}", flush=True)

        for task in TASKS:
            print(f"\n{'='*60}", flush=True)
            print(f"[INFO] Starting task: {task['name']} (difficulty={task['id']})", flush=True)
            print(f"{'='*60}", flush=True)

            score = run_task(client, http, task)
            all_scores.append(score)

            print(f"\n[INFO] Task '{task['id']}' completed. Score: {score:.4f}", flush=True)
            time.sleep(1)  # Brief pause between tasks

    print(f"\n{'='*60}", flush=True)
    print("[INFO] All tasks complete.", flush=True)
    print(f"[INFO] Scores: easy={all_scores[0]:.4f}  medium={all_scores[1]:.4f}  hard={all_scores[2]:.4f}", flush=True)
    print(f"[INFO] Average score: {sum(all_scores)/len(all_scores):.4f}", flush=True)


if __name__ == "__main__":
    main()
