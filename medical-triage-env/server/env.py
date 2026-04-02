import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from models import IncidentState, IncidentObservation, IncidentAction, Patient
from tasks import get_scenario
from simulator import Simulator
from grader import grade

class MedicalTriageEnv:
    def __init__(self):
        self.state = None
        self.simulator = None
        self.all_patients_history = []
        self.difficulty = "easy"

    def reset(self, **kwargs) -> IncidentObservation:
        env_diff = os.getenv("ENV_DIFFICULTY", "easy")
        self.difficulty = kwargs.get("difficulty", env_diff)
        scenario = get_scenario(self.difficulty)
        
        import random
        patients = [Patient(**p) for p in scenario["patients"]]
        random.shuffle(patients)
        
        self.all_patients_history = [p.model_copy(deep=True) for p in patients]
        
        self.state = IncidentState(
            queue=patients,
            active_beds={b: None for b in scenario["beds"]},
            current_step=0,
            max_steps=scenario["max_steps"]
        )
        self.simulator = Simulator(self.state)
        # Advance initial queue
        self.simulator._update_time()
        return self.simulator.get_observation()

    def step(self, action: IncidentAction):
        if not self.simulator:
            raise RuntimeError("Environment has not been reset.")

        self.simulator.step(action)
        obs = self.simulator.get_observation()
        
        # update the historical records for grading
        for p in self.all_patients_history:
            current = self.simulator._get_patient(p.id)
            if current:
                p.triage_level = current.triage_level
                p.tests_ordered = current.tests_ordered
                p.test_results = current.test_results
                p.treatments_given = current.treatments_given
                p.admitted_ward = current.admitted_ward
                p.discharged = current.discharged
                p.is_stable = current.is_stable

        # --- PARTIAL REWARD SIGNAL (at every step, for RL training) ---
        step_reward = 0.0

        if action.action_type == "order_test" and action.target:
            # Small positive reward for ordering a test (agent is actively investigating)
            step_reward += 0.01

        if action.action_type == "triage" and action.patient_id:
            # Reward immediately if triage level is set (positive action)
            step_reward += 0.02

        if action.action_type == "treat" and action.patient_id:
            # Check if a fatal interaction was just triggered — immediate penalty
            fatal_before = len(self.state.fatal_errors)
            if fatal_before > 0:
                step_reward -= 0.15  # immediate negative signal for lethal mistake

        if action.action_type in ("admit", "discharge") and action.patient_id:
            # Reward for completing patient care (positive disposition)
            step_reward += 0.05

        if action.action_type == "wait":
            # Small time penalty to discourage passive behavior
            step_reward -= 0.01

        done = self.state.is_done
        if done:
            # Final episode reward replaces partial signals for the terminal step
            final_reward = grade(self.state, self.all_patients_history)
            return obs, final_reward, done, {}

        return obs, round(step_reward, 4), done, {}

    def get_state(self) -> IncidentState:
        return self.state

