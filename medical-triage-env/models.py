from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import uuid

class Patient(BaseModel):
    id: str
    age: int
    vitals: Dict[str, str]
    symptoms: List[str]
    history: List[str]
    tests_ordered: List[str] = Field(default_factory=list)
    test_results: Dict[str, str] = Field(default_factory=dict)
    treatments_given: List[str] = Field(default_factory=list)
    triage_level: Optional[int] = None # 1 (Resuscitation) to 5 (Non-Urgent)
    admitted_ward: Optional[str] = None
    discharged: bool = False
    is_stable: bool = True
    hidden_condition: Optional[str] = None
    vitals_history: List[Dict[str, str]] = Field(default_factory=list)  # tracks deterioration

class IncidentState(BaseModel):
    episode_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    queue: List[Patient]
    active_beds: Dict[str, Optional[Patient]]
    current_step: int
    max_steps: int
    alerts: List[str] = Field(default_factory=list)
    fatal_errors: List[str] = Field(default_factory=list)
    score_components: Dict[str, float] = Field(default_factory=dict)
    is_done: bool = False

class IncidentObservation(BaseModel):
    episode_id: str = ""
    queue_summary: List[Dict[str, Any]]
    active_beds_summary: Dict[str, Any]
    alerts: List[str]
    current_step: int
    max_steps: int = 0
    action_feedback: str

class IncidentAction(BaseModel):
    action_type: str # "assess", "order_test", "triage", "treat", "admit", "discharge", "wait"
    patient_id: Optional[str] = None
    target: Optional[str] = None # Name of the test/treatment/ward/triage_level
