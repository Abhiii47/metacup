import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from server.env import MedicalTriageEnv
from models import IncidentAction

def test_easy_scenario():
    env = MedicalTriageEnv()
    obs = env.reset(difficulty="easy")
    
    # Assess patient
    obs, reward, done, _ = env.step(IncidentAction(action_type="assess", patient_id="P-101"))
    assert not done
    
    # Order ECG test
    obs, reward, done, _ = env.step(IncidentAction(action_type="order_test", patient_id="P-101", target="ECG"))
    
    # Triage Level 1
    obs, reward, done, _ = env.step(IncidentAction(action_type="triage", patient_id="P-101", target="1"))
    
    # Administer Aspirin
    obs, reward, done, _ = env.step(IncidentAction(action_type="treat", patient_id="P-101", target="Aspirin"))
    
    # Admit to Cardiology
    obs, reward, done, _ = env.step(IncidentAction(action_type="admit", patient_id="P-101", target="Cardiology"))
    
    # Step wait until done
    for _ in range(15):
        obs, reward, done, _ = env.step(IncidentAction(action_type="wait"))
        if done:
            break
            
    assert done
    print(f"Easy Scenario Reward: {reward}")
    assert reward >= 0.79 # Should get a perfect or near perfect score

def test_medium_fatal_error():
    env = MedicalTriageEnv()
    obs = env.reset(difficulty="medium")
    
    # Give Penicillin to Patient P-102 (who has Penicillin Allergy)
    env.step(IncidentAction(action_type="treat", patient_id="P-102", target="Penicillin"))
    
    # Wait to end
    for _ in range(20):
         obs, reward, done, _ = env.step(IncidentAction(action_type="wait"))
         if done:
             break
             
    # Fatal error should significantly reduce reward
    assert len(env.state.fatal_errors) > 0
    print(f"Medium Scenario with Fatal Error Reward: {reward}")
    assert reward < 0.5

if __name__ == "__main__":
    test_easy_scenario()
    test_medium_fatal_error()
    print("All tests passed.")
