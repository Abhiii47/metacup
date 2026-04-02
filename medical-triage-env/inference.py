import os
import json
import time
import argparse

from client import SimpleEnvClient

def run_llm_agent(client=None):
    from openai import OpenAI
    print("Running LLM Agent...")
    try:
        openai_client = OpenAI() # expects OPENAI_API_KEY in environment
    except Exception as e:
        print("OpenAI client failed to initialize. Ensure OPENAI_API_KEY is set.", e)
        return

    obs = client.reset(difficulty="medium")
    
    for step_num in range(20):
        if obs.get("current_step", 100) >= 20:
            break
            
        print(f"--- Step {step_num} ---")
        prompt = f"Queue: {obs.get('queue_summary')}, Beds: {obs.get('active_beds_summary')}, Alerts: {obs.get('alerts')}"
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an ER triage nurse. Choose an action from: assess, order_test, treat, triage, admit, discharge, wait. Return JSON format: {\"action_type\": \"...\", \"patient_id\": \"...\", \"target\": \"...\"}"},
                    {"role": "user", "content": prompt}
                ]
            )
            # Extract JSON from response
            content = response.choices[0].message.content
            # Quick parsing mock
            if "{" in content:
                json_str = content[content.find("{"):content.rfind("}")+1]
                action = json.loads(json_str)
            else:
                action = {"action_type": "wait"}
        except Exception as e:
            print("LLM Error or JSON parse error:", e)
            action = {"action_type": "wait"}

        print(f"Action Selection: {action}")
        obs = client.step(action)
        
def run_pytorch_agent(client=None):
    print("Running PyTorch Policy Agent...")
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("PyTorch is not installed. Run: pip install torch")
        return
        
    class TriagePolicy(nn.Module):
        def __init__(self, obs_size, action_size):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_size, 64),
                nn.ReLU(),
                nn.Linear(64, action_size)
            )
            
        def forward(self, x):
            return self.net(x)
            
    # Mock sizes for demonstration of PyTorch integration
    policy = TriagePolicy(obs_size=16, action_size=5)
    
    obs = client.reset(difficulty="medium")
    for _ in range(15):
        # Fake state embedding from observation
        state_tensor = torch.randn(1, 16)
        action_logits = policy(state_tensor)
        action_idx = torch.argmax(action_logits).item()
        
        action_map = ["assess", "order_test", "treat", "triage", "wait"]
        chosen = action_map[action_idx]
        print(f"PyTorch agent scored action {action_idx}: {chosen}")
        
        action = {"action_type": chosen, "patient_id": "P-102"}
        if chosen == "order_test": action["target"] = "Blood Test"
        if chosen == "treat": action["target"] = "Fluids"
        if chosen == "triage": action["target"] = "2"
        
        obs = client.step(action)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["llm", "pytorch"], default="llm", help="Which agent policy to run")
    args = parser.parse_args()
    
    c = SimpleEnvClient(url="http://127.0.0.1:7860")
    if args.agent == "pytorch":
        run_pytorch_agent(c)
    else:
        run_llm_agent(c)
