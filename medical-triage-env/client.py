import httpx
import json

class SimpleEnvClient:
    """A minimal mock for interacting with the raw FastAPI endpoints if openenv WebSocket fails."""
    def __init__(self, url="http://127.0.0.1:7860"):
        self.url = url
        
    def reset(self, difficulty="easy"):
        res = httpx.post(f"{self.url}/reset", json={"difficulty": difficulty})
        return res.json()
        
    def step(self, action_dict):
        res = httpx.post(f"{self.url}/step", json=action_dict)
        return res.json()
