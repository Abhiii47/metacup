import subprocess
import time
import sys

server = subprocess.Popen([sys.executable, "-m", "uvicorn", "server.app:app", "--host", "127.0.0.1", "--port", "7860"])
time.sleep(4)
res = subprocess.run([sys.executable, "inference.py", "--agent", "pytorch"], capture_output=True, text=True)
print("=== INFERENCE OUTPUT ===")
print(res.stdout)
if res.stderr:
    print("=== INFERENCE ERROR ===")
    print(res.stderr)
server.terminate()
