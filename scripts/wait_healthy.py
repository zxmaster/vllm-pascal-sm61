"""Wait for vLLM /health endpoint to return 200."""
import sys
import time
import urllib.request

for i in range(60):
    try:
        r = urllib.request.urlopen("http://localhost:8000/health", timeout=5)
        if r.status == 200:
            print("READY")
            sys.exit(0)
    except Exception:
        pass
    print(f"waiting... ({i+1})")
    time.sleep(10)

print("TIMEOUT")
sys.exit(1)
