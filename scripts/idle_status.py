#!/usr/bin/env python3
import json
import os
import ssl
import sys
import urllib.error
import urllib.request


def main() -> int:
    base_url = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("VLLM_URL", "http://vllm.test")
    url = base_url.rstrip("/") + "/healthz"
    context = ssl._create_unverified_context()

    try:
        with urllib.request.urlopen(url, timeout=30, context=context) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", "replace")
        print(json.dumps({"url": url, "status": error.code, "body": body}, indent=2))
        return 1
    except Exception as error:
        print(json.dumps({"url": url, "error": str(error)}, indent=2))
        return 1

    print(json.dumps({"url": url, **payload}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
