import json
import sys
import urllib.error
import urllib.request


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: http_probe.py <url> <request-json-path>", file=sys.stderr)
        return 2

    url = sys.argv[1]
    request_path = sys.argv[2]

    with open(request_path, "rb") as request_file:
        payload = request_file.read()

    request = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer EMPTY",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=240) as response:
            body = response.read().decode("utf-8", "replace")
            print(json.dumps({"status": response.status, "body": body}))
            return 0
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", "replace")
        print(json.dumps({"status": error.code, "body": body}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
