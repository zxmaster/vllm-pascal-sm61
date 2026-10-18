#!/bin/sh
set -eu

MAX_BYTES=${GIT_HYGIENE_MAX_BYTES:-52428800}

printf '== Git status ==\n'
git status --short --branch

printf '\n== Large files visible to Git (> %s bytes) ==\n' "$MAX_BYTES"
python3 - "$MAX_BYTES" <<'PY'
import os
import subprocess
import sys

limit = int(sys.argv[1])
proc = subprocess.run(
    ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
    check=True,
    stdout=subprocess.PIPE,
)
paths = [p.decode("utf-8", "surrogateescape") for p in proc.stdout.split(b"\0") if p]
large = []
for path in paths:
    try:
        size = os.path.getsize(path)
    except OSError:
        continue
    if size > limit:
        large.append((size, path))

if not large:
    print("none")
else:
    for size, path in sorted(large, reverse=True):
        print(f"{size:>12}  {path}")
    sys.exit(1)
PY

printf '\n== Secret/config filename candidates visible to Git ==\n'
if git ls-files --cached --others --exclude-standard \
  | grep -Ei '(^|/)(pass|password|secret|token|.*key|\.env($|\.))' >/tmp/git-hygiene-candidates.$$; then
  cat /tmp/git-hygiene-candidates.$$
  rm -f /tmp/git-hygiene-candidates.$$
  printf '\nReview these paths before committing.\n'
else
  rm -f /tmp/git-hygiene-candidates.$$
  printf 'none\n'
fi

printf '\n== Staged diff stat ==\n'
git diff --cached --stat
