#!/usr/bin/env python3
"""Long-context needle test for MiniCPM5-2B."""
import json, random, time, urllib.request, sys

URL = 'http://127.0.0.1:8000/v1/chat/completions'
HDRS = {'Content-Type': 'application/json', 'Authorization': 'Bearer EMPTY'}

target_chars = int(sys.argv[1]) if len(sys.argv) > 1 else 110000

random.seed(42)
words = ('alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima '
         'mike november oscar papa quebec romeo sierra tango uniform victor whiskey '
         'xray yankee zulu system data stream metric vector matrix signal network').split()
filler = []
n = 0
while n < target_chars:
    line = ' '.join(random.choice(words) for _ in range(12))
    filler.append(line)
    n += len(line) + 1
body = '\n'.join(filler)

SECRET = 'BLUE FALCON 7734'
mid = len(body) // 2
text = body[:mid] + f"\n\nIMPORTANT NOTE: The secret code is {SECRET}. Remember it.\n\n" + body[mid:]
text += "\n\nQuestion: What is the secret code that was mentioned in the middle of this text? Answer with the code only."

payload = {
    'model': 'MiniCPM5-2B',
    'messages': [{'role': 'user', 'content': text}],
    'max_tokens': 64,
    'temperature': 0.0,
}
req = urllib.request.Request(URL, data=json.dumps(payload).encode(), headers=HDRS)
t0 = time.time()
d = json.loads(urllib.request.urlopen(req, timeout=3600).read())
el = time.time() - t0
u = d.get('usage', {})
msg = d['choices'][0]['message']
content = msg.get('content') or ''
print(f"OK {el:.0f}s prompt_tokens={u.get('prompt_tokens')} completion_tokens={u.get('completion_tokens')}")
print(f"finish={d['choices'][0]['finish_reason']!r}")
print('content:', repr(content[-220:]))
print('SECRET FOUND:', SECRET in content)
