#!/usr/bin/env python3
from gguf import GGUFReader
r = GGUFReader('./volumes/models/Qwen3.5-4B-GGUF/Qwen3.5-4B-Q4_K_M.gguf')
for t in r.tensors:
    if 'ssm' in t.name or 'blk.0.' in t.name:
        print(f'{t.name}: shape={t.shape}, type={t.tensor_type}')
