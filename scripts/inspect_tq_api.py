#!/usr/bin/env python3
"""Inspect TurboQuant APIs for reset methods."""
from turboquant.store import CompressedKVStore
from turboquant.capture import KVCaptureEngine

print("=== CompressedKVStore methods ===")
for m in sorted(dir(CompressedKVStore)):
    if not m.startswith("__"):
        print(f"  {m}")

print("\n=== KVCaptureEngine methods ===")
for m in sorted(dir(KVCaptureEngine)):
    if not m.startswith("__"):
        print(f"  {m}")
