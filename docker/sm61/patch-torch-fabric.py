#!/usr/bin/env python3
"""
patch-torch-fabric.py — disable the PyTorch NVML fabric path for Pascal.

The container's nvml.h (libnvidia-ml-dev) predates the NVML fabric API
(nvmlDeviceGetGpuFabricInfoV / nvmlGpuFabricInfoV_t), so torch 2.13's fabric
code fails to compile. When the header lacks the API:

- driver_api.h:       disable the `CUDA_VERSION >= 12040` member declaration
- PeerToPeerAccess.cpp (c10/cuda/): disable the same guards AND stub
  get_fabric_access (it has no version guard of its own in 2.13)

Callers (CUDACachingAllocator) fall back to the fd-handle exchange path.
Fail-fast via SystemExit on missing anchors.
"""
from pathlib import Path

NVML_HEADER = Path("/usr/local/cuda/include/nvml.h")
DRIVER_API = Path("/tmp/pytorch/c10/cuda/driver_api.h")
P2P_CPP = Path("/tmp/pytorch/c10/cuda/PeerToPeerAccess.cpp")

if NVML_HEADER.exists() and "nvmlDeviceGetGpuFabricInfoV" in NVML_HEADER.read_text():
    print("Installed nvml.h provides nvmlDeviceGetGpuFabricInfoV; no patch needed")
    raise SystemExit(0)

print("Installed nvml.h lacks nvmlDeviceGetGpuFabricInfoV; disabling PyTorch NVML fabric path")

# 1. driver_api.h: disable the member declaration guard
d = DRIVER_API.read_text()
old_guard = "CUDA_VERSION >= 12040"
if old_guard not in d:
    raise SystemExit("driver_api.h: CUDA_VERSION >= 12040 guard not found")
d = d.replace(old_guard, "CUDA_VERSION >= 12040 && 0")
DRIVER_API.write_text(d)
print("patched driver_api.h: fabric member declaration disabled")

# 2. PeerToPeerAccess.cpp: disable guards + stub get_fabric_access
p = P2P_CPP.read_text()
if "CUDA_VERSION >= 12040" in p:
    p = p.replace("CUDA_VERSION >= 12040", "CUDA_VERSION >= 12040 && 0")
    print("patched PeerToPeerAccess.cpp: fabric guards disabled")

old_a = "bool get_fabric_access(c10::DeviceIndex dev) {\n"
new_a = "bool get_fabric_access(c10::DeviceIndex dev) {\n#if 0  // Pascal: fabric NVML API unavailable in container nvml.h\n"
old_b = "}\n\nint get_fabric_clique_id(c10::DeviceIndex dev) {\n"
new_b = "}\n#endif\n  return false;\n}\n\nint get_fabric_clique_id(c10::DeviceIndex dev) {\n"
if old_a not in p or old_b not in p:
    raise SystemExit("fabric stub anchors not found in c10/cuda/PeerToPeerAccess.cpp")
p = p.replace(old_a, new_a, 1).replace(old_b, new_b, 1)
P2P_CPP.write_text(p)
print("patched get_fabric_access -> stub (Pascal)")

# 3. torch 2.13 upstream bug: torch_cuda's symm_mem sources miss the
#    -U__CUDA_NO_HALF_OPERATORS__ that torch_nvshmem gets (caffe2/CMakeLists:1060),
#    so OpSum<__half> (build/nccl/include/nccl_device/impl/reduce_copy__types.h:26,
#    "return a + b") fails to compile whenever torch's global
#    -D__CUDA_NO_HALF_OPERATORS__ is active. Mirror torch's own fix.
CMAKE_LISTS = Path("/tmp/pytorch/caffe2/CMakeLists.txt")
c = CMAKE_LISTS.read_text()
old_flags = 'PROPERTIES COMPILE_FLAGS "-DPYTORCH_C10_DRIVER_API_SUPPORTED=1"'
new_flags = (
    'PROPERTIES COMPILE_FLAGS "-DPYTORCH_C10_DRIVER_API_SUPPORTED=1 '
    '-U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ '
    '-U__CUDA_NO_HALF2_OPERATORS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__"'
)
if old_flags not in c:
    raise SystemExit("caffe2/CMakeLists.txt COMPILE_FLAGS anchor not found")
CMAKE_LISTS.write_text(c.replace(old_flags, new_flags, 1))
print("patched caffe2/CMakeLists.txt: symm_mem files get -U half/bf16 operators")
