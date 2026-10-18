#!/usr/bin/env python3
"""
patch-vllm-sm61-v028.py — build-time patches for vLLM v0.28.0 on Pascal (sm_61).

v0.28 migration: minimal set — only FlashAttention extension disable.
Everything else (GGUF, GDN fallbacks, SDPA) is handled natively by the
vllm-gguf-plugin and runtime patches in start-vllm-v028.sh.

Fail-fast: every replace_once raises SystemExit with a descriptive error if
the anchor is missing, so a mismatched upstream version is detected at build
time instead of at runtime.
"""
from pathlib import Path

VLLM_ROOT = Path("/tmp/vllm")


def replace_once(path: Path, old: str, new: str, error: str) -> None:
    text = path.read_text()
    if old in text:
        path.write_text(text.replace(old, new, 1))
        print(f"patched {path.name}: {error}")
        return
    if new in text:
        print(f"patched {path.name} (already applied): {error}")
        return
    raise SystemExit(error)


# 1. Disable the flash-attention extension (FA2/FA3 require sm_80+; the
#    extension would fail to build or crash on sm_61).
setup_path = VLLM_ROOT / "setup.py"
setup_text = setup_path.read_text()
old_fa = 'if _is_cuda():\n    ext_modules.append(CMakeExtension(name="vllm.vllm_flash_attn._vllm_fa2_C"))'
new_fa = 'if False and _is_cuda():\n    ext_modules.append(CMakeExtension(name="vllm.vllm_flash_attn._vllm_fa2_C"))'
if old_fa in setup_text:
    setup_path.write_text(setup_text.replace(old_fa, new_fa, 1))
    print("patched setup.py: flash-attention extension disabled")
else:
    # v0.28 keeps the same two-line block; fail loudly if upstream changed it.
    if "vllm.vllm_flash_attn._vllm_fa2_C" not in setup_text:
        raise SystemExit("failed to locate flash-attention extension block in setup.py")
    print("setup.py: flash-attention extension block already disabled")

# 2. Relax the vllm_flash_attn import guard so model loading can fall back to
#    non-flash-attention backends instead of aborting.
fa_init_path = VLLM_ROOT / "vllm/vllm_flash_attn/__init__.py"
fa_text = fa_init_path.read_text()
old_guard = '''if not (FA2_AVAILABLE or FA3_AVAILABLE):
    raise ImportError(
        "vllm.vllm_flash_attn requires the CUDA flash attention extensions "
        "(_vllm_fa2_C or _vllm_fa3_C). On ROCm, use upstream flash_attn."
    )'''
new_guard = '''if not (FA2_AVAILABLE or FA3_AVAILABLE):
    # Experimental Pascal build: allow import so backend selection can fall back
    # to non-flash-attention implementations instead of aborting model loading.
    pass'''
if old_guard in fa_text:
    fa_init_path.write_text(fa_text.replace(old_guard, new_guard, 1))
    print("patched vllm_flash_attn/__init__.py: import guard relaxed")
elif new_guard in fa_text:
    print("vllm_flash_attn/__init__.py: import guard already relaxed")
else:
    raise SystemExit("failed to locate flash-attention import guard")

# 3. v0.28 build gate rejects pre-7.5 archs (CMakeLists.txt CUDA_SUPPORTED_ARCHS
#    by nvcc version). Allow sm_61 (Pascal) — kernels that cannot compile for
#    sm_61 will fail at compile time with their own clear errors.
cmake_path = VLLM_ROOT / "CMakeLists.txt"
cmake_text = cmake_path.read_text()
n_supported = cmake_text.count("set(CUDA_SUPPORTED_ARCHS ")
if n_supported == 0:
    raise SystemExit("failed to locate CUDA_SUPPORTED_ARCHS in CMakeLists.txt")
patched_text = cmake_text.replace(
    'set(CUDA_SUPPORTED_ARCHS "7.', 'set(CUDA_SUPPORTED_ARCHS "6.1;7.', 
)
if patched_text == cmake_text:
    raise SystemExit("CUDA_SUPPORTED_ARCHS anchors did not match")
cmake_path.write_text(patched_text)
print(f"patched CMakeLists.txt: sm_61 added to {n_supported} CUDA_SUPPORTED_ARCHS branches")

# 4. DCP (direct context parallelism) kernels use PTX .acquire/.release which
#    require sm_70+; they are multi-GPU-only and unused on a single Pascal GPU.
#    Remove them from VLLM_STABLE_EXT_SRC. Re-read the file (patch 3 rewrote it).
cmake_text = cmake_path.read_text()
dcp_anchor = '''    list(APPEND VLLM_STABLE_EXT_SRC
      "csrc/libtorch_stable/attention/dcp_utils/dcp_direct_a2a_lse_reduce.cu"
      "csrc/libtorch_stable/attention/dcp_utils/dcp_direct_kv_gather.cu"
      "csrc/libtorch_stable/attention/dcp_utils/dcp_direct_q_gather.cu")
'''
if dcp_anchor in cmake_text:
    cmake_text = cmake_text.replace(dcp_anchor, "", 1)
    cmake_path.write_text(cmake_text)
    print("patched CMakeLists.txt: DCP kernels (sm_70+ PTX) excluded from build")
else:
    raise SystemExit("failed to locate DCP source list in CMakeLists.txt")
# 5. moe_wna16.cu (upstream bug): the pre-sm_80 fallback path calls
#    atomicAdd(__half*/__nv_bfloat16*) which requires sm_70+ (Pascal sm_61
#    has no such overload). Add CAS-based 16-bit atomic add helpers (mirror of
#    vllm::gptq::atomicAdd_half in quantization/gptq/compat.cuh) and use them.
moe_path = VLLM_ROOT / "csrc/libtorch_stable/moe/moe_wna16.cu"
moe_text = moe_path.read_text()
moe_anchor = "#define DIVIDE(x, size) (((x) + (size) - 1) / (size))\n"
moe_helpers = moe_anchor + '''
// Pascal (sm_61) fix: atomicAdd for 16-bit types requires sm_70+; CAS-based
// equivalents (mirror of vllm::gptq::atomicAdd_half for CC < 7.x).
__device__ __forceinline__ void atomic_add_fp16(half* address, half val) {
  unsigned int* address_as_ui =
      (unsigned int*)((char*)address - ((size_t)address & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;
  do {
    assumed = old;
    __half_raw hsum;
    hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    half tmpres = __hadd(hsum, val);
    hsum = __half_raw(tmpres);
    old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16)
                              : (old & 0xffff0000) | hsum.x;
    old = atomicCAS(address_as_ui, assumed, old);
  } while (assumed != old);
}
__device__ __forceinline__ void atomic_add_fp16(__nv_bfloat16* address,
                                                __nv_bfloat16 val) {
  unsigned int* address_as_ui =
      (unsigned int*)((char*)address - ((size_t)address & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;
  do {
    assumed = old;
    unsigned short cur = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
    __nv_bfloat16 tmpres =
        __hadd(__ushort_as_bfloat16(cur), val);
    unsigned short res = __bfloat16_as_ushort(tmpres);
    old = (size_t)address & 2 ? (old & 0xffff) | (res << 16)
                              : (old & 0xffff0000) | res;
    old = atomicCAS(address_as_ui, assumed, old);
  } while (assumed != old);
}
'''
if moe_anchor not in moe_text:
    raise SystemExit("failed to locate moe_wna16.cu DIVIDE anchor")
moe_text = moe_text.replace(moe_anchor, moe_helpers, 1)
old_atomic = """      atomicAdd(&output[token_index * size_n + offset_n],
                Dtype::float2num(res[m]));"""
new_atomic = """      atomic_add_fp16(&output[token_index * size_n + offset_n],
                       Dtype::float2num(res[m]));"""
if old_atomic not in moe_text:
    raise SystemExit("failed to locate moe_wna16.cu atomicAdd call")
moe_text = moe_text.replace(old_atomic, new_atomic, 1)
moe_path.write_text(moe_text)
print("patched moe_wna16.cu: CAS-based 16-bit atomic add for Pascal")

print("patch-vllm-sm61-v028.py: all patches applied")
