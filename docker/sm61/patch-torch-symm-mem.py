#!/usr/bin/env python3
"""
patch-torch-symm-mem.py — exclude NCCL symm_mem (NVLink/SHARP) from torch 2.13.

Pascal (sm_61) has no NVLink/SHARP; the vendored NCCL device codegen fails to
compile with nvcc 12.9 (reduce_copy__types.h: "no operator +"). Remove the
symm_mem sources and neutralize the consumers:

1. build_variables.bzl — drop symm_mem entries from
   libtorch_cuda_distributed_extra_sources
2. torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp — drop the intra_node_comm
   / symm_mem include and usage
3. torch/csrc/distributed/c10d/cuda/StreamBlock.cu — drop the symm_mem include
4. torch/csrc/distributed/c10d/init.cpp — drop symm_mem bindings + includes

The Python-side torch.distributed.symmetric_memory module becomes unavailable
(acceptable: single-GPU Pascal stack, vLLM does not use it).
"""

import re
import sys
from pathlib import Path

TORCH = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/pytorch")
BZL = TORCH / "build_variables.bzl"
PNCCL = TORCH / "torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp"
STREAM = TORCH / "torch/csrc/distributed/c10d/cuda/StreamBlock.cu"
INIT = TORCH / "torch/csrc/distributed/c10d/init.cpp"

errors = []


def must(fn, msg):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        errors.append(f"{msg}: {e}")


def patch_bzl():
    t = BZL.read_text()
    # drop symm_mem entries inside the extra_sources list
    new_lines = []
    for line in t.splitlines():
        if "/c10d/symm_mem/" in line and "distributed_extra" not in line:
            continue
        new_lines.append(line)
    out = "\n".join(new_lines) + "\n"
    assert "symm_mem" not in out, "symm_mem still present in bzl"
    BZL.write_text(out)


def patch_pncl():
    t = PNCCL.read_text()
    t = t.replace(
        '#include <torch/csrc/distributed/c10d/symm_mem/nccl_devcomm_manager.hpp>\n',
        "",
    )
    # neutralize intra_node_comm usage
    t = re.sub(
        r"c10::intrusive_ptr<intra_node_comm::IntraNodeComm> ProcessGroupNCCL::\n\s*getIntraNodeComm\([^)]*\)\s*\{.*?\n\s*\}\n",
        "c10::intrusive_ptr<intra_node_comm::IntraNodeComm> ProcessGroupNCCL::\n"
        "getIntraNodeComm() {\n  return nullptr;\n}\n",
        t,
        flags=re.S,
    )
    t = t.replace(
        "this->setUsePgForSymmMemRendezvous(options_->use_pg_for_symm_mem_rendezvous);",
        "(void)options_;",
    )
    PNCCL.write_text(t)


def patch_stream():
    t = STREAM.read_text()
    t = t.replace(
        '#include <torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory-inl.cuh>\n',
        "",
    )
    STREAM.write_text(t)


def patch_init():
    t = INIT.read_text()
    for inc in (
        "#include <torch/csrc/distributed/c10d/symm_mem/intra_node_comm.hpp>\n",
        "#include <torch/csrc/distributed/c10d/symm_mem/nccl_devcomm_manager.hpp>\n",
        "#include <torch/csrc/distributed/c10d/symm_mem/DMAConnectivity.hpp>\n",
        "#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>\n",
        "#include <torch/csrc/distributed/c10d/symm_mem/nvshmem_extension.hpp>\n",
    ):
        t = t.replace(inc, "")
    # drop the symmetric_memory python bindings block
    t = re.sub(
        r"\nPYBIND11_MODULE\(TORCH_EXTENSION_NAME, m\) \{\n.*?symmetric_memory.*?\n\}\n",
        "",
        t,
        flags=re.S,
    )
    INIT.write_text(t)


must(patch_bzl, "build_variables.bzl")
must(patch_pncl, "ProcessGroupNCCL.cpp")
must(patch_stream, "StreamBlock.cu")
must(patch_init, "init.cpp")

if errors:
    print("\n".join(errors))
    sys.exit(1)
print("symm_mem excluded: bzl + ProcessGroupNCCL + StreamBlock + init patched")
