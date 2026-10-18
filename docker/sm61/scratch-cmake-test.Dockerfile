# scratch env: prepared torch build environment (no configure) for experiments
FROM local/vllm-source-cache:sm61-v028 AS src
FROM vllm/vllm-openai:v0.28.0-cu129

COPY --from=src /opt/src/pytorch /tmp/pytorch

RUN apt-get update && apt-get install -y --no-install-recommends build-essential git libnuma-dev ccache \
    cuda-cudart-dev-12-9 cuda-profiler-api-12-9 libnvidia-ml-dev \
    && rm -rf /var/lib/apt/lists/* \
 && python3 -m pip install --upgrade pip 'setuptools<81' wheel packaging ninja cmake setuptools-scm setuptools-rust jinja2 regex build protobuf

# CUDA symlink layer (as in the real build): pip nvidia packages -> /usr/local/cuda
RUN set -eu; \
    cuda_root=/usr/local/cuda; \
    cuda_include="$cuda_root/include"; \
    cuda_lib64="$cuda_root/lib64"; \
    cuda_targets_include="$cuda_root/targets/x86_64-linux/include"; \
    cuda_targets_lib="$cuda_root/targets/x86_64-linux/lib"; \
    packages_root=/usr/local/lib/python3.12/dist-packages/nvidia; \
    mkdir -p "$cuda_include" "$cuda_targets_include" "$cuda_targets_lib"; \
    for include_dir in "$packages_root"/*/include; do \
      [ -d "$include_dir" ] || continue; \
      for item in "$include_dir"/*; do \
        ln -sf "$item" "$cuda_include/$(basename "$item")"; \
        ln -sf "$item" "$cuda_targets_include/$(basename "$item")"; \
      done; \
    done; \
    for lib_dir in "$packages_root"/*/lib; do \
      [ -d "$lib_dir" ] || continue; \
      for item in "$lib_dir"/*; do \
        base=$(basename "$item"); \
        ln -sf "$item" "$cuda_lib64/$base"; \
        ln -sf "$item" "$cuda_targets_lib/$base"; \
        case "$base" in \
          *.so.*) \
            unversioned=${base%%.so.*}.so; \
            ln -sf "$base" "$cuda_lib64/$unversioned"; \
            ln -sf "$base" "$cuda_targets_lib/$unversioned"; \
            ;; \
        esac; \
      done; \
    done; \
    ln -sf "$cuda_lib64" "$cuda_root/lib"; \
    echo "CUDA symlink layer ready"

# fabric patch (same as the real build)
COPY docker/sm61/patch-torch-fabric.py /usr/local/bin/patch-torch-fabric.py
RUN python3 /usr/local/bin/patch-torch-fabric.py

# torch deps (requirements.txt)
RUN cd /tmp/pytorch && python3 -m pip install -r requirements.txt

# docker run commands:
#   configure: cd /tmp/pytorch && rm -rf third_party/nccl build && USE_CUDA=1 BUILD_TEST=0 python3 setup.py build_ext
#   (kill after build.ninja appears) then: ninja -C build <target>
