# Git workflow for the Pascal vLLM testbed

This repository is the private, customized Pascal deployment wrapper for vLLM. It should stay small and focused on Docker, scripts, patches, reports, and configuration profiles. Upstream vLLM source is treated as a pinned build input, not vendored source.

## Remotes

- `origin`: private project repository (`https://github.com/zxmaster/vllm-pascal.git`). Push project work here.
- `upstream`: official vLLM repository (`https://github.com/vllm-project/vllm.git`). Fetch and compare only; pushing should be disabled.

Recommended setup:

```sh
git remote add upstream https://github.com/vllm-project/vllm.git 2>/dev/null || true
git remote set-url --push upstream DISABLED
git fetch upstream --tags --prune
```

## Branches

- `master`: stable, tested private branch.
- `feature/<name>`: normal development work.
- `sync/vllm-<version-or-date>`: upstream vLLM ref updates and patch repairs.

Do not update upstream source refs directly on `master`. Create a sync branch, rebuild, validate, then merge.

## What belongs in this repo

Include:

- Docker and compose files.
- Runtime/build patch scripts under `docker/` and `scripts/`.
- Small request fixtures used for reproducible tests.
- Reports and handoff docs.
- Submodule metadata for `external/turboquant`.
- Small environment profiles with no secrets, when they are intended to be shared.

Exclude:

- Real `.env` files, passwords, tokens, and host-only private config.
- Runtime logs and benchmark output.
- Docker/build caches.
- Model weights and other large binary artifacts unless explicitly handled with Git LFS and licensing is clear.
- Generated local OpenCode output artifacts.

## First push / cleanup checklist

Before pushing a mixed working tree:

1. Run `git status --short --branch`.
2. Run `sh ./scripts/check-repo-hygiene.sh`.
3. Split work into logical commits instead of one catch-all commit.
4. Confirm `git diff --cached --stat` does not include model weights, logs, generated output, private env files, or secrets.
5. Confirm submodule state with `git submodule status --recursive`.
6. Push to `origin`, not `upstream`.

## Upstream vLLM sync procedure

1. Start from a tested `master`:

   ```sh
   git checkout master
   git pull --ff-only origin master
   git checkout -b sync/vllm-v0.19.0
   ```

2. Fetch upstream:

   ```sh
   git fetch upstream --tags --prune
   ```

3. Review upstream changes around the local patch surface:

   ```sh
   git log v0.28.0..upstream/main --oneline -- vllm/model_executor vllm/v1
   ```

4. Update only the pinned build refs on the sync branch (актуально для v0.28):

   - `SM61_VLLM_REF` / `SM61_TORCH_REF` в `.env.v028` (основной источник).
   - При необходимости — соответствующие Dockerfile-ы (`Dockerfile.v028`,
     `Dockerfile.sources-v028`).
   - Проверить `SM61_GGUF_PLUGIN_VERSION` (внешний плагин).

5. Refresh and rebuild:

   ```sh
   make refresh-source-cache
   make source-cache
   make build
   ```

6. Fix patch failures in the wrapper source of truth (полная карта патчей — `docs/UPSTREAM.md`):

   - `docker/sm61/patch-vllm-sm61-v028.py` (build-time, v0.28)
   - `scripts/pascal_runtime_patch.py` (runtime, 11 патчей)
   - `scripts/pascal_gdn_torch.py`
   - `scripts/start-vllm-v028.sh`

   Do not manually edit the fetched `/tmp/vllm` tree as the long-term fix.

7. Validate runtime:

   ```sh
   make up
   make status
   make smoke
   ```

8. Run one known-good short request and one long-context sanity request if feasible.

9. Update reports/docs if behavior, limits, or defaults changed.

10. Merge the sync branch only after build and runtime validation pass.

## Rollback

Before a major upstream sync, tag the last known-good commit after the working tree is clean:

```sh
git tag -a local/current-pascal-working-YYYY-MM-DD -m "Known-good Pascal vLLM state before upstream sync"
git push origin local/current-pascal-working-YYYY-MM-DD
```

If a sync fails badly, reset or branch from that tag.
