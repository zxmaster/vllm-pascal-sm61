# UPSTREAM — обновление vLLM и наш слой патчей

> Цель: безопасно обновлять pinned upstream (vLLM / torch / transformers /
> gguf-plugin) и точно знать, что и где мы патчим. Актуально на 2026-09-16 (v0.28).
> Связано: `docs/git-workflow.md` (процесс веток/пушей), `docs/FIXES.md`
> (почему каждый патч существует), `docs/MIGRATION.md` (развёртывание).

## 1. Модель работы

Репо — **обёртка**, upstream не вендорится. Upstream-исходники приходят из
**source-cache образа** (зафиксированные refs) и патчатся на этапе сборки и
рантайма:

- Pinned refs (`.env.v028`): `SM61_TORCH_REF=v2.13.0`, `SM61_VLLM_REF=v0.28.0`,
  `SM61_GGUF_PLUGIN_VERSION=0.0.5`.
- Dockerfile-ы: `docker/sm61/Dockerfile.sources-v028` (source cache) и
  `docker/sm61/Dockerfile.v028` (сборка образа).
- Сборка: `scripts/build-sm61-source-cache.sh` → `scripts/build-sm61-image.sh`
  (или `make`), **torch не пересобирается** при `SM61_USE_TORCH_ARTIFACT=1`
  (артефакт `artifacts/torch/site-packages/current/`).
- Образы: основной `local/vllm-openai:sm61-v029`; резерв/откат `local/vllm-openai:sm61-v028` (v0.18 списан 16.09).

## 2. Слои нашего кода в собранном образе

### 2a. Build-time патчи (`docker/sm61/`)
| Файл | Что делает |
|---|---|
| `patch-torch-fabric.py` | правки собираемого torch (fabric/… под sm_61) |
| `patch-torch-symm-mem.py` | правки symmetric memory в torch |
| `patch-transformers-qwen35.py` | правки transformers под qwen3.5 |
| `patch-vllm-sm61-v028.py` | **v0.28: минимальный** vLLM-патч (FA disable; GGUF-фиксы живут в плагине) |
| `patch-vllm-sm61.py` | ⚠️ исторический (v0.18-эра, ~3155 строк); для v0.28 **не используется** |

### 2b. Плагины/зависимости
- `vllm-gguf-plugin` версии `0.0.5` (внешний пакет; его внутренний API —
  тоже часть «upstream-поверхности»!). Основные GGUF-фиксы v0.28 — здесь.
- `external/turboquant` (git submodule, GPL-3.0) — только эра v0.18 (кастомный TQ-hook); в public-репозиторий НЕ входит. В v0.28
  TurboQuant **нативный** (`--kv-cache-dtype turboquant_k3v4_nc`); submodule
  монтируется, но в v0.28-рантайме не активен.

### 2c. Runtime-патчи (`scripts/pascal_runtime_patch.py`)
Применяются **на каждом старте контейнера** к site-packages внутри контейнера
(файл монтируется read-only: `./scripts:/workspace/scripts:ro`).

| # | Функция | Файл-цель в контейнере | Назначение |
|---|---|---|---|
| 1 | `patch_gdn_torch` | `vllm/model_executor/models/qwen3_next.py` (+ `scripts/pascal_gdn_torch.py`) | FLA Triton GDN-ядра → чистый torch (Pascal) |
| 2 | `patch_triton_attn` | `vllm/v1/attention/backends/triton_attn.py` | flash → SDPA fallback; blockwise + chunked 4D пути; dtype-дисциплина |
| 3 | `patch_gguf_adapter` | `vllm_gguf_plugin/weights_adapter/default.py` | qwen35 normalize, dt_bias, A_log=log(-s), RMSNorm−1 **только qwen3_5**, conv1d unsqueeze, multimodal fix, `[PASCAL-LLAMA-QK-PERM]` |
| 4 | `patch_gguf_quant` | quant/linear пути GGUF | Pascal dequant-фиксы (нет тензорных ядер) |
| 5 | `patch_tq_backend` | `vllm/v1/attention/backends/turboquant_attn.py` | `_HAS_FLASH_ATTN=False` ниже sm_80 |
| 6 | `patch_tq_prefill_sdpa` | `turboquant_attn.py` | 4D mem-efficient SDPA (prefill + continuation), dtype-sync, stride-0 expand |
| 7 | `patch_tq_continuation_groups` | `turboquant_attn.py` | per-KV-group attention для continuation-чанков |
| 8 | `patch_penalties_bincount` | `vllm/v1/worker/gpu/sample/penalties.py` | Triton `_bincount_kernel` (sm_70+ атомики) → torch-эквивалент |
| 9 | `patch_llama_layer_probe` | `vllm/model_executor/models/llama.py` | TEMP debug-проба (`PASCAL_GDN_DEBUG=1`) |
| 10 | `patch_qwen_embeds` | `vllm/model_executor/models/qwen3_5.py` / `qwen3_next.py` | фикс загрузки GGUF-эмбеддингов |
| 11 | `patch_qwen_layer_probe` | qwen-модели | debug-проба слоёв |

Идемпотентность: каждый патч проверяет свой маркер («already applied» → skip).
Список фактически применённых виден в логе: `[Pascal] runtime patches applied: [...]`.

## 3. Якоря и разбор поломок

- Патчи ищут **уникальный фрагмент кода** (anchor); если найдено 0 или >1 —
  намеренный жёсткий отказ: `SystemExit: pascal_runtime_patch: failed to locate …`
  / `anchor count=N`. Это фича: лучше упасть, чем пропатчить не то.
- При обновлении upstream: старт упал на anchor-ошибке → открыть файл-цель,
  найти соответствующий фрагмент в новой версии, обновить якорь в
  `scripts/pascal_runtime_patch.py` (diff: `git fetch upstream && git log -p <файл>`).
- **`--force-recreate` обязателен**: патчи применяются при СТАРТЕ контейнера;
  в старом контейнере файлы уже пропатчены и маркеры стоят.
- Маркеры для быстрого поиска: `Pascal SDPA fallback`, `PASCAL-LLAMA-QK-PERM`,
  `4D mem-efficient prefill SDPA`, `(continuation, grouped)`, `_pascal_bincount_torch`,
  `llama probe`, `qwen_embeds`.

## 4. Процедура апгрейда (sync branch)

Подробный git-процесс — `docs/git-workflow.md`; краткая версия:

1. Тег known-good: `git tag -a local/current-pascal-working-$(date +%F)` (+ push).
2. Ветка `sync/vllm-<version>`.
3. `git fetch upstream --tags --prune`; просмотреть диффы по **hot spots** (§5).
4. Обновить pinned refs: `SM61_VLLM_REF` в `.env.v028` (+ Dockerfile-ы при необходимости);
   проверить свежесть `vllm-gguf-plugin`.
5. Пересборка: `make refresh-source-cache && make source-cache && make build`
   (часы; torch пропускается с артефактом).
6. Старт и разбор: смотреть `runtime patches applied: [...]` и anchor-ошибки;
   чинить якоря, пока старт не станет чистым.
7. Валидация: батарея 12/12 (`scripts/v028/battery_full.py`), длинный префилл
   (напр. `scripts/minicpm5/longctx.py` 240k chars), контроль KV-пула в логе.
8. Обновить `README.md`/`docs/*` при изменениях; merge в master только после зелёного.

### Приземление v0.31.0 (17–18.10.2026)

- Цель: **v0.31.0** (последний стабильный; v0.32 — только rc).
- Контур: `.env.v031`, `compose.sm61-v031.yaml`, `Dockerfile(.sources).v031`,
  `patch-vllm-sm61-v031.py`, `start-vllm-v031.sh`; контейнер `vllm-openai-v031`
  (172.18.100.9, трафик vllm31.test). Рабочий v029 не мутируется.
- Source-cache: torch-слой отделён от vllm-слоя (при смене только `VLLM_REF`
  torch-слой берётся из кэша); база `vllm/vllm-openai:v0.29.0-cu129`, torch-артефакт
  2.13.0 переиспользуется (torch-пин upstream не менялся).
- Build-патч: единственное изменение якоря — DCP-список в CMakeLists.txt теперь
  начинается с `csrc/libtorch_stable/hisparse_kernels.cu`; его оставляем, три
  `dcp_direct_*` исключаем.
- Runtime-патчи: все 11 применились без правок (TQ prefill/continuation — старые
  якоря сохранены: `new2` из `patch_tq_prefill_sdpa` дословно равен `old` из
  `patch_tq_continuation_groups`).
- Сборка: образ `local/vllm-openai:sm61-v031` (45.9 ГБ), vLLM **0.31.1.dev0+gdb9527a46**.
- Валидация: KV-пул **45 251** ток.; батарея **12/12**; long-prefill **28 453** ток.
  за **334 с** (v029: 29 781 за 630 с), ответ `BLUE FALCON 7734` найден.
- Статус: **основной стек с 18.10.2026 — v031** (`vllm.test`); v029 — резерв/откат
  (`vllm29.test`), поднимается одной командой при необходимости.

## 5. Hot spots (что ревьюить в upstream при каждом апгрейде)

- `vllm/v1/attention/backends/triton_attn.py` — наш SDPA-fallback (меняется часто!)
- `vllm/v1/attention/backends/turboquant_attn.py` — TQ-путь (`_HAS_FLASH_ATTN`, prefill/continuation)
- `vllm/v1/worker/gpu/sample/penalties.py` — `_bincount_kernel`
- `vllm/model_executor/models/llama.py` — probe + структура слоёв
- `vllm/model_executor/models/qwen3_5.py`, `qwen3_next.py` — эмбеддинги, GDN
- `vllm/model_executor/layers/quantization/gguf*` — dequant/quant пути
- `vllm_gguf_plugin/weights_adapter/default.py` — конверсия GGUF (пакет 0.0.5)

## 6. Кандидаты на отдачу в upstream

- **TQ prefill/continuation фиксы** (mem-efficiency, dtype-sync, expand-вьюхи) —
  это общие баги TQ-пути, не sm_61-специфичные; можно оформить PR.
- **Q/K interleaved + V-head tiled для llama/qwen GGUF** — репортить в
  vllm-gguf-plugin (внешний пакет).
- `penalties` torch-fallback — только для < sm_70; как compatibility-фикс
  может быть приемлем.
- Полные sm_61-обходы (GDN torch, dequant fallback, FA disable) — **не для
  upstream**: неподдерживаемое железо, остаются в нашем слое.

## 7. Откат и эталоны

- v0.18 — **списан 2026-09-16** (образ утрачен, контейнер удалён; `compose.sm61.yaml` оставлен как история). Откат: v028.
- Отключить TQ: закомментировать `KV_CACHE_DTYPE` в профиле → `--force-recreate`.
- Откатить стек целиком: см. `docs/MIGRATION.md` §5/§7.

## 8. Грабли воркфлоу (коротко)

- `restart` ≠ `--force-recreate` (env + патчи не подхватываются).
- Бэкенд lazy: `GET /v1/models` будит (~190 с); `GET /healthz` — не будит.
- `HF_HUB_OFFLINE=1` — при старте без сети всё должно быть в `volumes/hf-cache`/`hf-configs`.
- Сборка: долгая; `MAX_JOBS=4`; артефакт torch — беречь.
- **ВСЕГДА задавать `IMAGE=` при сборке** (`IMAGE=local/vllm-openai:sm61-v029 ./scripts/build-sm61-image.sh`, уже прописан в `.env.v029`): дефолт скрипта — `local/vllm-openai:sm61` (тег v0.18-эталона). Сборка без IMAGE= перезаписывает ЧУЖОЙ тег (инцидент 16.09); после перезаписи проверить `docker images` и перетегировать корректно.
- **Sablier-группы**: `dynamic`-стратегия поднимает **все** контейнеры группы при любом запросе. Держать действующий сервер в отдельной группе (v029: `sablier.group=vllm29`, priority 310; v028: `vllm28`, priority 100) — иначе запрос к vllm.test будит и резервные контейнеры (инцидент 16.09: поднялись v028 и v0.18).
- Все «почему так» — в `docs/FIXES.md`.

## 9. Журнал синков

- **2026-09-16: v0.28.0 → v0.29.0.** Итог: всё зелёное (батарея 12/12, длинный
  29 781 ток. OK/630 с, tools OK, KV-пул 45 251). Якоря патчей НЕ менялись —
  все 11 применились как есть (проверялись заранее против исходников v0.29).
  Ход: fetch test (31 с) → диффы hot-spots (triton: удалены kv_cache_shape-методы;
  TQ: +supported_kv_cache_layouts; penalties не менялся) → контур v029 (compose/env/
  Dockerfile/патч) → pull базы (сетевой обрыв, автодокачка) → source-cache 17 мин →
  image ~30 мин (torch-артефакт переиспользован, `vllm._C` → `_C_stable_libtorch`
  в v0.29) → прогон и валидация. Артефакты: ветка `sync/vllm-v0.29.0`,
  образ `local/vllm-openai:sm61-v029`.
