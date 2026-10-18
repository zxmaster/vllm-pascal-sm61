# MIGRATION — переезд и развёртывание с нуля (v0.28)

> Кому: агенту/человеку **без контекста**. Цель — поднять этот проект на
> другой машине (или после переустановки) **без пересборки** тяжёлых
> артефактов (сборка torch+vLLM = часы). Актуально на 2026-09-16.
> Связанные документы: `docs/UPSTREAM.md` (обновление vLLM и патчи),
> `docs/FIXES.md` (реестр проблем/фиксов), `README.md`.

## 1. Что это за проект и что лежит вне git

Репо — это **deployment wrapper** вокруг upstream vLLM (не форк): compose,
скрипты запуска, runtime-патчи, доки. Модели, docker-образы и артефакты
сборки — **вне git** и переносятся отдельно.

| Артефакт | Путь | Размер | Критичность |
|---|---|---|---|
| Компилированный torch (sm_61) | `artifacts/torch/site-packages/current/` (symlink → `extract-20260903T012735Z`) | ~850 МБ | 🔴 **высшая** (часы сборки) — либо скачать готовый wheel: `scripts/fetch-torch-artifact.sh` |
| Docker-образ v0.29 (основной) | `local/vllm-openai:sm61-v029` | ~46 ГБ распак. | 🔴 **высшая** |
| Docker-образ v0.28 (резерв/откат) | `local/vllm-openai:sm61-v028` | ~45 ГБ распак. | 🟡 для отката |
| ~~v0.18~~ | — | — | ✅ списан 16.09 (образ утрачен) |
| Source-cache образы | `local/vllm-source-cache:sm61-v029` / `:sm61-v028` | ~41 ГБ | ⚪ только для пересборки |
| GGUF-модели | `volumes/models/` (Qwen3.5-4B-*: IQ4_XS/IQ4_NL/DFlash/MTP, MiniCPM5-2B-GGUF, Qwen3.8-27B-GGUF) | ~29 ГБ | 🔴 |
| HF-кэш | `volumes/hf-cache/` (HF-эталоны для отладки) | ~21 ГБ | 🟡 нужен для HF-сверок |
| HF-конфиги | `volumes/hf-configs/` (Qwen3.5-4B-text, MiniCPM5-2B) | ~10 МБ | 🔴 (оффлайн-режим) |
| Профили конфигурации | `.env.v028`, `.env.minicpm5`, `.env.iq4nl`, … | КБ | 🔴 |
| Логи/тесты | `logs/` (в git **не** трекается) | — | 🟡 для верификации |
| Git-кэши сборки | `docker/sm61/git-cache/` | — | ⚪ (нужны только для пересборки) |

`volumes/vllm-cache/` (кэш компиляции vLLM) пересоберётся сам — переносить не обязательно.

**Torch не обязательно переносить руками:** на машине без артефакта достаточно
`./scripts/fetch-torch-artifact.sh` — он скачивает prebuilt sm_61 wheel из GitHub
Releases (`zxmaster/vllm-pascal-sm61`), сверяет SHA-256 и раскладывает в
`artifacts/torch/site-packages/current/`. Дальше `build-sm61-image.sh` сам видит дерево и
ставит `SM61_USE_TORCH_ARTIFACT=1`. Источник, состав и проверки — `docs/RELEASE-TORCH.md`.

**Профили без секретов:** в репо `API_KEY=EMPTY` (плейсхолдер). Реальные секреты в репо не хранятся.

## 2. Перенос на новую машину

```sh
# 1) Каталоги (models/hf-cache — большие, rsync с докачкой)
rsync -a --info=progress2 artifacts/torch volumes/ <user>@<newhost>:~/Projects/vllm-pascal/
rsync -a --info=progress2 logs/tests/ <user>@<newhost>:~/Projects/vllm-pascal/logs/tests/

# 2) Docker-образы (save сохраняет общие слои один раз; ~16-20 ГБ сжато)
docker save local/vllm-openai:sm61 local/vllm-openai:sm61-v028 \
  local/vllm-source-cache:sm61-v028 | gzip -1 > /tmp/vllm-pascal-images.tar.gz
rsync -a /tmp/vllm-pascal-images.tar.gz <user>@<newhost>:/tmp/
# на новой машине:
gunzip -c /tmp/vllm-pascal-images.tar.gz | docker load

# 3) Репо
git clone https://github.com/zxmaster/vllm-pascal.git && cd vllm-pascal
```

## 3. Развёртывание на новой машине (чеклист)

**Требования:** Linux x86_64; NVIDIA GPU **sm_61** (GTX 10xx; образ собран
только под compute 6.1); драйвер NVIDIA; Docker + runtime `nvidia`
(хостовый default `youki` **не подходит**); внешняя docker-сеть
`intranet_net` (Traefik; для локального использования достаточно
`docker network create intranet_net`); ~150 ГБ диска; sudo.

1. Клонировать репо, восстановить артефакты (§2). Быстрый путь без переноса артефактов
   с другой машины: `./scripts/prepare-host.sh` (проверки GPU/docker/диска, каталоги,
   сеть, `.env.v031`) → `./scripts/fetch-torch-artifact.sh` (prebuilt torch) →
   `./scripts/prepare-model.sh` (GGUF + токенизатор + config). Оба prepare-скрипта
   идемпотентны, у каждого есть `--check`.
2. Проверить образы: `docker images | grep sm61-v028`.
3. Поднять рабочий стек (теперь — v029):
   ```sh
   docker compose -f compose.sm61-v029.yaml up -d
   ```
   (env-файл по умолчанию `.env.v029`; контейнер `vllm-openai-v029`, IP `172.18.100.8`; бэкенд lazy — грузится по первому запросу ~190 с.)
   Резерв/откат — v028: `docker compose -f compose.sm61-v028.yaml up -d` (перед этим остановить v029).
4. Разбудить и проверить:
   ```sh
   docker exec vllm-openai-v029 python3 -c "
   import json,urllib.request,time; t0=time.time()
   r=json.load(urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8000/v1/models',
     headers={'Authorization':'Bearer EMPTY'}),timeout=300))
   print('ready in',round(time.time()-t0),'s:',[m['id'] for m in r['data']])"
   ```
5. Верифицировать по логам и батарее (§4).
6. Опционально — второй стек MiniCPM5 (§5), LibreChat/OpenCode (`scripts/opencode-local-vllm.sh`).

## 4. Верификация после переезда

**Ожидаемые маркеры в стартовом логе v029** (`docker logs vllm-openai-v029`):
- `[Pascal] runtime patches applied: ['sdpa', 'gguf_adapter', 'gguf_quant', 'tq_backend', 'tq_prefill_sdpa', 'tq_continuation_groups', 'penalties_bincount', 'llama_probe', 'qwen_embeds', 'layer_probe']`
- `GPU KV cache size: 45,251 tokens` (при MAX_MODEL_LEN=32768, util 0.85)
- `Application startup complete.`

**Быстрая батарея (12 промптов, ~1 мин):**
```sh
docker cp scripts/v028/battery_full.py vllm-openai-v029:/tmp/
docker exec vllm-openai-v029 python3 /tmp/battery_full.py
# Ожидаемо: Paris / 5 / hello / Blue / 1,2,3,4,5 / Berlin / Mercury… / 56 / Chat / 2 — все finish='stop'
```
Полная валидация: `sh scripts/validate-v028.sh` (мониторит температуры; BASE_URL зашит под IP этого стенда).

## 5. Второй стек: MiniCPM5-2B (TurboQuant, 81920 контекст)

```sh
docker compose -f compose.sm61-v028.yaml stop          # GPU одна — освободить
docker compose -f compose.minicpm5.yaml up -d --force-recreate
# маркеры: GPU KV cache size: 114,528 tokens, Maximum concurrency for 81920: 1.40x
# тесты: scripts/minicpm5/{battery,tools,longctx}.py (docker cp + docker exec; longctx N chars, 240k≈62k tok)
# вернуть основной:
docker compose -f compose.minicpm5.yaml stop && docker compose -f compose.sm61-v028.yaml up -d
```

## 6. Что НЕ делать

- **НЕ пересобирать образы «на всякий случай»**. Пересборка = часы. Если всё же нужна: `SM61_USE_TORCH_ARTIFACT=1` (в `.env.v028` уже) + артефакт на месте → torch пропускается.
- **НЕ `git clean`** (артефакты и данные вне git).
- **НЕ запускать два inference-контейнера одновременно** (одна GPU 4 ГБ; peer-stop).
- **НЕ `docker compose restart`** после правок `.env`/патчей — только `up -d --force-recreate` (в already-созданном контейнере env и патчи не обновляются).
- **НЕ редактировать патчи «на живой системе»** — источник истины `scripts/pascal_runtime_patch.py` (+ force-recreate).
- Прод-дефолт `PASCAL_GDN_DEBUG=0` (диагностика печатает до 32 строк на токен).

## 7. Known-good ориентиры (2026-09-16, после синка v0.29)

- Commit: ветка `sync/vllm-v0.29.0` (в master после merge); предыдущая точка — `c56eb0e804`.
- Образы: **v029 `9f3b36f577dd` — основной**; v028 `10dc42ce164f` — резерв/откат.
  ⚠️ **v0.18 списан 2026-09-16** (решение владельца): образ утрачен при сборке v029
  (тег `local/vllm-openai:sm61` перезаписан), контейнер удалён; пересборка не планируется.
  Для отката — только v028.
- vLLM `0.29.1.dev0` (база v0.29.0), torch `v2.13.0`, gguf-plugin `0.0.5`, CUDA в образе — под sm_61.
- v029: пул **45 251** ток. (32k лимит, util 0.85); длинный prefill **29 781 ток. — OK (630 с)**; батарея 12/12; tool calls OK.
- v028 (резерв): пул 43 690; длинный 29 781 — OK (734 с); батарея 12/12.
- MiniCPM5: пул **114 528** @ лимит **81 920**; длинный тест 61 789 ток. — OK (1190 с).

## 8. Если что-то не так

| Симптом | Куда смотреть |
|---|---|
| Контейнер Up, но запросы висят | бэкенд lazy: пнуть `GET /v1/models`; `docker logs -f` |
| HTTP 500 на длинном префилле | traceback в логах → `docs/FIXES.md` (dtype / OOM-буферы) |
| `anchor count=N` при старте | патч не нашёл якорь: `docs/UPSTREAM.md` §3 |
| `EngineDeadError` | движок умер, контейнер завершён: `up -d --force-recreate` |
| GPU занята «кем-то» | второй сервер из соседних стеков (LibreChat/bonsai); освободить |
