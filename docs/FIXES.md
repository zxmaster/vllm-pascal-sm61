# FIXES — реестр проблем и решений (все эры)

> Полный список того, с чем столкнулись при заведении vLLM на sm_61 / 4 ГБ.
> Формат: **симптом → корень → фикс → где**. Разделы: актуальное (v0.28+),
> историческое (v0.18, что остаётся в силе), «не делать», методика диагностики.
> Детали: `docs/minicpm5-pilot-20260915.md` (пилот llama), `MEMORIES.md` и
> `VLLM-PASCAL.md` (v0.18-история подробнее), механика патчей — `docs/UPSTREAM.md`.

## A. Эра v0.28 + llama + TurboQuant (2026-09) — актуальное

### A1. penalties: Triton `_bincount_kernel` не компилируется на sm_61
- **Симптом:** старт падает на warmup (PTXAS error; «No available kernel»).
- **Корень:** ядро использует атомики `.acq_rel/.relaxed` — требуют sm_70+; `sem=relaxed` не спасает.
- **Фикс:** `patch_penalties_bincount` — torch-эквивалент (`index_add_` + `torch.unique` для бит-маски), численно бит-в-бит (включая граничный бит-31).
- **Где:** `scripts/pascal_runtime_patch.py`.

### A2. Мусор на выходе llama-моделей (MiniCPM5) — составной баг
- **A2a. RMSNorm-сдвиг.** GGUF-плагин вычитал 1 из ВСЕХ норм (верно для qwen3_5). У llama-семейства нормы хранятся **как есть** (bit-exact с HF; 0.07 не 0.07-1). Симптом: «toget toget…», шумные логиты.
  → Вычитать 1 **только для `qwen3_5*`/`qwen35*`**.
- **A2b. Q/K interleaved (NEOX) в llama-GGUF.** Внутри каждой 128-канальной головы строки хранятся `[0,64,1,65,…]`; vLLM сверху применял HF-rope → каждый q/k повёрнут неверно. Доказательство: V совпадал с HF (cos 0.999), Q/K — нет (cos≈0).
  → `[PASCAL-LLAMA-QK-PERM]` в `map_weights`: un-permute строк q/k при `model_type == "llama"`. После — cos=1.0000, rel ≈0.4%.
- **Симптомы для опознания:** «заикание», `</think>` вместо `<think>`, ломаные tool calls, плоские топы.
- **Где:** `scripts/pascal_runtime_patch.py` (patch_gguf_adapter).

### A3. dtype Half/Float в чанкованном prefill (HTTP 500 на длинном префилле)
- **Симптом:** `RuntimeError: expected scalar type Half but found Float` в `triton_attn.py`.
- **Корень:** K-чанк кастовался в `.float()`, Q оставался fp16. Короткие запросы этот путь не активируют → баг «спит» до первого длинного префилла.
- **Фикс:** приводить Q к тому же типу (согласовано с fp32-аккумуляторами `_acc/_l`).
- **Правило:** если K/V кастуются во float (для точности) — кастовать и Q.

### A4. OOM в continuation-пути TQ (≥ ~50k, «не хватило 7 МБ»)
- **Симптом:** `CUDA out of memory: Tried to allocate 104 MiB. … 97 MiB free` в `_continuation_prefill`.
- **Корень:** буферы `[1, rep, seq, D]` ×2 материализовались на всю длину контекста (104 МБ @53k) + full-seq материализация `_k_chunk_full` в prefill-ветке.
- **Фикс:** **stride-0 expand-вьюхи прямо в SDPA** (SDPA на sm_61 их принимает), per-block expand внутри циклов. Численно бит-в-бит (`maxdiff 0.0`), пик **43 МБ vs 178 МБ** (@32k).
- **Правило:** никогда не материализовать K/V, растущие с seq_len.

### A5. TQ prefill OOM (миграция v0.28, первые длинные префиллы)
- **Симптом:** OOM в `turboquant_attn.py::_prefill_attention` (~8k).
- **Корень:** SDPA вызывался 3D + `enable_gqa` → fused-ядра недоступны (нужен 4D; GQA не поддержан сборкой) → math-ядро → интермедиаты O(seq²).
- **Фикс:** `patch_tq_prefill_sdpa` — 4D + `repeat_interleave` (порядок HF — НЕ enable_gqa!) + `sdpa_kernel(EFFICIENT_ATTENTION)` с `except RuntimeError`-фолбэком; позже continuation → per-KV-group (`patch_tq_continuation_groups`).

### A6. Аллокатор PyTorch — ничего не трогать
- `max_split_size_mb:*` → «swiss-cheese» reserved-не-unallocated (~260 МБ) → падения на мелких аллокациях.
- `expandable_segments:True` → ломает профайлинг KV в vLLM (0.43→0.23 GiB).
- **Итог:** дефолтная политика. (Проверено практикой трижды.)

### A7. Мелкие грабли v0.28
- `PASCAL_GDN_DEBUG` сравнивается c `"1"` явно (непустая строка `"0"` была truthy).
- 25k: после 4D-фикса понадобились chunk `MAX_NUM_BATCHED_TOKENS=1024` + grouped continuation; 32k: util 0.88→0.85 (баланс «пул ≥ лимита»).
- MiniCPM5: tool-парсер называется `minicpm5` (не `minicpm5xml` — это имя файла); спец-reasoning-парсера нет, но теги `<think>…</think>` MiniCPM5 матчит универсальный `deepseek_r1` (см. A8).
- После `--force-recreate` весь контейнерный `/tmp` пуст — тест-скрипты копировать заново (`docker cp`).
- Idle-прокси: первый запрос грузит модель ~190 с; `GET /healthz` не будит, `GET /v1/models` будит.

### A8. Файнтюн Fable5 «сходит с ума» по-русски: серверный дефолт temperature=1.0
- **Симптом:** в многошаговых сценариях (Hermes + 22 инструмента, стриминг) ответы превращаются в кашу с мешаниной языков («Температура APPROXИМАЛЬНОЛЯ — 29–30 °C», вплоть до CJK/арабских вставок); tool-call или пропускается, или уходит не туда («погода Punjab Pakistan»). Короткие прямые запросы при этом отвечают связно.
- **Корень:** MiniCPM5-2B-Fable5 — thinking-only файнтюн с `generation_config.json` `temperature=1.0, top_p=0.95`; vLLM применяет его как серверный дефолт, когда запрос не задаёт sampling. Hermes температуру **не шлёт** вовсе (для своих провайдеров опускает ключ; переопределения температуры в `model_overrides` нет) → модель всегда сэмплилась на 1.0. Рекомендация README карточки (temp=1.0) для этого файнтюна вредная: на 1.0 он плывёт, на 0.3–0.4 стабилен.
- **Фикс:** серверный дефолт `--override-generation-config {"temperature":0.4,"top_p":0.95}` (применяется ко всем клиентам, когда они не задают параметры). A/B-замер (`scripts/minicpm5/fable5-ab.py`): temp 0.3 — чистый tool-call и чистый русский; temp 1.0 — «Punjab/Pakistan» и мешанина.
- **Где:** `.env.minicpm5` → `VLLM_EXTRA_ARGS`. Класс проблемы: «клиент шлёт не все параметры — дефолты решает сервер».

### A9. Пол контекста Hermes (≥64k) диктует выбор кванта: Q8_0 не влезает
- **Симптом:** `Model MiniCPM5-2B has a context window of 32,768 tokens, which is below the minimum 64,000 required by Hermes Agent`.
- **Корень:** у Hermes жёсткий пол `MINIMUM_CONTEXT_LENGTH = 64_000` (tool-calling ниже нестабилен). В 4 ГБ: Q8_0 (2.56 GiB весов) даёт максимум ~50k токенов KV даже при `GPU_MEMORY_UTILIZATION=0.92`; Q4_K_M (1.49 GiB) — 81 920 при 0.90.
- **Фикс:** Q4_K_M + `MAX_MODEL_LEN=81920` + `GPU_MEMORY_UTILIZATION=0.90` (KV-пул 128 128 ток., 1.56×). Рекомендованный автором Q8_0 остаётся в `volumes/models/MiniCPM5-2B-Fable5-GGUF/` — годится для llama.cpp/других машин, но не для этой.
- **Где:** `.env.minicpm5`; причина выбора зафиксирована в шапке файла.

### A10. Мышление в ответе: `<think>`-блоки уходят в `reasoning_content`
- **Симптом:** в выводе клиента (Hermes CLI) перед ответом видна цепочка мыслей.
- **Корень:** без парсера vLLM отдаёт всё в `content`; модель thinking-only, CoT активен всегда.
- **Фикс:** `--reasoning-parser deepseek_r1` (матчит `<think>`/`</think>`; спец-парсера MiniCPM5 нет) → мысли в `reasoning_content`, `content` чистый; в `usage` виден `reasoning_tokens`.
- **Где:** `.env.minicpm5`.

## B. Эра v0.18 (2026 весна) — историческое, но пункты в силе

### B1. V-head tiled vs grouped (ГЛАВНЫЙ баг «token salad»)
- **Симптом:** мусор (`!!!!`), Pearson r 0.78.
- **Корень:** GGUF-конвертер перекладывает V-головы в **tiled**-layout (для `ggml_repeat`), vLLM FLA использовал grouped: `head_map = arange(H) // heads_per_group` — неверно.
- **Фикс:** `head_map = arange(H) % num_kv_heads` (tiled); в decode-ядре `i_h = i_hv % H`.
- **Результат:** Paris-логит 12.24 → 18.06 (ref 18.05), Pearson 0.9995.
- **Не откатывать** (попытка «un-reorder» даёт мусор — tiled сохранять как есть).

### B2. GGUF shard ordering
- Целочисленные stacked-параметры (`qkv`, `gate_up`) переупаковываются конвертером не в том порядке, что ждёт vLLM.
- **Фикс:** reorder в GGUF weight loader.

### B3. GemmaRMSNorm → RMSNorm (двойной +1)
- Конвертер gguf **пред-добавляет +1** к нормам; vLLM брал `GemmaRMSNorm` (тоже +1) → двойной сдвиг.
- **Фикс:** для GGUF-моделей использовать `RMSNorm`. `ssm_norm` — без +1 (проверено: добавление ухудшает).

### B4. Embed tokens GGUF
- `VocabParallelEmbedding` создавался без `quant_config` → Q6_K-эмбеддинг грузился случайным fp16 (~1.2 ГиБ), терялось ~700 МиБ, 0 токенов под KV.
- **Фикс:** прокинуть `quant_config`; доп. — chunked dequant (256 МиБ).

### B5. Pascal dequant fallback
- Нет тензорных ядер (sm_61) → GGUF MMVQ/MMQ небезопасны.
- **Фикс:** chunked dequant+matmul для cc < 7.0.

### B6. GDN / attention fallbacks
- FLA Triton-ядра GDN (gating, prefill) на Pascal дают неверные результаты / падают.
- **Фикс:** чистый torch (`scripts/pascal_gdn_torch.py`); flash → `scaled_dot_product_attention`.
- `VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE=0`.

### B7. conv1d-конвенция (найдено 14.09)
- State/вес-порядок: `acc = x_t*w[:,W-1] + (state*w[:,:W-1]).sum(1)`; прошлый вариант (w[:,0] для x_t) давал «повёрнутое» ядро — «почти работает, но врёт».
- **Валидация только юнит-тестом** против `F.conv1d(padding=W-1)`, а не прогонами.

### B8. A_log = log(-ssm_a)
- Негативная экспонента и log-трансформ — из llama.cpp; **не скипать** (реверс тестировали — хуже).

### B9. Хост-окружение
- `runtime: nvidia` обязателен (хостовый default `youki` ломает GPU-passthrough).
- `HF_HUB_OFFLINE=1` — DNS в контейнере ненадёжен; всё предзакачано.

### B10. Таймауты и idle (v0.18-эпоха)
- Клиентский `REQUEST_TIMEOUT_SECONDS` и прокси `IDLE_UNLOAD_REQUEST_TIMEOUT_SECONDS` — два разных реза; длинным запросам ставить 0.
- Idle-reaper не убивает запрос в полёте — режет только прокси-таймаут.

## C. «НИКОГДА НЕ ДЕЛАТЬ» (запреты, собранные из всех эр)

1. Не возвращать `enable_gqa` в наши SDPA-вызовы — на sm_61 тихо врёт (проверено численно: 1.5x расхождение).
2. Не материализовать K/V, растущие с длиной контекста (ни `repeat_interleave ×4`, ни буферы; только expand-вьюхи / per-block).
3. Не ставить `PYTORCH_CUDA_ALLOC_CONF` с `max_split_size_mb` / `expandable_segments` (см. A6).
4. Не вычитать 1 из норм llama-семейства (только qwen3_5*) — см. A2a.
5. Не «чинить» V-проекции MiniCPM5 (Q8_0, без перестановок; permuted только q/k).
6. Не править патчи в живом контейнере — источник истины `scripts/pascal_runtime_patch.py` + `--force-recreate`.
7. Не `docker compose restart` вместо `up -d --force-recreate` после правок env/патчей.
8. `patch_qwen_rope_probe` удалён (отключал RoPE) — никогда не возвращать.
9. Не `git clean` в репо; не пересобирать образы «на всякий» (артефакты дорогие).
10. В проде `PASCAL_GDN_DEBUG=0` (диагностика печатает десятки строк на токен).
11. Не запускать два inference-контейнера на одной GPU.

## D. Методика диагностики (как находили баги)

- **HF-CPU эталон** (`volumes/hf-cache/*`): послойные сейвы, сверка `sum_mean` после слоя k vs HF `hidden_states[k+1]` (норма ≤3% = квантование). Затем сузить до внутренностей слоя (norm-in, q/k/v после RoPE, attn-out).
- **GGUF-санити:** сначала прогнать через llama.cpp (внешний контейнер) — если GGUF даёт осмысленное, проблема в vLLM-пути.
- **Сравнения тензоров:** cos + rel по каналам, не только средние. Так найдены Q/K-perm (cos≈0 при «похожих» средних).
- **Probe-маркеры** (при `PASCAL_GDN_DEBUG=1`): `[LAYER-L]`, `[FINAL-L]`, `[PASCAL-SDPA]`, `[MAPW]`, `[PGDN*]`, `[EMBED]`; HF-сторона: `[HF-LAYER]`, `TOP10`.
- **Триаж HTTP 500:** `docker logs <c> 2>&1 | grep -iE 'error|traceback'`:
  - `expected scalar type` → dtype (A3);
  - `CUDA out of memory` → seq-зависимые копии (A4) или аллокатор (A6);
  - `EngineDeadError` → движок мёртв; контейнер завершён → `--force-recreate`.
- **Пробы без сервера** (GPU свободна): `docker run --rm --gpus all …` (напр. проверка SDPA expand: maxdiff/пик).
- Скрипты: `scripts/minicpm5/{battery,tools,longctx}.py`, `scripts/v028/battery_full.py`, `scripts/validate-v028.sh`, `logs/tests/*`; методички — `docs/minicpm5-pilot-20260915.md` §методика.

## E. Хронология вех (контекст)

- **2026-04:** обход Pascal (torch/vLLM под 6.1), найдена V-head tiled (07.04) — заработало 1:1 с llama.cpp.
- **2026-05:** TurboQuant эксперименты (hook, v0.18), длинные префиллы 10k, отчёты `TURBOQUANT-REPORT.md`.
- **2026-09-03:** v0.28: фикс утечки, чистая загрузка.
- **2026-09-12…14:** миграция v0.18 → **v0.28** (нативный TQ `k3v4_nc`), контекст 12k → 25k → **32k**; IQ4_XS (−240 МБ весов); батарея 12/12.
- **2026-09-15:** пилот **MiniCPM5-2B** (первый llama): три фикса (A1–A2); вечером **TQ-on-llama + два long-ctx фикса (A3–A4)** → контекст **81 920**, длинный тест **61 789 ток. OK**; v028 регресс 12/12.
- **2026-09-28:** MiniCPM5-2B → файнтюн **Claude-Fable5-1 Thinking/Agentic** (Q4_K_M; Q8_0 не проходит пол 64k в 4 ГБ), reasoning-парсер `deepseek_r1`, серверный дефолт temp 0.4 (A8–A10); сквозной прогон через Hermes-профиль `local` → ai-gateway → `vllm-minicpm5` (tool-calls + русский OK).
