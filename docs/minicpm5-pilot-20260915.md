# MiniCPM5-2B pilot (IQ4_XS GGUF) — 2026-09-15

Первый прогон **не-Qwen** архитектуры (llama-класс) в нашем sm_61/4GB стеке v0.28.
Задача: поднять `MiniCPM5-2B` (OpenBMB, 2.52B, BFCL-v4 66.6, Apache 2.0) как
агентную модель с tool-calling на IQ4_XS.

## Итог

- **Работает**: батарея, tool calls (get_weather / search_flights — точный JSON
  и корректный multi-turn), численная сверка с HF по всем слоям.
- **TurboQuant проверен и включён** для llama-арки: контекст **81 920**,
  KV-пул **114 528** токенов; длинный тест **61 789 токенов — OK** (см. Фазу 2).
- Конфиг: `compose.minicpm5.yaml` + `.env.minicpm5` — отдельный стек
  (`vllm-minicpm5`), IQ4_XS, `MAX_MODEL_LEN=81920`, `GPU_MEMORY_UTILIZATION=0.87`,
  `KV_CACHE_DTYPE=turboquant_k3v4_nc`, `--tool-call-parser minicpm5`.
- Тот же образ `local/vllm-openai:sm61-v028` и тот же `start-vllm-v028.sh`
  (полностью параметризован).

## Найденные и исправленные баги (все — в `scripts/pascal_runtime_patch.py`)

1. **Triton `_bincount_kernel` (sampler penalties) не компилируется на sm_61**
   (`.acq_rel`/`.relaxed` атомики требуют sm_70+; `sem=` не помогает).
   → `patch_penalties_bincount`: torch-эквивалент (`index_add_` +
   `torch.unique` для бит-маски). Численно сверен с эталоном, включая
   граничный бит-31. Вызывается только при penalties у запроса, но падал
   на warmup всегда.

2. **RMSNorm shift был безусловным.** Наш фикс для qwen3_5 («GGUF хранит
   нормы как 1+w → вычитать 1») применялся ко всем архитектурам.
   У llama-family нормы хранятся **как есть** (bit-exact проверено против HF:
   0.0703125 == 0.0703125). Вычитание 1 превращало выход слоя в шум
   («toget toget ...»).
   → вычитать 1 только для `qwen3_5*`/`qwen35*`.

3. **Q/K проекции в llama-GGUF хранятся в interleaved (NEOX) layout**
   — внутри каждой 128-канальной головы строки идут `[0,64,1,65,2,66,...]`.
   vLLM применяет сверху HF-style RoPE (rotate_half по полу-парам), поэтому
   каждый q/k поворачивался неправильно. Доказательство: при тех же входах
   `V` совпадает с HF (rel 0.4%), а `q/k` — нет (cos≈0). Симптомы: почти
   случайные/«плоские» топы токенов, `</think>` вместо `<think>`, поломанные
   tool calls.
   → `[PASCAL-LLAMA-QK-PERM]` в `map_weights`: un-permute строк q/k
   (`qweight`, форма `[out, bytes]`) для `model_type == "llama"`:
   `sel = arange(out).reshape(-1,128)[:, [0::2]+[1::2]]`.
   После фикса: `q/k` совпадают с HF-RoPE (cos = 1.0000, rel ≈ 0.4%).

4. Мелочи: probe для llama (`[LAYER-L]`, `[FINAL-L]`, layer0 internals,
   q/k/v-сейвы под `PASCAL_GDN_DEBUG=1`).

## Фаза 2 — TurboQuant на llama-арке (вечер 15.09)

В рамках первого прогона TQ был выключен («llama-арка под TQ не проверена»).
Проверка показала: **турбо-бэкенд архитектурно-нейтрален** (`supports_kv_cache_dtype`
проверяет только `kv_cache_dtype.startswith("turboquant_")`; head_dim=128
поддержан наравне с 256). С `turboquant_k3v4_nc`:

| | fp16 KV | TurboQuant k3v4_nc |
|---|---|---|
| KV-пул (util 0.87) | 32 816 ток. | **114 528 ток. (×3.5)** |
| Лимит контекста | 32 768 | **81 920** (запас 1.40x) |

Найденные и исправленные баги (тот же `pascal_runtime_patch.py`):

5. **Dtype mismatch в чанкованном prefill-пути (triton_attn fallback):
   `_k_chunk` приводился к `.float()`, `_q_chunk` оставался fp16** →
   `RuntimeError: expected scalar type Half but found Float` на первом же
   длинном префилле. На коротких запросах путь не активировался.
   → `_q_chunk` тоже приводится к float (согласовано с fp32-аккумулятором
   `_acc`/`_l`, которые и так float32).

6. **OOM в continuation-пути (seq > ~50k):** K/V-буферы для grouped-attention
   материализовались по всей длине контекста (`[1, rep, seq, D]` ×2, ~104 МБ
   каждый при seq≈53k) плюс глобальная материализация `_k_chunk_full`
   в prefill-пути. Падение: нехватка 7 МБ при свободных 97 МБ.
   → K/V передаются в SDPA **expand-вьюхами (stride-0) напрямую**, без копий;
   per-block expand вместо full-seq. SDPA принимает stride-0 вьюхи
   (проверено), численно бит-в-бит идентично (`maxdiff=0.0`), пик памяти
   на тесте 32k: **43 МБ vs 178 МБ**.

Результат фазы 2: длинный контекст **61 789 токенов — OK** (1190 с,
«needle» BLUE FALCON 7734 найден). Тот же патч улучшает и v028 (общий код).

## Методика отладки (пригодится в будущем)

- HF-эталон на CPU (float32) → послойная сверка через сейвы в `/logs/*.pt`:
  сначала `_full` (hidden+residual) по слоям, затем внутренности слоя 0
  (norm-in, q/k/v после RoPE, attn-out).
- Сравнение `vllm q vs hf_rot(perm_b(q_pre))` дало cos=1.0 — это указало
  на комбинацию «перестановка в файле + HF-rope в vLLM».
- Проверка «не при чём»: три attention-бэкенда (SDPA/Triton FLEX) и
  fp16/fp32 дают одинаковую ошибку → дефект не в бэкенде и не в dtype.
  Замена θ-rope не меняла ошибку → дефект не в частотах, а в layout.
- Для TQ-путей: сначала короткий смоук, затем длинный прогон; смотреть
  `GPU KV cache size` в логе старта и хвост `docker logs` при 500-х
  (`EngineCore encountered an issue` → смотреть traceback выше).

## Числа

- Q/K после фикса vs HF: cos = 1.0000, rel 0.3–0.6 % (уровень кванта).
- Tool calls: `{"city": "Bangkok"}`, выбор из двух инструментов — корректный.
- TQ: пул 114 528 ток., лимит 81 920, длинный тест 61 789 ток. OK (1190 с).
- CPU-эталон для отладки: `openbmb/MiniCPM5-2B` (bf16→fp32) в
  `volumes/hf-cache/MiniCPM5-2B-hf`.
- Диагностические скрипты: `/tmp/cmp_qkv*.py`, `/tmp/find_perm2.py`,
  `/tmp/hf_*.py` (сохранены в истории сессии); тесты —
  `scripts/minicpm5/{battery,tools,longctx}.py`.
