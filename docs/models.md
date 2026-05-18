# Какие модели использовать (зафиксировано)

**Все четыре модели уже зафиксированы в `prototype.py` и `config.yaml.example`.
Не меняй их без согласования с vda — выбор сделан после тестирования.**

## Главный апскейлер — **Nomos PLKSR x4** (по умолчанию используется)

- **HF:** `Phips/4xNomosWebPhoto_RealPLKSR`
- **Файл:** `weights/4xNomosWebPhoto_RealPLKSR.safetensors` (~190 MB)
- **Лицензия:** CC-BY-4.0 (коммерческая работа разрешена)
- **Скачать:**
  ```bash
  mkdir -p weights
  huggingface-cli download Phips/4xNomosWebPhoto_RealPLKSR \
      4xNomosWebPhoto_RealPLKSR.safetensors --local-dir weights
  ```

**Скорость на разных машинах:**
- Mac MPS (M-серия): ~4 сек на фото 400×400 → 4×
- RTX 4090: ~100-150 ms на фото без батчинга, ~30-50 фото/сек с batch=8

**Почему именно она:** мы сравнивали на `sample/3_orig.webp` против:
- DAT2 (`Phips/4xRealWebPhoto_v4_dat2`) — тот же visual quality, но 15× медленнее на Mac, ~10× медленнее на 4090
- Real-ESRGAN x4 (`ai-forever/Real-ESRGAN`) — немного хуже на мелком тексте артикулов
- Nomos HAT-L (`Phips/4xNomos8kHAT-L_otf`) — чуть чище визуально, но 90× медленнее
- Коммерческий iloveimg.com — наш Nomos почти не отличается

**Skip-rule:** если у входного фото длинная сторона ≥ 1000px → апскейл
пропускается, делается только Lanczos downscale. Проверено — Nomos на больших
исходниках ничего не добавляет (см. `6_orig.webp`: 2500×812, Nomos дал тот же
результат что Lanczos за 46 секунд впустую).

## Что происходит с фото любого размера (resize mode)

Финал всегда **1000×1000 WebP q90** с белым паддингом, деталь ~85% по длинной
стороне. Различается только средняя ветка пайплайна:

| Длинная сторона исходника | Что делает пайплайн |
|---|---|
| **≥ 1000 px** (например 2500×812) | trim белых полей → **skip Nomos** → Lanczos downscale до 850px по длинной стороне → паддинг до 1000×1000 |
| **< 1000 px** (например 400×400) | trim белых полей → **Nomos x4** → Lanczos downscale до 850px по длинной стороне → паддинг до 1000×1000 |

**Конкретные примеры на наших 7 sample фото:**

| Файл | Исходник | Что произошло |
|---|---|---|
| `6_orig.webp` | 2500×812 | trim → skip Nomos → Lanczos в ~850×275 → центрировано в 1000×1000 |
| `3_orig_ilove.jpg` | 798×798 | 798 < 1000 → Nomos x4 → 3192×3192 → Lanczos в 850×850 → 1000×1000 |
| `3_orig.webp` | 399×399 | 399 < 1000 → Nomos x4 → 1596×1596 → Lanczos в 850×850 → 1000×1000 |
| `9_orig.webp` | 600×324 | Nomos x4 → 2400×1296 → Lanczos в 850×459 → 1000×1000 |
| `10_orig.webp` | 600×248 | Nomos x4 → 2400×992 → Lanczos в 850×351 → 1000×1000 |

Можно увидеть глазами в `output_resize/<имя>.webp` после прогона
`python prototype.py --input sample/ --output output_resize/ --mode resize`.

## Почему порог именно 1000

Цель — 1000×1000 финал. Поднимать порог выше 1000 не нужно: Nomos
добавляет детали только если итог будет крупнее исходника. На исходниках
≥1000 Lanczos справляется без потерь.

Если хочется страховаться (например — фото вдруг сильно сжатое JPEG
выглядит мутным даже при 1200×800), можно поднять порог:

```yaml
# В config.yaml
upscale_skip_long_side_px: 1200    # вместо 1000
```

И запустить с `--config config.yaml`.

## Полный список моделей (для режима `--mode full`)

| Этап | Модель | HF путь | Лицензия |
|---|---|---|---|
| Детектор водяных знаков + OCR | Florence-2-base | `microsoft/Florence-2-base` | MIT |
| Инпейнтер (стирание ВЗ) | Big-LaMa | через pip `simple-lama-inpainting` | Apache-2.0 |
| **Апскейл** | **Nomos PLKSR x4** | `Phips/4xNomosWebPhoto_RealPLKSR` | CC-BY-4.0 |
| Удаление фона | BiRefNet-matting | `ZhengPeng7/BiRefNet-matting` | MIT |

В режиме `--mode resize` (для массового прогона 13M) используется
**только апскейлер Nomos PLKSR** — остальные модели не загружаются,
видяные знаки остаются на фото.

## Конкретные строчки в коде, где модели зафиксированы

```python
# prototype.py:88-93
florence_model: str = "microsoft/Florence-2-base"
birefnet_model: str = "ZhengPeng7/BiRefNet-matting"
upscaler_weights_path: str = "weights/4xNomosWebPhoto_RealPLKSR.safetensors"
```

```yaml
# config.yaml.example
upscaler_weights_path: "weights/4xNomosWebPhoto_RealPLKSR.safetensors"
florence_model: "microsoft/Florence-2-base"
birefnet_model: "ZhengPeng7/BiRefNet-matting"
```

## Если хочется попробовать другую модель

**НЕ меняй дефолты в `prototype.py` или `config.yaml.example`** — это сломает
воспроизводимость. Вместо этого:

1. Скачай альтернативные веса в `weights/`
2. Скопируй `config.yaml.example` в `config.yaml` (он gitignored)
3. Меняй путь в локальном `config.yaml`
4. Запускай `python prototype.py --config config.yaml ...`

Список протестированных альтернатив есть в `scripts/sr_compare.py` —
там я скриптом гоняю одно и то же фото через DAT2, Real-ESRGAN, Nomos и HAT-L
для side-by-side сравнения.
