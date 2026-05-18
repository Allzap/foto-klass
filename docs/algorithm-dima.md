# Алгоритм Димы — production pipeline для 13M фото автозапчастей

Зафиксировано **2026-05-19** после калибровочного прогона на `articles-archives/6378.7z`
(484 фото, RTX 4090 Secure Cloud, all real-world data из Hetzner S3).

Это **единственный официальный алгоритм** для prod-прогона. Не отклоняться без согласования с vda.

---

## 1. Source-of-truth pipeline (mode=resize)

```
┌─────────────────────────────────────────────────────────────┐
│ FOR each archive in articles-archives/*.7z:                 │
│                                                              │
│   1. DOWNLOAD       boto3 → /tmp/N.7z                       │
│   2. UNPACK         7z x → /tmp/N/*.webp                    │
│   3. FOR each photo (batched 16):                           │
│      a. content_bbox     trim white margins (thr=230)       │
│      b. maybe_upscale    Nomos PLKSR x4 if long_side<1000   │
│                          otherwise SKIP (Lanczos handles)   │
│      c. frame_to_square  scale to 85% × 1000, pad white     │
│      d. encode           WebP q90, 1000×1000                │
│   4. UPLOAD         parallel ×20 → up_v2/articles-archives/N/│
│   5. CLEANUP        rm /tmp/N* и переход к следующему архиву│
└─────────────────────────────────────────────────────────────┘
```

## 2. Зафиксированные модели и параметры

| Параметр | Значение | Почему именно так |
|---|---|---|
| Upscaler | `Phips/4xNomosWebPhoto_RealPLKSR` | Качество на уровне iloveimg.com, скорость 100-300ms на 4090, CC-BY-4.0 |
| **Skip-rule SR** | long_side ≥ **1000 px** | На 2500-pixel схеме Nomos дал тот же результат за +46 сек впустую |
| **batch_size** | **16** на CUDA, **1** на Mac MPS | MPS не оптимизирован для batched conv |
| **white_threshold** | **230** (было 245) | Захватывает чуть серые фоны поставщиков |
| **product_fill_ratio** | **0.85** (85%) | Спека Google Merchant: 75-90% |
| **output_size** | **1000×1000** | Target Google Merchant с запасом |
| **WebP quality** | **90** | Качество vs размер баланс |
| **Padding color** | **RGB(255, 255, 255)** | Чисто белый, Google требование |
| **Upload threads** | **20** parallel | ×40 ускорение vs последовательный (3 сек vs 119 сек на 484 файла) |

## 3. Полный CLI вызов (production)

```bash
python prototype.py \
    --input  /tmp/unpacked_archive/ \
    --output /tmp/processed/ \
    --mode resize \
    --device cuda \
    --batch-size 16
```

## 4. Замерные данные (RTX 4090, archive 6378.7z, 484 фото)

| Стадия | Время |
|---|---|
| Download 14.3 MB архив | 0.6 с |
| Unpack 7z (484 фото) | 0.8 с |
| Process (Nomos batch=16) | 151 с (= 312 ms/фото) |
| Upload parallel ×20 | 3.0 с |
| **TOTAL** | **155 с** на 484 фото |
| **Throughput** | **3.12 фото/сек на 1 GPU** |

## 5. Прогноз на полный 13M

**Допущение:** 1229 архивов в `articles-archives/`, avg 392 MB, avg ~13K фото на архив.

```
Per archive (avg):
  Download:  20 сек
  Unpack:    10 сек
  Process:   66 мин   ← 95% времени тут (GPU)
  Upload:    15 сек   (parallel ×20)
  ──────────────────
  Total:     ~67 мин = 1.12 GPU-час

Whole job:
  1229 archives × 1.12 = 1373 GPU-часов
  Community RTX 4090 ($0.34/ч): $467
  Secure RTX 4090 ($0.69/ч):    $948
```

**Wall time** зависит от параллельных pod'ов (стоимость одинаковая):

| Pods | Wall time |
|---|---|
| 5 | ~11.5 дней |
| 10 | ~5.7 дней |
| **20** ✅ | **~2.9 дня** |

## 6. Где НЕ обращаться через этот алгоритм (исключения)

Этот алгоритм — **mode=resize**, который НЕ:
- ❌ Удаляет watermark поставщиков (они остаются на фото)
- ❌ Заменяет тёмный фон поставщика на белый
- ❌ Удаляет тени

Если фото уходит в Google Merchant и есть жалоба на watermark/фон — гонять его отдельно через `--mode full`:
- В 10× медленнее (~3 сек/фото на 4090)
- Дороже: $3000-5000 на 13M
- Полная цепочка: Florence-2 → LaMa → Nomos → BiRefNet → composite

Для prod-прогона **все 13M обрабатываем mode=resize**. Полировка через full mode — только отобранный subset когда (и если) понадобится.

## 7. Что НЕ менять без согласования с vda

1. Модель апскейлера (`Phips/4xNomosWebPhoto_RealPLKSR`) — выбрана после сравнения с DAT2, Real-ESRGAN, HAT-L и iloveimg.com
2. `white_threshold=230` — порог трима по фону
3. `product_fill_ratio=0.85` — заполнение продуктом
4. `output_size=1000` и `webp_quality=90` — формат вывода
5. Pipeline `--mode resize` для всех 13M

Менять можно (после теста):
- `batch_size` 16 → 32 если хватает VRAM (4090 имеет 24GB, потенциально влезает)
- Upload thread count 20 → 40 если Hetzner S3 не ограничивает
- Pod cloud type (Community vs Secure) — Community в 2× дешевле, но реже доступен

## 8. Структура выходов в S3

```
allzap-tecdoc-photos/
├── articles-archives/        ← вход (1229 архивов 7z)
│   ├── 1.7z
│   ├── 2.7z
│   └── ...
└── up_v2/                    ← выход алгоритма Димы
    └── articles-archives/
        ├── 1/                ← один pod обработал один архив
        │   ├── 002ab468....webp   (1000×1000 WebP q90)
        │   ├── 00f7094f....webp
        │   └── ...
        ├── 2/
        └── ...
```

Имя файла = исходный SHA1 хэш (из `articles-archives/N.7z`). Это сохраняет ссылочную целостность для downstream consumer (Google Merchant feed).

## 9. История изменений алгоритма

| Дата | Что изменилось | Кем |
|---|---|---|
| 2026-05-19 | Зафиксирован алгоритм | vda + Claude |
