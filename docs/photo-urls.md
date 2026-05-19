# Как формировать URL к фото — для программиста

Все обработанные фото лежат в Cloudflare R2 bucket `allzap-photos`, доступном
через CDN-домен **`photo.allzap.site`**.

## URL-схема

```
https://photo.allzap.site/<hash>/unpacked/<archive_id>/<filename>
                         └─┬─┘            └────┬─────┘  └───┬───┘
                          2 chars            номер архива   SHA1 имя из исходника
                          (filename[:2])
```

### Пример

| Что | Значение |
|---|---|
| Файл из архива | `articles-archives/6378.7z` → внутри `002ab468d8d5c29236f788294e278bd373d5a6ce.webp` |
| `hash` | `00` (первые 2 символа имени файла) |
| `archive_id` | `6378` |
| **URL** | `https://photo.allzap.site/00/unpacked/6378/002ab468d8d5c29236f788294e278bd373d5a6ce.webp` |

## Готовая функция на Python

```python
def photo_url(filename: str, archive_id: str | int) -> str:
    """
    Вернёт публичный CDN-URL для фото из распакованного архива.

    filename:   SHA1-имя файла как лежит в исходном архиве,
                например '002ab468d8d5c29236f788294e278bd373d5a6ce.webp'.
                Должно содержать минимум 2 символа.
    archive_id: номер архива из articles-archives/<N>.7z,
                например '6378' (строка или int).
    """
    if len(filename) < 2:
        raise ValueError(f"filename too short: {filename!r}")
    shard = filename[:2].lower()
    return f"https://photo.allzap.site/{shard}/unpacked/{archive_id}/{filename}"
```

## PHP-эквивалент

```php
function photoUrl(string $filename, $archiveId): string {
    if (strlen($filename) < 2) {
        throw new InvalidArgumentException("filename too short");
    }
    $shard = strtolower(substr($filename, 0, 2));
    return "https://photo.allzap.site/{$shard}/unpacked/{$archiveId}/{$filename}";
}
```

## JavaScript / TypeScript

```typescript
function photoUrl(filename: string, archiveId: string | number): string {
    if (filename.length < 2) throw new Error(`filename too short: ${filename}`);
    const shard = filename.slice(0, 2).toLowerCase();
    return `https://photo.allzap.site/${shard}/unpacked/${archiveId}/${filename}`;
}
```

## Почему такая схема

1. **Hash-shard `<hash>/` в начале** — Cloudflare R2 distributes load по prefix.
   256 шард (`00`-`ff`) дают равномерную нагрузку без bottleneck'а на одном
   prefix'e. Имена SHA1 уже равномерно распределены — первые 2 hex-символа
   как раз дают идеальное разбиение.

2. **`unpacked/` в середине** — отделяет распакованные фото от других данных
   в bucket'е (futureproofing: если будем класть и другие категории).

3. **`<archive_id>/` группирует** все файлы из одного исходного `<N>.7z`
   архива — полезно для batch-operations и debug.

4. **Имя файла = SHA1** — уже было в исходных архивах, идемпотентность,
   нет коллизий.

## CDN и кэширование

- **Cloudflare CDN автоматически кэширует** все запросы к `photo.allzap.site`
  на edge-серверах по всему миру. Первый запрос фото — миллисекунды от R2;
  второй и далее — кэш.
- **Без аутентификации** — публично читаемые URL.
- **HTTPS включён** (Let's Encrypt SSL от Cloudflare).
- **Free egress** — раздача в Google Merchant и пользователям бесплатна.

## Что НЕ нужно делать

- ❌ Не использовать прямой R2 URL вида
  `https://9e59bb89d71eac740db5f3dcf004f847.r2.cloudflarestorage.com/...` —
  он приватный, не работает без auth, и не идёт через CDN.
- ❌ Не сохранять `r2.dev` URL (для отладки можно, для прода нет).
- ❌ Не строить URL руками без функции — легко забыть `[:2].lower()` или
  перепутать порядок частей.

## Тестовый URL

Заведомо работает:
```
https://photo.allzap.site/test/hello.txt   → "hello R2 from foto-klass setup"
```

## Status (на момент 2026-05-19)

- Идёт распаковка 1229 архивов из Hetzner → R2 (5 VPS параллельно).
- Throughput ~370 fps, ETA ~12 часов.
- К концу распаковки в R2 будут все ~17M фото в указанной схеме.
