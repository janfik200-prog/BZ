"""Получение статьи в память. На диск PDF не пишется.

Три проверки, без которых этот шаг тихо отравляет базу:

* сигнатура файла. Ссылка из метаданных с именем pdf_url очень часто отдаёт
  HTML-страницу статьи, а не файл. Смотрим на первые байты (%PDF-), а не на
  Content-Type, который у таких страниц бывает каким угодно;
* размер. Читаем потоком и обрываем, если файл больше лимита, — иначе один
  атлас на 900 МБ съест память ночного прогона;
* sha256 и итоговый URL после редиректов. Это единственное, чем потом можно
  подтвердить, что разбирали именно этот файл: самого PDF мы не храним.

Три приёма, без которых половина открытых статей не берётся:

* браузерная подпись про запас. Часть издателей отвечает 403 всему, что не похоже
  на браузер, — даже по статьям в открытом доступе;
* заход через страницу статьи. У MDPI и подобных PDF отдаётся только тому, кто
  пришёл со своей же страницы и принёс её куки. Сессия одна на обе загрузки;
* citation_pdf_url. Если вместо файла пришла HTML-страница, в её мета-тегах почти
  всегда лежит прямой адрес PDF — этот тег издатели ставят для Google Scholar.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urljoin, urlsplit

PDF_MAGIC = b"%PDF-"

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
)
BLOCKED_CODES = (401, 403, 406, 429)

_CITATION_PDF_RE = re.compile(
    rb"""<meta[^>]+citation_pdf_url[^>]*>""", re.IGNORECASE
)
_CONTENT_RE = re.compile(rb"""content\s*=\s*["']([^"']+)["']""", re.IGNORECASE)


@dataclass
class FetchResult:
    ok: bool
    url: str
    final_url: str = ""
    data: bytes | None = None
    sha256: str = ""
    bytes_len: int = 0
    content_type: str = ""
    fetched_at: str = ""
    error: str = ""


def citation_pdf_url(html: bytes, base_url: str) -> str | None:
    """Прямой адрес PDF из мета-тега страницы статьи, если он там есть."""
    tag = _CITATION_PDF_RE.search(html)
    if not tag:
        return None
    found = _CONTENT_RE.search(tag.group(0))
    if not found:
        return None
    value = found.group(1).decode("utf-8", errors="replace").strip()
    return urljoin(base_url, value) if value else None


def _same_host(a: str, b: str) -> bool:
    return bool(a) and bool(b) and urlsplit(a).netloc.lower() == urlsplit(b).netloc.lower()


def _headers(agent: str, url: str, referer: str = "") -> dict:
    parts = urlsplit(url)
    return {
        "User-Agent": agent,
        "Accept": "application/pdf,text/html;q=0.8,*/*;q=0.5",
        "Accept-Language": "en,ru;q=0.8",
        "Referer": referer or f"{parts.scheme}://{parts.netloc}/",
    }


def fetch_pdf(
    url: str,
    timeout: int = 60,
    max_mb: int = 80,
    user_agent: str = "georag/0.1 (knowledge base builder)",
    referer: str = "",
    follow_meta: bool = True,
) -> FetchResult:
    import requests

    result = FetchResult(ok=False, url=url)
    result.fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    limit = max_mb * 1024 * 1024
    session = requests.Session()

    # Заход через страницу статьи: нужен ради куки, ответ нам не интересен.
    if referer and _same_host(referer, url) and referer != url:
        try:
            session.get(
                referer,
                timeout=timeout,
                allow_redirects=True,
                headers=_headers(user_agent, referer),
            ).close()
        except Exception:  # noqa: BLE001 — не получилось, идём напрямую
            pass

    data = b""
    try:
        for agent in (user_agent, BROWSER_UA):
            with session.get(
                url,
                timeout=timeout,
                stream=True,
                allow_redirects=True,
                headers=_headers(agent, url, referer),
            ) as response:
                result.final_url = response.url
                result.content_type = (response.headers.get("Content-Type") or "").split(";")[0]

                if response.status_code != 200:
                    result.error = f"HTTP {response.status_code}"
                    if response.status_code in BLOCKED_CODES and agent != BROWSER_UA:
                        continue  # закрылись от робота — пробуем ещё раз как браузер
                    return result

                chunks: list[bytes] = []
                size = 0
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    chunks.append(chunk)
                    size += len(chunk)
                    if size > limit:
                        result.error = f"файл больше {max_mb} МБ — пропущен"
                        return result
                data = b"".join(chunks)
                result.error = ""
                break
    except Exception as exc:  # noqa: BLE001
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    if not data:
        result.error = "пустой ответ"
        return result

    if not data.lstrip()[:8].startswith(PDF_MAGIC):
        # Пришла страница статьи. У неё в мета-тегах обычно лежит адрес самого PDF.
        if follow_meta:
            direct = citation_pdf_url(data[:200_000], result.final_url or url)
            if direct and direct != url:
                deeper = fetch_pdf(
                    direct,
                    timeout=timeout,
                    max_mb=max_mb,
                    user_agent=user_agent,
                    referer=result.final_url or url,
                    follow_meta=False,
                )
                if deeper.ok:
                    return deeper
                result.error = f"страница статьи; её citation_pdf_url тоже не дался: {deeper.error}"
                return result

        head = data.lstrip()[:40].decode("latin-1", errors="replace")
        kind = "html-страница" if b"<html" in data[:2048].lower() else "неизвестный формат"
        result.error = f"по ссылке не PDF ({kind}): начало {head!r}"
        return result

    result.ok = True
    result.data = data
    result.bytes_len = len(data)
    result.sha256 = hashlib.sha256(data).hexdigest()
    return result
