"""Почти-дубли фрагментов: повтор модели не отдаётся (план графа v2, А1).

Одна и та же статья из двух источников, шаблонные куски (лицензия, «Конфликт
интересов») — модель выписывала бы из них те же факты второй раз. Отпечаток
фрагмента — MinHash по пятёркам слов: сходство двух отпечатков ≈ доля общих пятёрок
(мера Жаккара). Похожие ищутся не перебором, а по полосам подписи (LSH): совпала хоть
одна полоса — кандидат, и только для кандидатов сходство считается по всей подписи.

Отпечатки лежат в базе (store: chunk_minhash, chunk_bands) — «первый» фрагмент
дубля тот, что прошёл раньше; graph --redo их стирает, и порядок задаёт новый проход.
Без внешних библиотек: хэши — blake2b, перестановки — numpy.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

import numpy as np

NUM_PERM = 64
BANDS = 16  # 16 полос по 4 значения: кандидатом становится пара со сходством от ~0,5
ROWS = NUM_PERM // BANDS
SHINGLE = 5  # слов в пятёрке
MIN_WORDS = 20  # короче — отпечаток ненадёжен, дублем не считается
DUP_SIMILARITY = 0.8  # с какой доли общих пятёрок фрагмент — дубль

_PRIME = (1 << 31) - 1
_rng = np.random.default_rng(20261027)  # одни и те же перестановки при каждом запуске
_A = _rng.integers(1, _PRIME, NUM_PERM, dtype=np.int64)
_B = _rng.integers(0, _PRIME, NUM_PERM, dtype=np.int64)
_WORD = re.compile(r"[a-zа-яё0-9]+")


def signature(text: str) -> list[int] | None:
    """MinHash-подпись фрагмента; None — слов меньше MIN_WORDS."""
    words = _WORD.findall((text or "").lower().replace("ё", "е"))
    if len(words) < MIN_WORDS:
        return None
    shingles = {" ".join(words[i : i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)}
    x = np.fromiter(
        (
            int.from_bytes(hashlib.blake2b(s.encode(), digest_size=4).digest(), "little") % _PRIME
            for s in shingles
        ),
        dtype=np.int64,
        count=len(shingles),
    )
    # (a·x + b) mod p по каждой перестановке; минимум по пятёркам — значение подписи.
    values = (_A[:, None] * x[None, :] + _B[:, None]) % _PRIME
    return [int(v) for v in values.min(axis=1)]


def similarity(a: list[int], b: list[int]) -> float:
    """Оценка доли общих пятёрок по двум подписям."""
    return sum(x == y for x, y in zip(a, b, strict=True)) / NUM_PERM


def bands(sig: list[int]) -> list[int]:
    """Хэш каждой полосы подписи — ключ для поиска кандидатов индексом."""
    out = []
    for band in range(BANDS):
        part = ",".join(str(v) for v in sig[band * ROWS : (band + 1) * ROWS])
        out.append(
            int.from_bytes(
                hashlib.blake2b(part.encode(), digest_size=8).digest(), "big", signed=True
            )
        )
    return out


def find_duplicate(conn: Any, chunk_id: int, text: str) -> int | None:
    """Фрагмент — почти-дубль уже пройденного? Вернуть его id; иначе запомнить отпечаток
    этого фрагмента (он сам станет «первым» для следующих) и вернуть None."""
    sig = signature(text)
    if sig is None:
        return None
    keys = bands(sig)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT m.chunk_id, m.sig FROM chunk_bands b
            JOIN chunk_minhash m ON m.chunk_id = b.chunk_id
            WHERE (b.band, b.hash) IN (SELECT * FROM unnest(%s::smallint[], %s::bigint[]))
              AND b.chunk_id <> %s
            """,
            (list(range(BANDS)), keys, chunk_id),
        )
        best = None
        for other, other_sig in cur.fetchall():
            score = similarity(sig, list(other_sig))
            if score >= DUP_SIMILARITY and (best is None or score > best[1]):
                best = (int(other), score)
        if best is not None:
            return best[0]
        cur.execute(
            "INSERT INTO chunk_minhash (chunk_id, sig) VALUES (%s, %s) "
            "ON CONFLICT (chunk_id) DO UPDATE SET sig = EXCLUDED.sig",
            (chunk_id, sig),
        )
        cur.execute("DELETE FROM chunk_bands WHERE chunk_id = %s", (chunk_id,))
        cur.executemany(
            "INSERT INTO chunk_bands (chunk_id, band, hash) VALUES (%s, %s, %s)",
            [(chunk_id, band, key) for band, key in enumerate(keys)],
        )
    return None
