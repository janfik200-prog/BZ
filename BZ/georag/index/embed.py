"""Эмбеддинги BGE-M3 — через Ollama или напрямую.

Почему именно эта модель: мультиязычная (русский и английский в одном пространстве —
запрос на русском находит английскую статью), 1024 измерения, 8192 токена контекста,
и её же токенайзер уже используется при чанкинге. Токенайзер и модель обязаны
совпадать, иначе счёт токенов при нарезке врёт.

Два способа её позвать:

* **local** (по умолчанию) — sentence-transformers прямо на видеокарте. Без
  квантования и без сети: на индексации тысяч чанков заметно быстрее. Главное —
  поиск и индексация не зависят от того, запущена ли Ollama;
* **ollama** — по HTTP к уже работающей службе, ключ `--embedder ollama`. Модель
  та же, но сжатая: точность чуть ниже, на больших пачках медленнее. Смысл
  в одном — не держать sentence-transformers в питоновском окружении.

Ollama при этом остаётся нужна для языковой работы: отбор статей и, дальше,
ответы по найденному. Векторы — не языковая работа, их незачем гонять по сети.

Векторы в обоих случаях нормируются, поэтому косинус и скалярное произведение
совпадают, и в базе работает оператор <=>.

Размерность у обоих способов одна (1024), так что схема базы не меняется. Но
векторы, посчитанные разными способами, чуть-чуть различаются: если переключаете
способ, переиндексируйте всё разом (`ingest --force`), чтобы база была однородной.
"""

from __future__ import annotations

import math

VECTOR_DIM = 1024
OLLAMA_MODEL = "bge-m3"
OLLAMA_HOST = "http://localhost:11434"


def _normalize(vector: list[float]) -> list[float]:
    length = math.sqrt(sum(v * v for v in vector))
    return [v / length for v in vector] if length else vector


class Embedder:
    """BGE-M3 через sentence-transformers, на видеокарте."""

    name = "local"

    def __init__(
        self,
        model_id: str = "BAAI/bge-m3",
        device: str = "auto",
        batch_size: int = 8,
        max_length: int = 1024,
    ):
        self.model_id = model_id
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None

    # Модель весит около 2 ГБ и грузится секунды: берём её только когда реально нужна.
    def _load(self):
        if self._model is not None:
            return self._model
        from sentence_transformers import SentenceTransformer

        device = None if self.device == "auto" else self.device
        model = SentenceTransformer(self.model_id, device=device)
        model.max_seq_length = self.max_length
        self._model = model
        return model

    @property
    def dim(self) -> int:
        return VECTOR_DIM

    def encode(self, texts: list[str], progress: bool = False) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,   # косинус = скалярное произведение
            show_progress_bar=progress,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]

    def encode_one(self, text: str) -> list[float]:
        # У BGE-M3 нет отдельной инструкции для запроса: запрос и документ
        # кодируются одинаково. Добавлять префиксы от других моделей — вредно.
        return self.encode([text])[0]


class OllamaEmbedder:
    """BGE-M3 через HTTP к Ollama."""

    name = "ollama"

    def __init__(
        self,
        model_id: str = OLLAMA_MODEL,
        host: str = OLLAMA_HOST,
        batch_size: int = 16,
        timeout: int = 600,
    ):
        self.model_id = model_id
        self.host = host.rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout

    @property
    def dim(self) -> int:
        return VECTOR_DIM

    def _post(self, path: str, payload: dict) -> dict:
        import requests

        response = requests.post(f"{self.host}{path}", json=payload, timeout=self.timeout)
        if response.status_code == 404:
            raise FileNotFoundError(path)
        if response.status_code >= 400:
            raise RuntimeError(f"Ollama ответила {response.status_code}: {response.text[:200]}")
        return response.json() or {}

    def _batch(self, batch: list[str]) -> list[list[float]]:
        try:
            data = self._post("/api/embed", {"model": self.model_id, "input": batch})
            vectors = data.get("embeddings") or []
        except FileNotFoundError:
            # Старые сборки Ollama знают только /api/embeddings и только по одному тексту.
            vectors = [
                (self._post("/api/embeddings", {"model": self.model_id, "prompt": text})
                 .get("embedding") or [])
                for text in batch
            ]

        if len(vectors) != len(batch):
            raise RuntimeError(
                f"Ollama вернула {len(vectors)} векторов на {len(batch)} текстов"
            )
        for vector in vectors:
            if len(vector) != VECTOR_DIM:
                raise RuntimeError(
                    f"ожидался вектор на {VECTOR_DIM} чисел, пришёл на {len(vector)}. "
                    f"Проверьте, что модель «{self.model_id}» — это BGE-M3"
                )
        # Ollama не обещает нормированных векторов, а база считает косинус.
        return [_normalize(v) for v in vectors]

    def encode(self, texts: list[str], progress: bool = False) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            out.extend(self._batch(batch))
            if progress and len(texts) > self.batch_size:
                print(f"    векторов посчитано: {len(out)}/{len(texts)}", end="\r")
        if progress and len(texts) > self.batch_size:
            print(" " * 50, end="\r")
        return out

    def encode_one(self, text: str) -> list[float]:
        return self.encode([text])[0]


def build_embedder(
    backend: str = "local",
    model_id: str | None = None,
    device: str = "auto",
    batch_size: int = 8,
    host: str = OLLAMA_HOST,
    quiet: bool = False,
):
    """Собрать считалку векторов.

    По умолчанию — местная модель: так поиск работает независимо от того, запущена
    ли Ollama. При явном выборе ollama, если она молчит, всё равно берём местную:
    обрывать долгую индексацию из-за выключенной службы глупо.
    """
    if backend == "local":
        return Embedder(model_id or "BAAI/bge-m3", device=device, batch_size=batch_size)

    ollama = OllamaEmbedder(model_id or OLLAMA_MODEL, host=host, batch_size=max(batch_size, 8))
    try:
        ollama.encode_one("проверка связи")
        return ollama
    except Exception as exc:  # noqa: BLE001 — служба не поднята или модель не скачана
        if not quiet:
            print(f"Ollama для векторов недоступна ({type(exc).__name__}: {exc}).")
            print("Считаю модель локально. Чтобы через Ollama: ollama pull bge-m3")
        return Embedder("BAAI/bge-m3", device=device, batch_size=batch_size)
