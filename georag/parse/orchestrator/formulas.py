"""Формулы: чистка LaTeX, который выдала модель формул docling.

Отдельно от docling.py: здесь только регулярные выражения — чистка работает в основном
процессе (оркестратор), куда тяжёлый docling не тянем.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
# Номер формулы, который модель дописывает в LaTeX («\quad \ \ ( 8 )», «& & ( 1 3 )»),
# и всё после него: там бывает мусор («\quad p a r { x }»).
_EQ_NUMBER = re.compile(r"(?:\s*(?:\\q?quad|\\ |&))+\s*\(\s*\d(?:\s*\d)*\s*\)[\s\S]*$")
# Модель дочитывает после формулы соседнюю строку текста: «\\ \intertext { f l e t h r
# o s …}» (бывает до предела длины ответа) или «\\ \text{simplified data through …}».
# Это не формула: в статье Nature (13.10.2026) так в 5 формулах из 13. Между переносом
# и \text бывает выравнивание: «\\ & \text {derivation} …» (статья FMM, 14.10.2026).
_INTERTEXT = re.compile(
    r"\s*(?:\\\\\s*)?(?:&\s*)*\\intertext\b[\s\S]*$|\s*\\\\\s*(?:&\s*)*\\text\b[\s\S]*$"
)
_ALIGN = re.compile(r"\s*(?<!\\)&\s*")  # выравнивание «a & = b» — в одной формуле не нужно
_TRAILING_BREAK = re.compile(r"(?:\s*\\\\)+\s*$")
_ORIG_NUMBER = re.compile(r"\s*\(\s*\d+\s*\)\s*$")
# Номер формулы в тексте PDF: у одной строки — в конце, у системы строк — посередине
# («… Southlist(c); (31) c ∈ Eastlist(b) …»). «f(2)» — не номер: перед скобкой буква.
_ORIG_TAG = re.compile(r"(?<![\w.])\(\s*(\d{1,3})\s*\)(?![\w(])")
# Что дописывает сама чистка: текст из PDF и номер. Повторная чистка (перечанкинг)
# снимает их и ставит заново — иначе «(2) (2)». Старый вид номера (\tag) — тоже.
_OURS = re.compile(
    r"(?:\s*\\quad \\text\{\[.*\]\})?(?:\s*(?:\\tag\{\d{1,3}\}|\\quad \(\d{1,3}\)))?\s*$"
)


@dataclass
class FormulaFix:
    """Чистая формула и что с ней сделано — для отчёта оркестратора."""

    text: str
    tail: bool = False  # убрана соседняя строка, которую модель дописала к формуле
    number: str = ""  # номер из текста PDF — «(N)» в конце формулы
    russian: bool = False  # приписан текст из PDF с русскими обозначениями
    from_text: bool = False  # модель ошиблась — LaTeX собран из текста PDF


# Модель формул иногда выдаёт не ту формулу: «Σ = Σᵢ xᵢ = ∫₀ᴸ g(x) dx» → «\hat { a }»
# (тест-план 8.10). Текст формулы в PDF при этом есть. Знак, который в нём есть, должен
# быть и в LaTeX; нет — или LaTeX вчетверо короче текста — LaTeX собирается из текста.
_OPERATORS = {
    "∑": (r"\sum",), "Σ": (r"\Sigma", r"\sum"), "∫": (r"\int",), "∏": (r"\prod",),
    "√": (r"\sqrt",), "∂": (r"\partial",), "∮": (r"\oint",),
}  # fmt: skip
_TEX_COMMAND = re.compile(r"\\[a-zA-Z]+|[{}_^\s\\,]")
# Пределы в тексте PDF: верхний — перед знаком, нижний — после («n∑ i =1»); у интеграла
# оба после, верхний первым («∫ L 0 g(x) dx»).
_SUM_LIMITS = re.compile(r"(\w{1,3})?\s*([∑∏])\s*(\w)\s*=\s*(\w{1,3})")
_INT_LIMITS = re.compile(r"([∫∮])\s+([\w∞]{1,3})\s+([\w∞]{1,3})\s")
_UNICODE_TEX = {
    "∑": r"\sum", "∏": r"\prod", "∫": r"\int", "∮": r"\oint", "√": r"\sqrt", "∂": r"\partial",
    "∞": r"\infty", "≤": r"\le", "≥": r"\ge", "≠": r"\ne", "≈": r"\approx", "±": r"\pm",
    "×": r"\times", "·": r"\cdot", "→": r"\to", "∈": r"\in", "Δ": r"\Delta", "Σ": r"\Sigma",
    "Ω": r"\Omega", "Φ": r"\Phi", "Π": r"\Pi", "Γ": r"\Gamma", "Λ": r"\Lambda", "Θ": r"\Theta",
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta", "ε": r"\varepsilon",
    "ζ": r"\zeta", "η": r"\eta", "θ": r"\theta", "κ": r"\kappa", "λ": r"\lambda", "μ": r"\mu",
    "ν": r"\nu", "ξ": r"\xi", "π": r"\pi", "ρ": r"\rho", "σ": r"\sigma", "τ": r"\tau",
    "φ": r"\varphi", "χ": r"\chi", "ψ": r"\psi", "ω": r"\omega",
}  # fmt: skip
_BIG = "∑∏∫∮√∂"  # после команды знака — пробел: «\sum x», а не «\sumx»


def _sum_limits(m: re.Match[str]) -> str:
    upper = f"^{{{m.group(1)}}}" if m.group(1) else ""
    return f"{m.group(2)}_{{{m.group(3)}={m.group(4)}}}{upper} "


def lost_by_model(latex: str, orig: str) -> bool:
    """LaTeX модели не та формула: знака из текста PDF в нём нет, или он вчетверо короче."""
    if any(sign in orig and not any(t in latex for t in tex) for sign, tex in _OPERATORS.items()):
        return True
    body = _TEX_COMMAND.sub("", latex)
    text = re.sub(r"\s|\(\s*\d+\s*\)", "", orig)
    return len(text) >= 8 and len(body) * 4 < len(text)


def from_text(orig: str) -> str:
    """LaTeX из текста формулы в PDF: знаки — командами, пределы суммы и интеграла —
    индексами. Расположение (дроби, степени) текст PDF не сохраняет — это лучшее, что
    из него собирается, и всё же та формула, а не чужая."""
    text = _SUM_LIMITS.sub(_sum_limits, orig)
    text = _INT_LIMITS.sub(lambda m: f"{m.group(1)}_{{{m.group(3)}}}^{{{m.group(2)}}} ", text)
    for sign, tex in _UNICODE_TEX.items():
        text = text.replace(sign, f"{tex} " if sign in _BIG else tex)
    return " ".join(text.split())


def fix_formula(latex: str, orig: str) -> FormulaFix:
    """LaTeX формулы без выравнивания и мусора после формулы. Номер — «\\quad (N)» из
    текста PDF: модель его не видит (рамка узкая), а текст на него ссылается («(55)»).
    Русские обозначения модель читает латиницей («V_{обр}» → «V_{op6}»), а в тексте PDF
    они верные, — тогда он приписывается к формуле: \\text{[K п = V пор V обр · 100 %]}."""
    latex = _OURS.sub("", latex or "")
    tail = bool(_INTERTEXT.search(latex))
    latex = _EQ_NUMBER.sub("", _INTERTEXT.sub("", latex))
    latex = _TRAILING_BREAK.sub("", _ALIGN.sub(" ", latex)).strip()
    if not latex:
        return FormulaFix("", tail=tail)
    orig = " ".join((orig or "").split())
    numbers = _ORIG_TAG.findall(orig)
    number = numbers[-1] if numbers else ""
    # \tag{N} просмотрщик без его поддержки показывает буквами «tag» — похоже на тангенс;
    # «(N)» после отступа виден одинаково везде (так же пишет номер Marker).
    tag = f" \\quad ({number})" if number else ""
    orig = _ORIG_NUMBER.sub("", orig)
    if orig and not _CYRILLIC.search(orig) and lost_by_model(latex, orig):
        return FormulaFix(from_text(orig) + tag, tail, number, from_text=True)
    if _CYRILLIC.search(orig):
        return FormulaFix(f"{latex} \\quad \\text{{[{orig}]}}{tag}", tail, number, russian=True)
    return FormulaFix(latex + tag, tail, number)


def tidy_formula(latex: str, orig: str) -> str:
    """Только чистый LaTeX (см. fix_formula)."""
    return fix_formula(latex, orig).text
