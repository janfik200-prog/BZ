"""Буквы со сдвигом: шрифт рисует русские буквы, а коды у них сдвинуты на 0x1D6.

Примеры выдуманные, но сдвинуты так же, как таблица запасов из закрытой БЗ (29.10.2026):
«Категория» читается «Ʉɚɬɟɝɨɪɢɹ»."""

from __future__ import annotations

from types import SimpleNamespace

from georag.parse.orchestrator import run, words


def shift(text: str) -> str:
    """Как пишет такой шрифт: буквы — со сдвигом, остальное как есть."""
    return "".join(chr(ord(ch) - 0x1D6) if "А" <= ch <= "я" or ch in "Ёё" else ch for ch in text)


TABLE = [
    "Категория запасов",
    "Зона насыщения",
    "Начальные балансовые запасы нефти",
    "Коэффициент пористости, доли",
    "НЗ",
    "тыс\x11м",
]


def test_shift_example_matches_real_pdf() -> None:
    assert shift("Категория") == "Ʉɚɬɟɝɨɪɢɹ"
    assert shift("Зона насыщения") == "Ɂɨɧɚ ɧɚɫɵɳɟɧɢɹ"


def test_letters_restored() -> None:
    texts = [shift(t) for t in TABLE]
    assert words.shifted_letters(texts) >= words.SHIFTED_MIN
    assert [words.unshift_cyrillic(t) for t in texts] == [
        "Категория запасов",
        "Зона насыщения",
        "Начальные балансовые запасы нефти",
        "Коэффициент пористости, доли",
        "НЗ",  # короткая ячейка — тоже: решает документ, а не ячейка
        "тыс м",
    ]


def test_dots_numbers_and_latin_kept() -> None:
    """Точки не трогаются: «Катего-.рия.запасов» — так docling склеивает шапку таблицы в
    несколько строк, «т.е.» и «скв.620» — настоящие точки."""
    for text in ("Катего-.рия.запасов", "т.е. скв.620, F = 2410. , h"):
        assert words.unshift_cyrillic(shift(text)) == text


def test_ipa_word_in_ordinary_text_untouched() -> None:
    """Одно слово в транскрипции МФА — не сдвиг: у документа таких букв мало."""
    texts = ["The vowel [ɪ] and schwa [ə] in English.", "Обычный русский текст статьи."]
    assert words.shifted_letters(texts) == 0


def test_ipa_heavy_text_is_not_russian() -> None:
    """Много МФА — но после сдвига гласных не как у русского текста: не трогается."""
    texts = ["ɪɪɪɪ ɛɛɛɛ ɪɛɪɛ ɪɪɛɛ ɪɛɛɪ ɛɪɪɛ"] * 3
    assert words.shifted_letters(texts) == 0


def test_ordinary_paragraph_with_few_shifted_untouched() -> None:
    """В абзаце обычного текста одна «ɪ» — меньше половины букв: абзац как был."""
    text = "Обычный абзац с транскрипцией [ɪ] внутри"
    assert words.unshift_cyrillic(text) == text


def test_orchestrator_fixes_paragraphs_and_cells() -> None:
    paragraphs = [SimpleNamespace(text=shift(t), label="text") for t in TABLE[:3]]
    cell = SimpleNamespace(text=shift("Коэффициент пористости"))
    table = SimpleNamespace(data=SimpleNamespace(table_cells=[cell]))
    doc = SimpleNamespace(texts=paragraphs, tables=[table])
    [finding] = run._shifted(doc, None)
    assert finding.case == "буквы со сдвигом" and finding.found
    assert "исправлено абзацев и ячеек: 4" in finding.detail
    assert paragraphs[0].text == "Категория запасов"
    assert cell.text == "Коэффициент пористости"


def test_orchestrator_quiet_on_normal_text() -> None:
    paragraph = SimpleNamespace(text="Категория запасов", label="text")
    doc = SimpleNamespace(texts=[paragraph], tables=[])
    [finding] = run._shifted(doc, None)
    assert not finding.found and doc.texts[0].text == "Категория запасов"
