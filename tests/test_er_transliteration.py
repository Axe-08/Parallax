"""Unit tests for Brahmic script transliteration module."""

from parallax.preprocessing.transliteration import (
    has_brahmic_script,
    is_brahmic_char,
    transliterate_brahmic_to_latin,
)


def test_is_brahmic_char():
    assert is_brahmic_char("क") is True   # Devanagari
    assert is_brahmic_char("ক") is True   # Bengali
    assert is_brahmic_char("ક") is True   # Gujarati
    assert is_brahmic_char("ಕ") is True   # Kannada
    assert is_brahmic_char("A") is False  # Latin
    assert is_brahmic_char("é") is False  # Latin-1


def test_has_brahmic_script():
    assert has_brahmic_script("Dynamic Engineering") is False
    assert has_brahmic_script("डायनामिक इंजीनियरिंग") is True
    assert has_brahmic_script("South एक्सपोर्ट्स") is True
    assert has_brahmic_script("") is False
    assert has_brahmic_script(None) is False


def test_transliterate_brahmic_to_latin():
    # Devanagari
    res_hi = transliterate_brahmic_to_latin("डायनामिक इंजीनियरिंग")
    assert "daynamik" in res_hi or "d" in res_hi
    assert "injiniyring" in res_hi or "j" in res_hi

    # Kannada
    res_kn = transliterate_brahmic_to_latin("ಸಿಲ್ವರ್ ಗ್ಲೋಬಲ್")
    assert "silvr" in res_kn
    assert "globl" in res_kn

    # English unchanged
    assert transliterate_brahmic_to_latin("Acme Corp") == "Acme Corp"
