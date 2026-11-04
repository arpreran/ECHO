"""Deterministic Unicode script identification.

Script is decided from Unicode code-point blocks, independently of language.
Language (e.g. Romanized Kannada vs English) is a separate, model-based step
(see lid.py), because Latin script alone does not identify a language.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

# (name, first code point, last code point)
SCRIPT_BLOCKS: tuple[tuple[str, int, int], ...] = (
    ("Devanagari", 0x0900, 0x097F),
    ("Tamil", 0x0B80, 0x0BFF),
    ("Kannada", 0x0C80, 0x0CFF),
    ("Malayalam", 0x0D00, 0x0D7F),
)

NATIVE_SCRIPT_LANG = {
    "Kannada": "kn",
    "Tamil": "ta",
    "Malayalam": "ml",
    "Devanagari": "hi",
}

# Joiners are part of Indic orthography and must not split a word.
JOINERS = {"‌", "‍"}

TOKEN_RE = re.compile(
    r"https?://\S+|www\.\S+"          # URLs
    r"|@\w+|#\w+"                       # mentions / hashtags
    r"|[\w‌‍ऀ-ൿ']+"  # words incl. Indic marks and joiners
    r"|[^\w\s]",                        # single punctuation / emoji / symbols
    re.UNICODE,
)


def char_script(ch: str) -> str:
    """Return the script name of a single character."""
    if ch in JOINERS:
        return "Joiner"
    cp = ord(ch)
    for name, lo, hi in SCRIPT_BLOCKS:
        if lo <= cp <= hi:
            return name
    if ch.isascii() and ch.isalpha():
        return "Latin"
    if ch.isdigit():
        return "Digit"
    if ch.isspace():
        return "Space"
    cat = unicodedata.category(ch)
    if cat.startswith("L"):
        # Accented Latin letters etc.
        try:
            if "LATIN" in unicodedata.name(ch):
                return "Latin"
        except ValueError:
            pass
        return "OtherLetter"
    if cat in ("So", "Sk", "Cs") or 0x1F000 <= cp <= 0x1FAFF or 0x2600 <= cp <= 0x27BF:
        return "Emoji"
    return "Punct"


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def token_script(token: str) -> str:
    """Dominant script of a token; 'Mixed' if two letter scripts co-occur."""
    counts = Counter(char_script(c) for c in token)
    for k in ("Joiner", "Space"):
        counts.pop(k, None)
    letters = {k: v for k, v in counts.items() if k in NATIVE_SCRIPT_LANG or k in ("Latin", "OtherLetter")}
    if token.startswith(("http://", "https://", "www.")):
        return "URL"
    if len(letters) > 1:
        return "Mixed"
    if letters:
        return next(iter(letters))
    if counts.get("Digit"):
        return "Digit"
    if counts.get("Emoji"):
        return "Emoji"
    return "Punct"


def script_profile(text: str) -> dict:
    """Character-level script distribution over letters, plus a text type.

    text_type is one of: native (only one Indic script), romanized (only Latin
    letters), mixed_script (Latin and Indic letters), other.
    """
    counts = Counter(char_script(c) for c in text)
    letter_scripts = {k: v for k, v in counts.items() if k in NATIVE_SCRIPT_LANG or k == "Latin"}
    total = sum(letter_scripts.values())
    dist = {k: round(v / total, 4) for k, v in sorted(letter_scripts.items(), key=lambda kv: -kv[1])} if total else {}
    indic = [k for k in letter_scripts if k in NATIVE_SCRIPT_LANG]
    if not total:
        text_type = "other"
    elif indic and "Latin" in letter_scripts:
        text_type = "mixed_script"
    elif indic:
        text_type = "native"
    else:
        text_type = "romanized"
    dominant = max(letter_scripts, key=letter_scripts.get) if total else None
    return {
        "distribution": dist,
        "dominant_script": dominant,
        "text_type": text_type,
        "letter_count": total,
        "emoji_count": counts.get("Emoji", 0),
    }
