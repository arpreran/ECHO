"""Text normalisation.

basic_normalize: script-agnostic cleaning used by every configuration.
romanized_fold:  a rule-based spelling-variant key for Romanized Indic words
                 (e.g. "hengidiyaa" / "hengidiya" / "hengidiyaaa" -> same key).
language_aware_normalize: applies romanized_fold only to tokens that the
                 word-level LID tags as Romanized Kannada/Tamil/Malayalam/Hindi,
                 leaving English and native-script tokens untouched.

The fold rules are deliberately small and documented; their effect on
sentiment accuracy is measured in the ablation experiments, not assumed.
"""
from __future__ import annotations

import re
import unicodedata

from .script import JOINERS, token_script, tokenize

URL_RE = re.compile(r"^(https?://|www\.)", re.I)
ELONGATION_RE = re.compile(r"(.)\1{2,}")
RUN_RE = re.compile(r"([a-z])\1+")

# Applied in order, on lowercase Latin tokens only.
FOLD_RULES: tuple[tuple[str, str], ...] = (
    ("ph", "f"),
    ("w", "v"),
    ("q", "k"),
    ("ck", "k"),
    ("z", "j"),   # "aaz"/"aaj" (hi); kept although it also merges Tamil "zh"
)

FOLD_RULES_DOC = [
    "Lowercase; Unicode NFC.",
    "Collapse any run of a repeated letter to one letter (aa->a, nn->n, iii->i).",
    *[f"Replace '{a}' with '{b}'." for a, b in FOLD_RULES],
]


def basic_normalize_token(tok: str) -> str:
    if URL_RE.match(tok):
        return "<url>"
    if tok.startswith("@") and len(tok) > 1:
        return "<user>"
    if tok.isdigit():
        return "<num>"
    tok = tok.lower()
    # Reduce elongations inside a word to two characters: "superrrr" -> "superr".
    # (Punctuation runs are already split into single-character tokens.)
    return ELONGATION_RE.sub(r"\1\1", tok)


def basic_tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text)
    return [basic_normalize_token(t) for t in tokenize(text)]


def basic_normalize(text: str) -> str:
    return " ".join(basic_tokens(text))


def romanized_fold(tok: str) -> str:
    t = tok.lower()
    t = RUN_RE.sub(r"\1", t)
    for a, b in FOLD_RULES:
        t = t.replace(a, b)
    return t


def strip_joiners(tok: str) -> str:
    return "".join(c for c in tok if c not in JOINERS)


ROMANIZED_INDIC = {"kn_rom", "ta_rom", "ml_rom", "hi_rom"}


def language_aware_tokens(text: str, word_tagger) -> list[str]:
    """Tokens after language-aware normalisation.

    word_tagger(tokens) -> list of language tags (see lid.WordLID.tag_tokens).
    """
    toks = basic_tokens(text)
    tags = word_tagger(toks)
    out = []
    for tok, tag in zip(toks, tags):
        if tag in ROMANIZED_INDIC:
            out.append(romanized_fold(tok))
        elif token_script(tok) in ("Kannada", "Tamil", "Malayalam", "Devanagari"):
            out.append(strip_joiners(tok))
        else:
            out.append(tok)
    return out
