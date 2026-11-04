"""Code-mixing statistics computed from word-level language tags.

Script and language are kept apart: 'kn_rom' and 'kn' are the same language
(Kannada) in different scripts, so mixing indices use the base language.

Metrics:
  CMI      - Code-Mixing Index, Das & Gambäck (2014).
  M-index  - Multilingual index, Barnett et al. (2000).
  I-index  - Integration index (switch points / (n-1)), Guzmán et al. (2017).
"""
from __future__ import annotations

from collections import Counter

BASE = {"en": "en", "kn": "kn", "kn_rom": "kn", "ta": "ta", "ta_rom": "ta",
        "ml": "ml", "ml_rom": "ml", "hi": "hi", "hi_rom": "hi"}


def stats(tags: list[str]) -> dict:
    n = len(tags)
    lang_tags = [BASE[t] for t in tags if t in BASE]
    u = n - len(lang_tags)  # language-independent / unsupported tokens
    counts = Counter(lang_tags)
    total = len(lang_tags)
    dist = {k: round(v / total, 4) for k, v in counts.most_common()} if total else {}

    cmi = 0.0
    if total:
        cmi = 100.0 * (1 - max(counts.values()) / total)

    k = len(counts)
    m_index = 0.0
    if k > 1:
        s = sum(p * p for p in dist.values())
        m_index = (1 - s) / ((k - 1) * s)

    switches = sum(1 for a, b in zip(lang_tags, lang_tags[1:]) if a != b)
    i_index = switches / (total - 1) if total > 1 else 0.0

    # Spans of consecutive same-language tokens (over the language tokens only).
    spans: list[dict] = []
    for t in lang_tags:
        if spans and spans[-1]["lang"] == t:
            spans[-1]["length"] += 1
        else:
            spans.append({"lang": t, "length": 1})

    return {
        "tokens": n,
        "language_tokens": total,
        "language_independent_tokens": u,
        "language_distribution": dist,
        "languages": list(counts),
        "is_code_mixed": k > 1,
        "cmi": round(cmi, 2),
        "m_index": round(m_index, 4),
        "i_index": round(i_index, 4),
        "switch_points": switches,
        "spans": spans,
    }


def scripts_mixing(tags_with_scripts: list[tuple[str, str]]) -> dict:
    """Script distribution over language tokens (Latin vs native)."""
    c = Counter(s for t, s in tags_with_scripts if t in BASE)
    total = sum(c.values())
    return {k: round(v / total, 4) for k, v in c.items()} if total else {}
