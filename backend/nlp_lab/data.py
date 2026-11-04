"""DravidianCodeMix-2020 sentiment loader (Chakravarthi et al., 2022; CC BY 4.0).

The release uses two layouts:
  kannada / malayalam:  text;label;
  tamil:                text<TAB>label
Lines are parsed from the right so that ';' or tabs inside comments survive.
"""
from __future__ import annotations

import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

from .paths import DATASETS_DIR
from .script import script_profile

DCM_DIR = DATASETS_DIR / "raw" / "dravidiancodemix" / "DravidianCodeMix"

FILES = {
    "kn": {s: f"kannada_sentiment_{s}.csv" for s in ("train", "dev", "test")},
    "ml": {s: f"mal_full_sentiment_{s}.csv" for s in ("train", "dev", "test")},
    "ta": {s: f"tamil_sentiment_full_{s}.csv" for s in ("train", "dev", "test")},
}

# Raw label -> research label. Documented in the paper (Section: Dataset).
#  - unknown_state  : annotators found no sentiment expressed -> neutral
#  - Mixed_feelings : both polarities; excluded from the 3-class task
#  - not-<language> : comment is not in the target language; excluded from
#                     sentiment but used as negative evidence nowhere else.
LABEL_MAP = {
    "positive": "positive",
    "negative": "negative",
    "unknown_state": "neutral",
    "unknown state": "neutral",
    "mixed_feelings": None,
    "mixed feelings": None,
}


def canonical_raw_label(raw: str) -> str:
    return raw.strip().strip(";").strip().lower()


def map_label(raw: str) -> str | None:
    lab = canonical_raw_label(raw)
    if lab.startswith("not-") or lab.startswith("not "):
        return None
    return LABEL_MAP.get(lab)


def parse_line(line: str) -> tuple[str, str] | None:
    line = line.rstrip("\r\n")
    if not line.strip():
        return None
    if "\t" in line:
        text, _, label = line.rpartition("\t")
    else:
        body = line[:-1] if line.endswith(";") else line
        text, _, label = body.rpartition(";")
    if not text:
        return None
    return text.strip(), label.strip()


def read_split(lang: str, split: str) -> pd.DataFrame:
    path = DCM_DIR / FILES[lang][split]
    rows = []
    with open(path, encoding="utf-8-sig", errors="strict") as f:
        for i, line in enumerate(f):
            parsed = parse_line(line)
            if parsed is None:
                continue
            text, raw = parsed
            if i == 0 and canonical_raw_label(raw) in ("category", "label"):
                continue  # header row
            rows.append({"text": unicodedata.normalize("NFC", text), "raw_label": raw})
    df = pd.DataFrame(rows)
    df["lang"] = lang
    df["split"] = split
    df["label"] = df["raw_label"].map(map_label)
    return df


def load_all() -> pd.DataFrame:
    frames = [read_split(l, s) for l in FILES for s in ("train", "dev", "test")]
    df = pd.concat(frames, ignore_index=True)
    df["text_type"] = df["text"].map(lambda t: script_profile(t)["text_type"])
    return df


def dedup_key(text: str) -> str:
    return " ".join(text.lower().split())


def clean_for_sentiment(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep 3-class rows; remove duplicates and cross-split leakage.

    Rules (per language):
      1. drop rows whose label maps to None (mixed / not-<lang>);
      2. inside a split, drop exact duplicate (normalised text) rows; if the
         duplicates disagree on label, drop all of them;
      3. drop dev/test rows whose normalised text also occurs in train, and
         test rows that occur in dev.
    """
    report: dict = {}
    out = []
    for lang, g in df.groupby("lang"):
        r = {"input": len(g)}
        g = g[g["label"].notna()].copy()
        r["after_label_filter"] = len(g)
        g["key"] = g["text"].map(dedup_key)
        conflicts = g.groupby(["split", "key"])["label"].nunique()
        conflict_keys = set(conflicts[conflicts > 1].index)
        r["conflicting_duplicates_removed"] = int(g.apply(lambda x: (x["split"], x["key"]) in conflict_keys, axis=1).sum())
        g = g[~g.apply(lambda x: (x["split"], x["key"]) in conflict_keys, axis=1)]
        before = len(g)
        g = g.drop_duplicates(subset=["split", "key"])
        r["within_split_duplicates_removed"] = before - len(g)
        train_keys = set(g.loc[g.split == "train", "key"])
        dev_keys = set(g.loc[g.split == "dev", "key"])
        leak = ((g.split != "train") & g.key.isin(train_keys)) | ((g.split == "test") & g.key.isin(dev_keys))
        r["cross_split_leakage_removed"] = int(leak.sum())
        g = g[~leak]
        r["final"] = len(g)
        r["final_by_split"] = g.split.value_counts().to_dict()
        report[lang] = r
        out.append(g.drop(columns=["key"]))
    return pd.concat(out, ignore_index=True), report


def audit(df: pd.DataFrame) -> dict:
    """Aggregate statistics only (no comment text)."""
    res = {}
    for lang, g in df.groupby("lang"):
        res[lang] = {
            "rows": len(g),
            "by_split": g.split.value_counts().to_dict(),
            "raw_labels": dict(Counter(g.raw_label.map(canonical_raw_label)).most_common()),
            "text_type": g.text_type.value_counts().to_dict(),
        }
    return res
