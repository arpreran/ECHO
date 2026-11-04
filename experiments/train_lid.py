"""Train and evaluate the word- and sentence-level romanized LID models.

Run from the project root:
    .venv\\Scripts\\python experiments\\train_lid.py

Outputs:
    models/lid/word_lid.joblib, models/lid/sentence_lid.joblib
    experiments/results/lid_results.json
"""
from __future__ import annotations

import bz2
import hashlib
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score  # noqa: E402

from nlp_lab.data import load_all  # noqa: E402
from nlp_lab.lid import LATIN_CLASSES, LIDSystem, make_sentence_pipeline, make_word_pipeline  # noqa: E402
from nlp_lab.paths import DATASETS_DIR, MODELS_DIR, RESULTS_DIR, SEED  # noqa: E402
from nlp_lab.script import token_script, tokenize  # noqa: E402

DAK = DATASETS_DIR / "raw" / "dakshina" / "dakshina_dataset_v1.0"
TATOEBA = DATASETS_DIR / "raw" / "tatoeba" / "eng_sentences.tsv.bz2"
INDIC = {"kn": "kn_rom", "ta": "ta_rom", "ml": "ml_rom", "hi": "hi_rom"}
ALPHA = re.compile(r"^[a-z]+$")
rng = np.random.default_rng(SEED)


def hash_split(s: str) -> str:
    h = int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % 10
    return "train" if h < 8 else ("dev" if h == 8 else "test")


# ---------------------------------------------------------------- data
def load_tatoeba() -> list[str]:
    with bz2.open(TATOEBA, "rt", encoding="utf-8") as f:
        sents = [line.rstrip("\n").split("\t", 2)[2] for line in f]
    return [s for s in sents if s.isascii()]


def word_data(eng_sents: list[str]) -> pd.DataFrame:
    rows = []
    for lang, tag in INDIC.items():
        for split in ("train", "dev", "test"):
            df = pd.read_csv(DAK / lang / "lexicons" / f"{lang}.translit.sampled.{split}.tsv", sep="\t",
                             header=None, names=["native", "roman", "count"], quoting=3, dtype=str)
            # sorted(): set iteration order varies between processes (hash randomisation),
            # and training-row order slightly changes the fitted model.
            words = sorted({str(w).lower() for w in df["roman"].dropna() if ALPHA.match(str(w).lower())})
            rows += [{"word": w, "lang": tag, "split": split, "source": "dakshina-lexicon"} for w in words]
    counts = Counter(w for s in eng_sents for w in re.findall(r"[a-z]+", s.lower()))
    eng = [w for w, c in counts.items() if c >= 3 and len(w) >= 2]
    rows += [{"word": w, "lang": "en", "split": hash_split(w), "source": "tatoeba-vocab"} for w in eng]
    return pd.DataFrame(rows)


def latin_only(text: str) -> str:
    return " ".join(t for t in tokenize(text) if token_script(t) == "Latin")


def sentence_data(eng_sents: list[str]) -> pd.DataFrame:
    rows = []
    # Dakshina romanized Wikipedia sentences: dev -> train, test -> test.
    for lang, tag in INDIC.items():
        for split, target in (("dev", "train"), ("test", "test")):
            with open(DAK / lang / "romanized" / f"{lang}.romanized.rejoined.{split}.roman.txt", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rows.append({"text": line.strip(), "lang": tag, "split": target, "source": "dakshina"})
    # DravidianCodeMix Latin-script comments, excluding "not-<lang>" rows.
    dcm = load_all()
    dcm = dcm[(dcm.text_type == "romanized") & ~dcm.raw_label.str.lower().str.startswith("not")]
    for _, r in dcm.iterrows():
        if r.split == "dev":
            continue
        rows.append({"text": r.text, "lang": INDIC[r.lang], "split": r.split, "source": "dravidiancodemix"})
    # Tatoeba English.
    idx = rng.permutation(len(eng_sents))
    for i in idx[:12000]:
        rows.append({"text": eng_sents[i], "lang": "en", "split": "train", "source": "tatoeba"})
    for i in idx[12000:15000]:
        rows.append({"text": eng_sents[i], "lang": "en", "split": "test", "source": "tatoeba"})
    df = pd.DataFrame(rows)
    df["text"] = df["text"].map(latin_only)
    df = df[df.text.str.len() > 0]
    # Cap any (source, lang) training group at 8000 to limit imbalance.
    parts = []
    for _, g in df[df.split == "train"].groupby(["source", "lang"]):
        parts.append(g.sample(n=min(len(g), 8000), random_state=SEED))
    train = pd.concat(parts)
    test = df[df.split == "test"]
    # Leakage guard: drop test sentences that also occur in train.
    test = test[~test.text.str.lower().isin(set(train.text.str.lower()))]
    return pd.concat([train, test], ignore_index=True)


# ------------------------------------------------------------ metrics
def evaluate(y_true, y_pred, labels) -> dict:
    return {
        "n": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4),
        "per_class": {k: {m: round(float(v), 4) for m, v in d.items()}
                      for k, d in classification_report(y_true, y_pred, labels=labels, output_dict=True,
                                                        zero_division=0).items() if k in labels},
        "confusion_matrix": {"labels": labels, "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()},
    }


def langid_baseline(texts: list[str]) -> list[str]:
    """Off-the-shelf langid.py (Lui & Baldwin, 2012), mapped to our classes.

    langid outputs ISO codes; anything that is not kn/ta/ml/hi/en is 'other'.
    It cannot output a 'romanized' variant, so kn/ta/ml/hi map to *_rom.
    """
    import langid
    mapping = {"en": "en", "kn": "kn_rom", "ta": "ta_rom", "ml": "ml_rom", "hi": "hi_rom"}
    return [mapping.get(langid.classify(t)[0], "other") for t in texts]


def main() -> None:
    t0 = time.time()
    out_dir = MODELS_DIR / "lid"
    out_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    eng = load_tatoeba()
    results: dict = {"seed": SEED, "classes": LATIN_CLASSES}

    # ---------------- word level
    wd = word_data(eng)
    results["word_data"] = {
        "by_split_lang": {f"{s}/{l}": int(n) for (s, l), n in wd.groupby(["split", "lang"]).size().items()},
        "ambiguous_train_strings": int(wd[wd.split == "train"].groupby("word").lang.nunique().gt(1).sum()),
    }
    tr, dv, te = (wd[wd.split == s] for s in ("train", "dev", "test"))
    best = None
    for C in (1.0, 3.0, 10.0):
        m = make_word_pipeline(C).fit(tr.word, tr.lang)
        f1 = f1_score(dv.lang, m.predict(dv.word), average="macro")
        print(f"word C={C} dev macroF1={f1:.4f}", flush=True)
        if best is None or f1 > best[0]:
            best = (f1, C, m)
    results["word_model"] = {"selected_C": best[1], "dev_macro_f1": round(best[0], 4),
                             "test": evaluate(te.lang, best[2].predict(te.word), LATIN_CLASSES)}
    word_model = best[2]
    joblib.dump(word_model, out_dir / "word_lid.joblib", compress=3)

    # ---------------- sentence level
    sd = sentence_data(eng)
    results["sentence_data"] = {f"{s}/{src}/{l}": int(n) for (s, src, l), n in
                                sd.groupby(["split", "source", "lang"]).size().items()}
    tr = sd[sd.split == "train"]
    te = sd[sd.split == "test"]
    # Hold out 10% of train for C selection.
    perm = rng.permutation(len(tr))
    cut = int(0.9 * len(tr))
    tr_a, tr_b = tr.iloc[perm[:cut]], tr.iloc[perm[cut:]]
    best = None
    for C in (3.0, 10.0, 30.0):
        m = make_sentence_pipeline(C).fit(tr_a.text, tr_a.lang)
        f1 = f1_score(tr_b.lang, m.predict(tr_b.text), average="macro")
        print(f"sent C={C} val macroF1={f1:.4f}", flush=True)
        if best is None or f1 > best[0]:
            best = (f1, C)
    sent_model = make_sentence_pipeline(best[1]).fit(tr.text, tr.lang)
    joblib.dump(sent_model, out_dir / "sentence_lid.joblib", compress=3)
    pred = sent_model.predict(te.text)
    base = langid_baseline(te.text.tolist())
    labels_b = LATIN_CLASSES + ["other"]
    results["sentence_model"] = {
        "selected_C": best[1],
        "validation_macro_f1": round(best[0], 4),
        "test_overall": evaluate(te.lang, pred, LATIN_CLASSES),
        "test_by_source": {src: evaluate(g.lang, pred[g.index.map(te.index.get_loc)], LATIN_CLASSES)
                           for src, g in te.groupby("source")},
    }
    results["langid_baseline"] = {
        "description": "langid.py 1.1.6 off-the-shelf (Naive Bayes, 97 languages), no retraining",
        "test_overall": evaluate(te.lang, base, labels_b),
        "test_by_source": {src: evaluate(g.lang, [base[te.index.get_loc(i)] for i in g.index], labels_b)
                           for src, g in te.groupby("source")},
    }

    # ---------------- sentence-length robustness (short, chat-like inputs)
    lens = te.text.str.split().str.len()
    results["sentence_model"]["accuracy_by_length"] = {
        b: round(float(accuracy_score(te.lang[m], pred[m.values])), 4)
        for b, m in (("1-3 words", lens <= 3), ("4-7 words", (lens > 3) & (lens <= 7)),
                     ("8-15 words", (lens > 7) & (lens <= 15)), ("16+ words", lens > 15)) if m.sum() > 0
    }
    results["sentence_model"]["n_by_length"] = {
        b: int(m.sum()) for b, m in (("1-3 words", lens <= 3), ("4-7 words", (lens > 3) & (lens <= 7)),
                                     ("8-15 words", (lens > 7) & (lens <= 15)), ("16+ words", lens > 15))}

    # ---------------- qualitative probes (unlabelled; reported verbatim, not scored)
    lid = LIDSystem(word_model, sent_model)
    probes = ["Yeno hengidiyaa", "Neenga epdi irukeenga", "Sukham ano", "How are you?",
              "Naan office ge hogbeku today.", "Bro, kal college jaana hai.", "Today college ge hogalla.",
              "ಇವತ್ತು class ತುಂಬಾ boring ide."]
    results["qualitative_probes"] = [
        {"text": p, "primary": (r := lid.identify(p))["primary_language"], "confidence": r["confidence"],
         "tokens": [(t["token"], t["lang"], t["confidence"]) for t in r["tokens"]]}
        for p in probes]
    results["runtime_seconds"] = round(time.time() - t0, 1)
    (RESULTS_DIR / "lid_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: results[k] for k in ("word_model", "sentence_model")}, indent=1)[:3000])
    print(json.dumps(results["langid_baseline"]["test_by_source"], indent=1)[:1500])
    for p in results["qualitative_probes"]:
        print(p)


if __name__ == "__main__":
    main()
