"""Short-input LID experiment: does fusing the word-level model into the
sentence-level decision help for 1-3-word inputs?

Fused score (only for texts with <= MAX_WORDS Latin words):
    log p(c) = log p_sentence(c) + alpha * mean_t log p_word(c | t)

Protocol (no test data is used for any choice):
  * The sentence/test data and the 90/10 train/validation split are rebuilt
    exactly as in train_lid.py (same seed, same RNG call order).
  * A sentence model is fitted on the 90% part with train_lid's selected C;
    alpha is chosen on the short inputs of the 10% validation part.
  * The FINAL sentence model (models/lid) with the chosen alpha is then
    evaluated once on the short inputs of the held-out test set, with a paired
    bootstrap CI against alpha = 0.
Run after train_lid.py:  .venv\\Scripts\\python experiments\\short_text_lid.py
Output: experiments/results/short_text_lid.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score  # noqa: E402

import train_lid as T  # noqa: E402  (also puts backend/ on sys.path)
from nlp_lab.lid import LATIN_CLASSES, LIDSystem, make_sentence_pipeline  # noqa: E402
from nlp_lab.paths import RESULTS_DIR, SEED  # noqa: E402

MAX_WORDS = 3
ALPHAS = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]


def fused_predict(sent_model, word_model, texts, alpha):
    classes = list(sent_model.classes_)
    assert classes == list(word_model.classes_)
    P = np.log(np.clip(sent_model.predict_proba(texts), 1e-12, 1))
    if alpha:
        for i, t in enumerate(texts):
            toks = [w.lower() for w in t.split() if w.isalpha()]
            if toks:
                W = np.log(np.clip(word_model.predict_proba(toks), 1e-12, 1)).mean(axis=0)
                P[i] += alpha * W
    return np.array(classes)[P.argmax(axis=1)]


def boot(y, pa, pb, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    y, pa, pb = map(np.asarray, (y, pa, pb))
    d = np.empty(n)
    for k in range(n):
        s = rng.integers(0, len(y), len(y))
        d[k] = (pb[s] == y[s]).mean() - (pa[s] == y[s]).mean()
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta_accuracy": round(float((pb == y).mean() - (pa == y).mean()), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)]}


def short(df):
    return df[df.text.str.split().str.len() <= MAX_WORDS]


def main():
    eng = T.load_tatoeba()
    sd = T.sentence_data(eng)                    # same RNG call as train_lid.main
    tr = sd[sd.split == "train"]
    te = sd[sd.split == "test"]
    perm = T.rng.permutation(len(tr))            # same as train_lid.main
    cut = int(0.9 * len(tr))
    tr_a, tr_b = tr.iloc[perm[:cut]], tr.iloc[perm[cut:]]
    final = LIDSystem.load()
    C = json.loads((RESULTS_DIR / "lid_results.json").read_text(encoding="utf-8"))["sentence_model"]["selected_C"]

    val_model = make_sentence_pipeline(C).fit(tr_a.text, tr_a.lang)
    vb = short(tr_b)
    val = {}
    for a in ALPHAS:
        p = fused_predict(val_model, final.word_model, vb.text.tolist(), a)
        val[a] = {"accuracy": round(float(accuracy_score(vb.lang, p)), 4),
                  "macro_f1": round(float(f1_score(vb.lang, p, average="macro")), 4)}
        print("val alpha", a, val[a], flush=True)
    best = max(ALPHAS, key=lambda a: (val[a]["macro_f1"], -a))

    ts = short(te)
    base = fused_predict(final.sentence_model, final.word_model, ts.text.tolist(), 0.0)
    fused = fused_predict(final.sentence_model, final.word_model, ts.text.tolist(), best)
    out = {
        "max_words": MAX_WORDS, "alphas": ALPHAS, "selected_alpha": best, "validation_short_n": int(len(vb)),
        "validation": {str(a): v for a, v in val.items()},
        "test_short_n": int(len(ts)),
        "test_short_by_source_n": ts.source.value_counts().to_dict(),
        "test_base": {"accuracy": round(float(accuracy_score(ts.lang, base)), 4),
                      "macro_f1": round(float(f1_score(ts.lang, base, average="macro")), 4),
                      "confusion_matrix": confusion_matrix(ts.lang, base, labels=LATIN_CLASSES).tolist()},
        "test_fused": {"accuracy": round(float(accuracy_score(ts.lang, fused)), 4),
                       "macro_f1": round(float(f1_score(ts.lang, fused, average="macro")), 4),
                       "confusion_matrix": confusion_matrix(ts.lang, fused, labels=LATIN_CLASSES).tolist()},
        "labels": LATIN_CLASSES,
        "fused_vs_base": boot(ts.lang.values, base, fused),
        "by_source": {src: {"n": int(len(g)),
                            "base_accuracy": round(float((base[ts.index.get_indexer(g.index)] == g.lang.values).mean()), 4),
                            "fused_accuracy": round(float((fused[ts.index.get_indexer(g.index)] == g.lang.values).mean()), 4)}
                      for src, g in ts.groupby("source")},
        "note": "alpha selected on validation short inputs only; test evaluated once with the final models.",
    }
    (RESULTS_DIR / "short_text_lid.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("selected_alpha", "test_short_n", "test_base", "test_fused", "fused_vs_base", "by_source")}, indent=1))


if __name__ == "__main__":
    main()
