"""Sentiment experiments on DravidianCodeMix (3-class).

Run from the project root (after train_lid.py):
    .venv\\Scripts\\python experiments\\run_sentiment.py

Protocol
  * Official train/dev/test splits after de-duplication and leakage removal
    (nlp_lab.data.clean_for_sentiment).
  * Vectorizers are fitted on train only.
  * C is selected on dev by macro-F1; the selected model is evaluated once on test.
  * Fixed seed (nlp_lab.paths.SEED).
Outputs
  experiments/results/sentiment_results.json
  experiments/results/dataset_audit.json
  models/sentiment/<lang>_<kind>.joblib (deployed: best feature config per language)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,  # noqa: E402
                             precision_recall_fscore_support)

from nlp_lab.data import LABEL_MAP, audit, clean_for_sentiment, load_all  # noqa: E402
from nlp_lab.lid import LIDSystem  # noqa: E402
from nlp_lab.paths import MODELS_DIR, RESULTS_DIR, SEED  # noqa: E402
from nlp_lab.sentiment import (CLASSES, CONFIG_DESCRIPTIONS, CONFIGS, SentimentModel,  # noqa: E402
                               make_classifier, make_features, prepare)

C_GRID = {"logreg": [0.3, 1.0, 3.0, 10.0], "linear_svm": [0.03, 0.1, 0.3, 1.0]}
LANGS = ["kn", "ta", "ml"]
N_BOOT = 1000
rng = np.random.default_rng(SEED)


def metrics(y, p) -> dict:
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=CLASSES, zero_division=0)
    return {
        "n": int(len(y)),
        "accuracy": round(float(accuracy_score(y, p)), 4),
        "macro_precision": round(float(pr.mean()), 4),
        "macro_recall": round(float(rc.mean()), 4),
        "macro_f1": round(float(f1.mean()), 4),
        "weighted_f1": round(float(f1_score(y, p, labels=CLASSES, average="weighted", zero_division=0)), 4),
        "per_class": {c: {"precision": round(float(a), 4), "recall": round(float(b), 4),
                          "f1": round(float(d), 4), "support": int(s)}
                      for c, a, b, d, s in zip(CLASSES, pr, rc, f1, sup)},
        "confusion_matrix": confusion_matrix(y, p, labels=CLASSES).tolist(),
    }


def paired_bootstrap(y, pa, pb, n=N_BOOT) -> dict:
    """Macro-F1 difference (b - a) with a 95% percentile CI."""
    y, pa, pb = map(np.asarray, (y, pa, pb))
    m = len(y)
    diffs = np.empty(n)
    for i in range(n):
        s = rng.integers(0, m, m)
        diffs[i] = (f1_score(y[s], pb[s], labels=CLASSES, average="macro", zero_division=0)
                    - f1_score(y[s], pa[s], labels=CLASSES, average="macro", zero_division=0))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    point = (f1_score(y, pb, labels=CLASSES, average="macro", zero_division=0)
             - f1_score(y, pa, labels=CLASSES, average="macro", zero_division=0))
    return {"delta_macro_f1": round(float(point), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "p_le_0": round(float((diffs <= 0).mean()), 4), "resamples": n}


def primary_languages(texts, lid) -> list[str]:
    """Batch equivalent of LIDSystem.identify(...)['primary_language']."""
    from nlp_lab.script import NATIVE_SCRIPT_LANG, script_profile, token_script, tokenize
    out, rom_idx, rom_txt = [], [], []
    for i, t in enumerate(texts):
        prof = script_profile(t)
        native = [s for s in prof["distribution"] if s in NATIVE_SCRIPT_LANG]
        if prof["text_type"] in ("native", "mixed_script"):
            out.append(NATIVE_SCRIPT_LANG[native[0]])
        elif prof["text_type"] == "romanized":
            out.append(None)
            rom_idx.append(i)
            rom_txt.append(" ".join(x for x in tokenize(t) if token_script(x) == "Latin"))
        else:
            out.append("other")
    for i, p in zip(rom_idx, lid.sentence_model.predict(rom_txt)):
        out[i] = p
    return out


def fit_select(Xtr, ytr, Xdv, ydv, kind):
    best = None
    grid = []
    for C in C_GRID[kind]:
        clf = make_classifier(kind, C).fit(Xtr, ytr)
        f1 = f1_score(ydv, clf.predict(Xdv), average="macro")
        grid.append({"C": C, "dev_macro_f1": round(float(f1), 4)})
        if best is None or f1 > best[0]:
            best = (f1, C, clf)
    return best, grid


def main():
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "sentiment").mkdir(parents=True, exist_ok=True)
    lid = LIDSystem.load()

    raw = load_all()
    df, clean_report = clean_for_sentiment(raw)
    print("preparing views ...", flush=True)
    views = prepare(df.text.tolist(), lid)
    df = pd.concat([df.reset_index(drop=True), views], axis=1)
    df["is_code_mixed"] = df.langfeat.map(lambda f: f["is_code_mixed"] > 0)
    df["lid_primary"] = primary_languages(df.text.tolist(), lid)

    audit_out = {
        "source": "DravidianCodeMix-2020 (Chakravarthi et al., 2022), CC BY 4.0",
        "label_mapping": {k: v for k, v in LABEL_MAP.items()},
        "excluded": ["Mixed_feelings", "not-<language>"],
        "raw_audit": audit(raw),
        "cleaning": clean_report,
        "clean_label_distribution": {f"{l}/{s}": g.label.value_counts().to_dict()
                                     for (l, s), g in df.groupby(["lang", "split"])},
        "clean_text_type_distribution": {l: g.text_type.value_counts().to_dict() for l, g in df.groupby("lang")},
        "clean_code_mixed_share": {l: round(float(g.is_code_mixed.mean()), 4) for l, g in df.groupby("lang")},
        "code_mixing_mean_cmi": {l: round(float(g.langfeat.map(lambda f: f["cmi"]).mean() * 100), 2)
                                 for l, g in df.groupby("lang")},
        "note": "text_type is from Unicode script; is_code_mixed and CMI are derived from the word-level "
                "LID model and therefore carry its error rate.",
    }
    (RESULTS_DIR / "dataset_audit.json").write_text(json.dumps(audit_out, indent=2, ensure_ascii=False), encoding="utf-8")

    results = {"seed": SEED, "c_grid": C_GRID, "configs": CONFIG_DESCRIPTIONS, "per_language": {},
               "significance": {}, "routing": {}}
    test_preds: dict = {}
    deployed = {}

    for lang in LANGS:
        d = df[df.lang == lang]
        tr, dv, te = (d[d.split == s] for s in ("train", "dev", "test"))
        maj = tr.label.value_counts().idxmax()
        results["per_language"][lang] = {"majority_baseline": metrics(te.label, [maj] * len(te)), "runs": {}}
        for config in CONFIGS:
            ft = make_features(config)
            Xtr = ft.fit_transform(tr)
            Xdv, Xte = ft.transform(dv), ft.transform(te)
            for kind in ("logreg", "linear_svm"):
                t1 = time.time()
                (f1dv, C, clf), grid = fit_select(Xtr, tr.label, Xdv, dv.label, kind)
                pred = clf.predict(Xte)
                test_preds[(lang, config, kind)] = pred
                m = metrics(te.label, pred)
                m["by_text_type"] = {tt: metrics(te.label[mask], pred[mask.values])
                                     for tt in ("romanized", "native", "mixed_script")
                                     if (mask := te.text_type == tt).sum() >= 20}
                m["by_code_mixing"] = {name: metrics(te.label[mask], pred[mask.values])
                                       for name, mask in (("code_mixed", te.is_code_mixed),
                                                          ("monolingual", ~te.is_code_mixed)) if mask.sum() >= 20}
                results["per_language"][lang]["runs"][f"{config}|{kind}"] = {
                    "config": config, "model": kind, "selected_C": C, "dev_macro_f1": round(float(f1dv), 4),
                    "c_search": grid, "n_features": int(Xtr.shape[1]), "test": m,
                    "fit_seconds": round(time.time() - t1, 1)}
                print(f"{lang} {config:22s} {kind:10s} C={C:<5} dev={f1dv:.4f} test={m['macro_f1']:.4f} "
                      f"acc={m['accuracy']:.4f}", flush=True)
                key = (lang, kind)
                if key not in deployed or f1dv > deployed[key][0]:
                    deployed[key] = (f1dv, config, C)

        y = te.label.values
        sig = {}
        for kind in ("logreg", "linear_svm"):
            P = {c: test_preds[(lang, c, kind)] for c in CONFIGS}
            sig[kind] = {
                "char vs word": paired_bootstrap(y, P["word"], P["char"]),
                "word+char vs word": paired_bootstrap(y, P["word"], P["word+char"]),
                "word+char+norm vs word+char": paired_bootstrap(y, P["word+char"], P["word+char+norm"]),
                "word+char+lang vs word+char": paired_bootstrap(y, P["word+char"], P["word+char+lang"]),
                "word+char+norm+lang vs word+char": paired_bootstrap(y, P["word+char"], P["word+char+norm+lang"]),
            }
        sig["linear_svm vs logreg (word+char)"] = paired_bootstrap(
            y, test_preds[(lang, "word+char", "logreg")], test_preds[(lang, "word+char", "linear_svm")])
        results["significance"][lang] = sig

    # ---------------- deploy: retrain the dev-selected config per (lang, kind) on train
    print("training deployed models ...", flush=True)
    models = {}
    for (lang, kind), (f1dv, config, C) in deployed.items():
        d = df[df.lang == lang]
        tr = d[d.split == "train"]
        ft = make_features(config)
        X = ft.fit_transform(tr)
        clf = make_classifier(kind, C).fit(X, tr.label)
        meta = {"dev_macro_f1": round(float(f1dv), 4), "C": C, "train_size": int(len(tr)),
                "test_metrics": results["per_language"][lang]["runs"][f"{config}|{kind}"]["test"]}
        sm = SentimentModel(ft, clf, config, kind, lang, meta)
        joblib.dump(sm, MODELS_DIR / "sentiment" / f"{lang}_{kind}.joblib", compress=3)
        models[(lang, kind)] = sm
    results["deployed"] = {f"{l}_{k}": {"config": c, "C": C, "dev_macro_f1": round(float(f), 4)}
                           for (l, k), (f, c, C) in deployed.items()}

    # ---------------- routing experiment (pooled test set of all three languages)
    # Compares: (a) one pooled multilingual model, (b) oracle routing to the
    # language-specific model using the gold dataset language, (c) routing by
    # our LID's primary-language prediction.
    print("routing experiment ...", flush=True)
    tr_all = df[df.split == "train"]
    dv_all = df[df.split == "dev"]
    te_all = df[df.split == "test"].reset_index(drop=True)
    for kind in ("logreg", "linear_svm"):
        cfg = "word+char"
        ft = make_features(cfg)
        Xtr = ft.fit_transform(tr_all)
        (f1dv, C, clf), _ = fit_select(Xtr, tr_all.label, ft.transform(dv_all), dv_all.label, kind)
        pooled = clf.predict(ft.transform(te_all))
        oracle = np.empty(len(te_all), dtype=object)
        routed = np.empty(len(te_all), dtype=object)
        # language-specific word+char models (same config for a fair comparison)
        spec = {}
        for lang in LANGS:
            d = df[(df.lang == lang) & (df.split == "train")]
            dd = df[(df.lang == lang) & (df.split == "dev")]
            f2 = make_features(cfg)
            X2 = f2.fit_transform(d)
            (_, C2, c2), _ = fit_select(X2, d.label, f2.transform(dd), dd.label, kind)
            spec[lang] = (f2, c2)
        lid_lang = te_all.lid_primary.map(lambda p: p.replace("_rom", "") if p.replace("_rom", "") in LANGS else None)
        for lang in LANGS:
            f2, c2 = spec[lang]
            mask = (te_all.lang == lang).values
            oracle[mask] = c2.predict(f2.transform(te_all[mask]))
            rmask = (lid_lang == lang).values
            if rmask.any():
                routed[rmask] = c2.predict(f2.transform(te_all[rmask]))
        fallback = pd.isna(lid_lang).values
        routed[fallback] = pooled[fallback]
        y = te_all.label.values
        results["routing"][kind] = {
            "config": cfg,
            "pooled": metrics(y, pooled),
            "oracle_language_routing": metrics(y, oracle),
            "lid_routing": metrics(y, routed),
            "lid_routing_language_accuracy": round(float((lid_lang == te_all.lang).mean()), 4),
            "lid_routing_fallback_to_pooled": int(fallback.sum()),
            "lid_routing vs pooled": paired_bootstrap(y, pooled, routed),
            "oracle vs lid_routing": paired_bootstrap(y, routed, oracle),
        }
        print(kind, "pooled", results["routing"][kind]["pooled"]["macro_f1"],
              "oracle", results["routing"][kind]["oracle_language_routing"]["macro_f1"],
              "lid", results["routing"][kind]["lid_routing"]["macro_f1"], flush=True)

    results["runtime_seconds"] = round(time.time() - t0, 1)
    (RESULTS_DIR / "sentiment_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("done in", results["runtime_seconds"], "s")


if __name__ == "__main__":
    main()
