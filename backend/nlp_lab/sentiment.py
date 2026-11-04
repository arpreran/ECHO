"""Classical sentiment models (TF-IDF + Logistic Regression / Linear SVM) and
exact linear explanations.

Every input text is first turned into three views (see `prepare`):
  basic    - script-agnostic normalised tokens, space-joined
  norm     - language-aware normalised tokens (romanized spelling folding)
  langfeat - dict of code-mixing / script features from word-level LID

A feature configuration picks which views feed which vectorizer.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from . import codemix
from .normalize import ROMANIZED_INDIC, basic_normalize_token, romanized_fold, strip_joiners
from .paths import SEED
from .script import NATIVE_SCRIPT_LANG, script_profile, tokenize

CLASSES = ["negative", "neutral", "positive"]

CONFIGS = {
    "word":                 {"word": "basic", "char": None,    "lang": False},
    "char":                 {"word": None,    "char": "basic", "lang": False},
    "word+char":            {"word": "basic", "char": "basic", "lang": False},
    "word+char+norm":       {"word": "norm",  "char": "norm",  "lang": False},
    "word+char+lang":       {"word": "basic", "char": "basic", "lang": True},
    "word+char+norm+lang":  {"word": "norm",  "char": "norm",  "lang": True},
}
CONFIG_DESCRIPTIONS = {
    "word": "Word uni+bigram TF-IDF on basic-normalised tokens",
    "char": "Character 2-5-gram TF-IDF (word-bounded) on basic-normalised text",
    "word+char": "Concatenation of word and character TF-IDF",
    "word+char+norm": "Word+char TF-IDF on language-aware normalised text (romanized spelling folding)",
    "word+char+lang": "Word+char TF-IDF plus LID-derived code-mixing features",
    "word+char+norm+lang": "Language-aware normalisation plus LID-derived code-mixing features",
}


# ------------------------------------------------------------------ views
def views_from_tags(text: str, tags) -> dict:
    """tags: list of lid.TokenTag for tokenize(text)."""
    basic, norm = [], []
    for t in tags:
        b = basic_normalize_token(t.token)
        basic.append(b)
        if t.lang in ROMANIZED_INDIC:
            norm.append(romanized_fold(b))
        elif t.script in NATIVE_SCRIPT_LANG:
            norm.append(strip_joiners(b))
        else:
            norm.append(b)
    langs = [t.lang for t in tags]
    st = codemix.stats(langs)
    n_lang = max(st["language_tokens"], 1)
    prof = script_profile(text)
    feat = {
        "frac_en": langs.count("en") / n_lang,
        "frac_romanized_indic": sum(l in ROMANIZED_INDIC for l in langs) / n_lang,
        "frac_native_script": sum(l in ("kn", "ta", "ml", "hi") for l in langs) / n_lang,
        "cmi": st["cmi"] / 100.0,
        "i_index": st["i_index"],
        "log_switch_points": math.log1p(st["switch_points"]),
        f"text_type={prof['text_type']}": 1.0,
        "is_code_mixed": float(st["is_code_mixed"]),
    }
    return {"basic": " ".join(basic), "norm": " ".join(norm), "langfeat": feat}


def prepare(texts: list[str], lid) -> pd.DataFrame:
    token_lists = [tokenize(t) for t in texts]
    tag_lists = lid.tag_batch(token_lists)
    return pd.DataFrame([views_from_tags(t, tags) for t, tags in zip(texts, tag_lists)])


# ------------------------------------------------------------- features
def make_features(config: str) -> ColumnTransformer:
    c = CONFIGS[config]
    parts = []
    if c["word"]:
        parts.append(("word", TfidfVectorizer(token_pattern=r"\S+", lowercase=False, ngram_range=(1, 2),
                                              min_df=2, sublinear_tf=True), c["word"]))
    if c["char"]:
        parts.append(("char", TfidfVectorizer(analyzer="char_wb", lowercase=False, ngram_range=(2, 5),
                                              min_df=2, sublinear_tf=True, max_features=300_000), c["char"]))
    if c["lang"]:
        parts.append(("lang", DictVectorizer(), "langfeat"))
    return ColumnTransformer(parts)


def make_classifier(kind: str, C: float):
    if kind == "logreg":
        return LogisticRegression(C=C, max_iter=3000, class_weight="balanced", random_state=SEED)
    if kind == "linear_svm":
        return LinearSVC(C=C, class_weight="balanced", random_state=SEED, max_iter=10000)
    raise ValueError(kind)


MODEL_NAMES = {"logreg": "Logistic Regression", "linear_svm": "Linear SVM"}


# --------------------------------------------------------- deployed model
class SentimentModel:
    """A fitted (features, classifier) pair plus metadata. Pickled with joblib."""

    def __init__(self, features: ColumnTransformer, clf, config: str, kind: str, lang: str, meta: dict):
        self.features = features
        self.clf = clf
        self.config = config
        self.kind = kind
        self.lang = lang
        self.meta = meta
        self.classes = list(clf.classes_)
        self._names = features.get_feature_names_out()

    def predict(self, text: str, lid) -> dict:
        tags = lid.tag(text)
        views = views_from_tags(text, tags)
        X = self.features.transform(pd.DataFrame([views]))
        scores = self.clf.decision_function(X)[0]
        k = int(np.argmax(scores))
        out = {
            "label": self.classes[k],
            "model": MODEL_NAMES[self.kind],
            "model_kind": self.kind,
            "feature_config": self.config,
            "trained_on": self.lang,
            "decision_scores": {c: round(float(s), 4) for c, s in zip(self.classes, scores)},
        }
        if hasattr(self.clf, "predict_proba"):
            p = self.clf.predict_proba(X)[0]
            out["probabilities"] = {c: round(float(v), 4) for c, v in zip(self.classes, p)}
            out["confidence"] = round(float(p[k]), 4)
            out["confidence_type"] = "softmax probability (Logistic Regression)"
        else:
            s = sorted(scores, reverse=True)
            out["probabilities"] = None
            out["confidence"] = None
            out["margin"] = round(float(s[0] - s[1]), 4)
            out["confidence_type"] = "decision-score margin (Linear SVM; not a probability)"
        out["explanation"] = self.explain(X, views, tags, k)
        return out

    def explain(self, X, views: dict, tags, k: int, top: int = 12) -> dict:
        """Exact decomposition of the class-k linear score into token-level and
        feature-level contributions: sum(token) + sum(non-token) + bias = score."""
        coef = self.clf.coef_
        bias = float(self.clf.intercept_[k])
        x = X.tocsr()
        idx, vals = x.indices, x.data
        contribs = vals * np.asarray(coef[k, idx]).ravel()
        view_key = "norm" if CONFIGS[self.config]["word"] == "norm" or CONFIGS[self.config]["char"] == "norm" else "basic"
        toks = views[view_key].split(" ") if views[view_key] else []
        token_scores = np.zeros(len(toks))
        feature_rows, non_token = [], 0.0
        padded = [f" {t} " for t in toks]
        for j, c in zip(idx, contribs):
            name = self._names[j]
            block, _, feat = name.partition("__")
            feature_rows.append({"feature": feat, "block": block, "contribution": float(c)})
            if block == "word":
                parts = feat.split(" ")
                hits = [i for i in range(len(toks) - len(parts) + 1) if toks[i:i + len(parts)] == parts]
                share = c / (len(hits) * len(parts)) if hits else 0.0
                for i in hits:
                    for q in range(len(parts)):
                        token_scores[i + q] += share
                if not hits:
                    non_token += c
            elif block == "char":
                occ = [p.count(feat) for p in padded]
                tot = sum(occ)
                if tot:
                    for i, o in enumerate(occ):
                        token_scores[i] += c * o / tot
                else:
                    non_token += c
            else:
                non_token += c
        feature_rows.sort(key=lambda r: -abs(r["contribution"]))
        token_rows = []
        for i, t in enumerate(tags[:len(toks)]):
            token_rows.append({"token": t.token, "normalized": toks[i], "lang": t.lang,
                               "contribution": round(float(token_scores[i]), 5)})
        return {
            "target_class": self.classes[k],
            "score": round(float(contribs.sum() + bias), 5),
            "bias": round(bias, 5),
            "non_token_contribution": round(non_token, 5),
            "tokens": token_rows,
            "top_features": [{**r, "contribution": round(r["contribution"], 5)} for r in feature_rows[:top]],
            "method": "Linear decomposition: contribution = TF-IDF value x model coefficient for the "
                      "predicted class; character n-grams are shared among the tokens that contain them.",
        }

    def global_top_features(self, n: int = 20) -> dict:
        out = {}
        for k, c in enumerate(self.classes):
            w = self.clf.coef_[k]
            order = np.argsort(w)
            out[c] = {
                "positive": [{"feature": self._names[j].partition("__")[2], "block": self._names[j].partition("__")[0],
                              "weight": round(float(w[j]), 4)} for j in order[::-1][:n]],
                "negative": [{"feature": self._names[j].partition("__")[2], "block": self._names[j].partition("__")[0],
                              "weight": round(float(w[j]), 4)} for j in order[:n]],
            }
        return out
