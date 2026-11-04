"""NLP Lab inference service. Classical models only; no external AI calls."""
from __future__ import annotations

import json
import threading

import joblib

from . import codemix
from .lid import LANG_NAMES, LIDSystem
from .normalize import FOLD_RULES_DOC
from .paths import MODELS_DIR, RESULTS_DIR
from .sentiment import CONFIG_DESCRIPTIONS, MODEL_NAMES, views_from_tags

SENTIMENT_LANGS = {"kn": "Kannada", "ta": "Tamil", "ml": "Malayalam"}
KINDS = ("logreg", "linear_svm")


class NotReady(Exception):
    pass


class NLPService:
    def __init__(self):
        self.lid: LIDSystem | None = None
        self.models: dict[tuple[str, str], object] = {}
        self.errors: list[str] = []
        self._dataset = None
        self._lock = threading.Lock()
        try:
            # The app uses the short-input fusion weight chosen on validation data
            # (experiments/short_text_lid.py); without that file it falls back to 0.
            alpha_file = RESULTS_DIR / "short_text_lid.json"
            alpha = json.loads(alpha_file.read_text(encoding="utf-8"))["selected_alpha"] if alpha_file.exists() else 0.0
            self.lid = LIDSystem.load(short_text_alpha=alpha)
        except FileNotFoundError:
            self.errors.append("LID models missing: run experiments/train_lid.py")
        for lang in SENTIMENT_LANGS:
            for kind in KINDS:
                p = MODELS_DIR / "sentiment" / f"{lang}_{kind}.joblib"
                if p.exists():
                    self.models[(lang, kind)] = joblib.load(p)
        if not self.models:
            self.errors.append("Sentiment models missing: run experiments/run_sentiment.py")

    # ---------------------------------------------------------------- status
    def status(self) -> dict:
        return {"lid_ready": self.lid is not None, "sentiment_models": [f"{l}_{k}" for l, k in self.models],
                "short_text_fusion_alpha": self.lid.short_text_alpha if self.lid else None, "errors": self.errors}

    def _need_lid(self) -> LIDSystem:
        if self.lid is None:
            raise NotReady("Language-identification models are not trained yet.")
        return self.lid

    # ------------------------------------------------------------------- LID
    def language_identification(self, text: str) -> dict:
        return self._need_lid().identify(text)

    def code_mixing(self, text: str) -> dict:
        lid = self._need_lid()
        tags = lid.tag(text)
        langs = [t.lang for t in tags]
        stats = codemix.stats(langs)
        views = views_from_tags(text, tags)
        return {
            "tokens": [t.__dict__ for t in tags],
            "stats": stats,
            "script_mixing": codemix.scripts_mixing([(t.lang, t.script) for t in tags]),
            "normalized": {"basic": views["basic"], "language_aware": views["norm"]},
            "normalization_rules": FOLD_RULES_DOC,
            "uncertain_tokens": sum(t.uncertain for t in tags if t.lang != "univ"),
            "language_names": LANG_NAMES,
        }

    # ------------------------------------------------------------- sentiment
    def route(self, text: str, language: str) -> dict:
        lid = self._need_lid()
        if language != "auto":
            return {"language": language, "source": "user-selected", "lid": None}
        ident = lid.identify(text)
        base = ident["primary_language"].replace("_rom", "")
        if base in SENTIMENT_LANGS:
            return {"language": base, "source": "lid", "lid": {k: ident[k] for k in
                    ("primary_language", "primary_language_name", "confidence", "uncertain")}}
        return {"language": None, "source": "lid", "lid": {k: ident[k] for k in
                ("primary_language", "primary_language_name", "confidence", "uncertain")}}

    def sentiment(self, text: str, model: str = "logreg", language: str = "auto") -> dict:
        r = self.route(text, language)
        if r["language"] is None:
            return {"supported": False, "routing": r,
                    "message": f"Detected {r['lid']['primary_language_name']}. Sentiment models were trained only "
                               "on Kannada, Tamil and Malayalam (DravidianCodeMix). Choose a model language "
                               "manually to run an out-of-distribution prediction."}
        m = self.models.get((r["language"], model))
        if m is None:
            raise NotReady(f"Model {r['language']}_{model} is not trained.")
        out = m.predict(text, self.lid)
        out.update({"supported": True, "routing": r, "language_name": SENTIMENT_LANGS[r["language"]],
                    "feature_config_description": CONFIG_DESCRIPTIONS[out["feature_config"]],
                    "test_macro_f1": m.meta["test_metrics"]["macro_f1"]})
        return out

    def compare(self, text: str, language: str = "auto") -> dict:
        r = self.route(text, language)
        if r["language"] is None:
            return {"supported": False, "routing": r, "results": []}
        res = []
        for kind in KINDS:
            m = self.models.get((r["language"], kind))
            if m is not None:
                p = m.predict(text, self.lid)
                p["test_macro_f1"] = m.meta["test_metrics"]["macro_f1"]
                p["test_accuracy"] = m.meta["test_metrics"]["accuracy"]
                res.append(p)
        return {"supported": True, "routing": r, "language_name": SENTIMENT_LANGS[r["language"]], "results": res,
                "agree": len({x["label"] for x in res}) == 1}

    def analyze(self, text: str, model: str = "logreg", language: str = "auto") -> dict:
        return {"language_identification": self.language_identification(text), "code_mixing": self.code_mixing(text),
                "sentiment": self.sentiment(text, model, language)}

    # -------------------------------------------------------- models / results
    def models_info(self) -> list[dict]:
        out = []
        for (lang, kind), m in sorted(self.models.items()):
            out.append({"id": f"{lang}_{kind}", "language": lang, "language_name": SENTIMENT_LANGS[lang],
                        "model": MODEL_NAMES[kind], "model_kind": kind, "feature_config": m.config,
                        "feature_config_description": CONFIG_DESCRIPTIONS[m.config], "C": m.meta["C"],
                        "train_size": m.meta["train_size"], "dev_macro_f1": m.meta["dev_macro_f1"],
                        "test_metrics": {k: m.meta["test_metrics"][k] for k in
                                         ("accuracy", "macro_precision", "macro_recall", "macro_f1", "per_class",
                                          "confusion_matrix", "n")},
                        "n_features": len(m._names)})
        return out

    def global_features(self, model_id: str, n: int = 20) -> dict:
        lang, _, kind = model_id.partition("_")
        m = self.models.get((lang, kind))
        if m is None:
            raise KeyError(model_id)
        return {"id": model_id, "classes": m.classes, "features": m.global_top_features(n)}

    def results(self) -> dict:
        out = {}
        for name in ("lid_results", "sentiment_results", "dataset_audit"):
            p = RESULTS_DIR / f"{name}.json"
            out[name] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        return out

    # ---------------------------------------------------------------- dataset
    def dataset(self):
        with self._lock:
            if self._dataset is None:
                from .data import clean_for_sentiment, load_all
                df, _ = clean_for_sentiment(load_all())
                self._dataset = df[["text", "raw_label", "label", "lang", "split", "text_type"]].reset_index(drop=True)
            return self._dataset

    def samples(self, lang=None, label=None, text_type=None, split=None, q=None, offset=0, limit=25) -> dict:
        df = self.dataset()
        for col, val in (("lang", lang), ("label", label), ("text_type", text_type), ("split", split)):
            if val:
                df = df[df[col] == val]
        if q:
            df = df[df.text.str.contains(q, case=False, regex=False)]
        page = df.iloc[offset:offset + limit]
        return {"total": int(len(df)), "offset": offset, "limit": limit, "rows": page.to_dict(orient="records")}
