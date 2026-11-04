"""Script-independent language identification.

Two classical models, both character n-gram TF-IDF + multinomial Logistic
Regression:

* WordLID     - classifies a single Latin-script token as en / kn_rom / ta_rom /
                ml_rom / hi_rom. Trained on Dakshina attested romanisations
                (Roark et al., 2020) and English vocabulary from Tatoeba.
* SentenceLID - classifies a Latin-script sentence into the same classes.
                Trained on Dakshina romanised sentences, DravidianCodeMix
                Latin-script comments (train split only) and Tatoeba English.

Native-script tokens never go through a model: their script decides the
language (Kannada script -> kn, etc.).
"""
from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .paths import MODELS_DIR, SEED
from .script import NATIVE_SCRIPT_LANG, script_profile, token_script, tokenize

LATIN_CLASSES = ["en", "kn_rom", "ta_rom", "ml_rom", "hi_rom"]
LANG_NAMES = {
    "en": "English",
    "kn_rom": "Romanized Kannada",
    "ta_rom": "Romanized Tamil",
    "ml_rom": "Romanized Malayalam",
    "hi_rom": "Romanized Hindi",
    "kn": "Kannada (native script)",
    "ta": "Tamil (native script)",
    "ml": "Malayalam (native script)",
    "hi": "Hindi (Devanagari)",
    "univ": "Language-independent (number, emoji, punctuation, URL)",
    "other": "Unsupported script",
}
INDIC_OF = {"kn_rom": "kn", "ta_rom": "ta", "ml_rom": "ml", "hi_rom": "hi",
            "kn": "kn", "ta": "ta", "ml": "ml", "hi": "hi"}

UNCERTAIN_THRESHOLD = 0.6
SHORT_TEXT_MAX_WORDS = 3


def make_word_pipeline(C: float = 3.0) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 5), lowercase=True,
                                  sublinear_tf=True, min_df=2)),
        ("clf", LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=SEED)),
    ])


def make_sentence_pipeline(C: float = 10.0) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), lowercase=True,
                                  sublinear_tf=True, min_df=3, max_features=300_000)),
        ("clf", LogisticRegression(C=C, max_iter=2000, class_weight="balanced", random_state=SEED)),
    ])


@dataclass
class TokenTag:
    token: str
    script: str
    lang: str
    confidence: float | None
    uncertain: bool
    probabilities: dict[str, float] | None


class LIDSystem:
    def __init__(self, word_model: Pipeline, sentence_model: Pipeline, short_text_alpha: float = 0.0):
        self.word_model = word_model
        self.sentence_model = sentence_model
        self.word_classes = list(word_model.classes_)
        self.sent_classes = list(sentence_model.classes_)
        # Weight of the word model in short-input fusion (experiments/short_text_lid.py).
        # 0 = sentence model only, which is what all reported LID/sentiment experiments use.
        self.short_text_alpha = short_text_alpha

    # ------------------------------------------------------------------ io
    @classmethod
    def load(cls, directory=MODELS_DIR / "lid", short_text_alpha: float = 0.0) -> "LIDSystem":
        return cls(joblib.load(directory / "word_lid.joblib"), joblib.load(directory / "sentence_lid.joblib"),
                   short_text_alpha)

    # --------------------------------------------------------- sentence
    def is_short_fused(self, text: str) -> bool:
        latin = [t for t in tokenize(text) if token_script(t) == "Latin"]
        return bool(self.short_text_alpha) and 0 < len(latin) <= SHORT_TEXT_MAX_WORDS

    def sentence_probs(self, text: str) -> dict[str, float] | None:
        latin_toks = [t for t in tokenize(text) if token_script(t) == "Latin"]
        if not latin_toks:
            return None
        logp = np.log(np.clip(self.sentence_model.predict_proba([" ".join(latin_toks)])[0], 1e-12, 1))
        if self.short_text_alpha and len(latin_toks) <= SHORT_TEXT_MAX_WORDS:
            words = [t.lower() for t in latin_toks if t.isalpha()]
            if words:
                logp = logp + self.short_text_alpha * np.log(np.clip(self.word_model.predict_proba(words), 1e-12, 1)).mean(axis=0)
        p = np.exp(logp - logp.max())
        p /= p.sum()
        return {c: float(v) for c, v in zip(self.sent_classes, p)}

    # ------------------------------------------------------------- words
    def tag(self, text: str) -> list[TokenTag]:
        tokens = tokenize(text)
        return self.tag_token_list(tokens, context_text=text)

    def tag_token_list(self, tokens: list[str], context_text: str | None = None,
                       indic: str | None = None, word_probs: dict | None = None) -> list[TokenTag]:
        scripts = [token_script(t) for t in tokens]
        latin_idx = [i for i, s in enumerate(scripts) if s == "Latin"]
        out: list[TokenTag | None] = [None] * len(tokens)

        # Sentence-constrained decoding: code-mixed text is usually English plus
        # ONE Indic language, so each Latin token chooses between English and
        # the Indic language implied by the sentence (native script present,
        # else the sentence-level LID).
        if indic is None:
            indic = self.context_indic(context_text if context_text is not None else " ".join(tokens))

        if latin_idx:
            if word_probs is None:
                rows = self.word_model.predict_proba([tokens[i] for i in latin_idx])
            else:
                rows = [word_probs[tokens[i].lower()] for i in latin_idx]
            for row, i in zip(rows, latin_idx):
                full = {c: float(v) for c, v in zip(self.word_classes, row)}
                if indic in full:
                    pe, pi = full["en"], full[indic]
                    z = pe + pi
                    lang, conf = ("en", pe / z) if pe >= pi else (indic, pi / z)
                else:
                    lang = max(full, key=full.get)
                    conf = full[lang]
                out[i] = TokenTag(tokens[i], "Latin", lang, round(conf, 4), conf < UNCERTAIN_THRESHOLD,
                                  {k: round(v, 4) for k, v in full.items()})
        for i, (tok, s) in enumerate(zip(tokens, scripts)):
            if out[i] is not None:
                continue
            if s in NATIVE_SCRIPT_LANG:
                out[i] = TokenTag(tok, s, NATIVE_SCRIPT_LANG[s], 1.0, False, None)
            elif s in ("Digit", "Emoji", "Punct", "URL"):
                out[i] = TokenTag(tok, s, "univ", None, False, None)
            else:  # Mixed-script token or unsupported script
                out[i] = TokenTag(tok, s, "other", None, True, None)
        return out  # type: ignore[return-value]

    def context_indic(self, text: str) -> str | None:
        prof = script_profile(text)
        native = [s for s in prof["distribution"] if s in NATIVE_SCRIPT_LANG]
        if native:
            return NATIVE_SCRIPT_LANG[native[0]] + "_rom"
        sp = self.sentence_probs(text)
        return max((c for c in sp if c != "en"), key=sp.get) if sp else None

    def tag_tokens(self, tokens: list[str]) -> list[str]:
        """Tags only; used by language-aware normalisation."""
        return [t.lang for t in self.tag_token_list(tokens)]

    def tag_batch(self, token_lists: list[list[str]]) -> list[list[TokenTag]]:
        """Vectorised tagging for many texts (used in experiments)."""
        contexts = [" ".join(toks) for toks in token_lists]
        latin_texts = [" ".join(t for t in toks if token_script(t) == "Latin") for toks in token_lists]
        has_latin = [i for i, t in enumerate(latin_texts) if t.strip()]
        sent_p = {}
        if has_latin:
            P = self.sentence_model.predict_proba([latin_texts[i] for i in has_latin])
            for i, row in zip(has_latin, P):
                sent_p[i] = {c: float(v) for c, v in zip(self.sent_classes, row)}
        vocab = sorted({t.lower() for toks in token_lists for t in toks if token_script(t) == "Latin"})
        word_probs = dict(zip(vocab, self.word_model.predict_proba(vocab))) if vocab else {}
        out = []
        for i, toks in enumerate(token_lists):
            prof = script_profile(contexts[i])
            native = [s for s in prof["distribution"] if s in NATIVE_SCRIPT_LANG]
            if native:
                indic = NATIVE_SCRIPT_LANG[native[0]] + "_rom"
            elif i in sent_p:
                sp = sent_p[i]
                indic = max((c for c in sp if c != "en"), key=sp.get)
            else:
                indic = None
            out.append(self.tag_token_list(toks, indic=indic, word_probs=word_probs))
        return out

    # ---------------------------------------------------------- analysis
    def identify(self, text: str) -> dict:
        prof = script_profile(text)
        tags = self.tag(text)
        sp = self.sentence_probs(text)
        word_langs = [t.lang for t in tags if t.lang not in ("univ",)]
        counts: dict[str, int] = {}
        for l in word_langs:
            counts[l] = counts.get(l, 0) + 1

        unsupported = any(t.lang == "other" for t in tags) and not word_langs.count("en") + sum(
            counts.get(k, 0) for k in INDIC_OF) > 0
        if prof["text_type"] == "native":
            primary = NATIVE_SCRIPT_LANG[prof["dominant_script"]]
            primary_conf, source = 1.0, "unicode-script"
        elif sp is not None and prof["text_type"] == "romanized":
            primary = max(sp, key=sp.get)
            primary_conf = sp[primary]
            source = "sentence-model + word-model fusion (short input)" if self.is_short_fused(text) else "sentence-model"
        elif prof["text_type"] == "mixed_script":
            native = [s for s in prof["distribution"] if s in NATIVE_SCRIPT_LANG]
            primary = NATIVE_SCRIPT_LANG[native[0]]
            primary_conf, source = None, "unicode-script + word-model"
        else:
            primary, primary_conf, source = "other", None, "none"

        return {
            "script_profile": prof,
            "primary_language": primary,
            "primary_language_name": LANG_NAMES.get(primary, primary),
            "confidence": None if primary_conf is None else round(primary_conf, 4),
            "decision_source": source,
            "sentence_probabilities": None if sp is None else {k: round(v, 4) for k, v in sp.items()},
            "uncertain": unsupported or (primary_conf is not None and primary_conf < UNCERTAIN_THRESHOLD),
            "unsupported": unsupported,
            "tokens": [t.__dict__ for t in tags],
            "word_language_counts": counts,
        }


def predict_matrix(model: Pipeline, items: list[str]) -> np.ndarray:
    return model.predict(items)
