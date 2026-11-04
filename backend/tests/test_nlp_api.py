"""NLP Lab API tests against the trained models (skipped if models are missing)."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from nlp_lab.paths import MODELS_DIR

pytestmark = pytest.mark.skipif(not (MODELS_DIR / "sentiment" / "kn_logreg.joblib").exists(),
                                reason="trained models not present")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    s = Settings(database_path=str(tmp_path_factory.mktemp("db") / "n.sqlite3"))
    with TestClient(create_app(s)) as c:
        yield c


def test_status(client):
    s = client.get("/api/nlp/status").json()
    assert s["lid_ready"] and len(s["sentiment_models"]) == 6


def test_native_script_lid_is_deterministic(client):
    r = client.post("/api/nlp/language-identification", json={"text": "ನಾನು ಚೆನ್ನಾಗಿದ್ದೀನಿ"}).json()
    assert r["primary_language"] == "kn" and r["decision_source"] == "unicode-script"


def test_english_vs_romanized(client):
    en = client.post("/api/nlp/language-identification", json={"text": "How are you doing today?"}).json()
    assert en["primary_language"] == "en"
    ta = client.post("/api/nlp/language-identification", json={"text": "Neenga epdi irukeenga"}).json()
    assert ta["primary_language"] == "ta_rom"
    probs = ta["sentence_probabilities"]
    assert abs(sum(probs.values()) - 1) < 1e-3


def test_short_input_fusion_matches_experiment():
    """The app's fused short-input LID must make the same decision as the
    function evaluated in experiments/short_text_lid.py."""
    import json
    import sys
    from nlp_lab.lid import LIDSystem
    from nlp_lab.paths import RESULTS_DIR, ROOT
    f = RESULTS_DIR / "short_text_lid.json"
    if not f.exists():
        pytest.skip("short_text_lid.json not present")
    alpha = json.loads(f.read_text(encoding="utf-8"))["selected_alpha"]
    sys.path.insert(0, str(ROOT / "experiments"))
    from short_text_lid import fused_predict
    lid = LIDSystem.load(short_text_alpha=alpha)
    texts = ["Yeno hengidiyaa", "Sukham ano", "enna panra", "Oota aayta", "kal milte hain", "see you soon"]
    expected = fused_predict(lid.sentence_model, lid.word_model, texts, alpha)
    got = [max(p, key=p.get) for p in (lid.sentence_probs(t) for t in texts)]
    assert got == list(expected)


def test_code_mixing_mixed_script(client):
    r = client.post("/api/nlp/code-mixing", json={"text": "ನಾನು today ತುಂಬಾ busy ಇದ್ದೀನಿ"}).json()
    langs = [t["lang"] for t in r["tokens"]]
    assert langs[0] == "kn" and langs[2] == "kn"
    assert r["stats"]["is_code_mixed"] in (True, False)
    assert set(r["stats"]["language_distribution"]) <= {"kn", "en"}


def test_sentiment_explanation_is_exact(client):
    app = client.app
    for kind in ("logreg", "linear_svm"):
        r = client.post("/api/nlp/sentiment", json={"text": "Super movie bro, thumba chennagide", "model": kind,
                                                    "language": "kn"}).json()
        assert r["supported"] and r["label"] in ("positive", "negative", "neutral")
        ex = r["explanation"]
        tok_sum = sum(t["contribution"] for t in ex["tokens"])
        # token contributions + non-token + bias reproduce the model's raw score
        assert abs(tok_sum + ex["non_token_contribution"] + ex["bias"] - r["decision_scores"][r["label"]]) < 1e-3
        if kind == "logreg":
            assert abs(sum(r["probabilities"].values()) - 1) < 1e-3
        else:
            assert r["probabilities"] is None and r["confidence"] is None


def test_sentiment_unsupported_language(client):
    r = client.post("/api/nlp/sentiment", json={"text": "This is an English sentence about the weather"}).json()
    assert r["supported"] is False and "Kannada, Tamil and Malayalam" in r["message"]


def test_compare_and_models(client):
    r = client.post("/api/nlp/compare", json={"text": "padam semma mass", "language": "ta"}).json()
    assert len(r["results"]) == 2
    models = client.get("/api/nlp/models").json()
    assert {m["language"] for m in models} == {"kn", "ta", "ml"}
    f = client.get(f"/api/nlp/models/{models[0]['id']}/features").json()
    assert set(f["features"]) == {"negative", "neutral", "positive"}


def test_dataset_samples_and_results(client):
    s = client.get("/api/nlp/datasets/samples", params={"lang": "kn", "label": "negative", "limit": 5}).json()
    assert s["total"] > 0 and all(r["lang"] == "kn" and r["label"] == "negative" for r in s["rows"])
    res = client.get("/api/nlp/results").json()
    assert res["sentiment_results"] and res["lid_results"]


def test_validation(client):
    assert client.post("/api/nlp/sentiment", json={"text": ""}).status_code == 422
    assert client.post("/api/nlp/sentiment", json={"text": "x", "model": "bert"}).status_code == 422
