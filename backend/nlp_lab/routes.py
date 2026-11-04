from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from .service import NotReady

router = APIRouter(prefix="/api/nlp", tags=["nlp-lab"])


class TextIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


class SentimentIn(TextIn):
    model: Literal["logreg", "linear_svm"] = "logreg"
    language: Literal["auto", "kn", "ta", "ml"] = "auto"


def nlp(request: Request):
    return request.app.state.nlp


def guard(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotReady as e:
        raise HTTPException(503, str(e))


@router.get("/status")
def status(request: Request):
    return nlp(request).status()


@router.post("/language-identification")
def language_identification(body: TextIn, request: Request):
    return guard(nlp(request).language_identification, body.text)


@router.post("/code-mixing")
def code_mixing(body: TextIn, request: Request):
    return guard(nlp(request).code_mixing, body.text)


@router.post("/sentiment")
def sentiment(body: SentimentIn, request: Request):
    return guard(nlp(request).sentiment, body.text, body.model, body.language)


@router.post("/compare")
def compare(body: SentimentIn, request: Request):
    return guard(nlp(request).compare, body.text, body.language)


@router.post("/analyze")
def analyze(body: SentimentIn, request: Request):
    return guard(nlp(request).analyze, body.text, body.model, body.language)


@router.get("/models")
def models(request: Request):
    return nlp(request).models_info()


@router.get("/models/{model_id}/features")
def model_features(model_id: str, request: Request, n: int = Query(20, ge=5, le=100)):
    try:
        return nlp(request).global_features(model_id, n)
    except KeyError:
        raise HTTPException(404, "Unknown model")


@router.get("/results")
def results(request: Request):
    return nlp(request).results()


@router.get("/datasets")
def datasets(request: Request):
    res = nlp(request).results()
    return {
        "sources": [
            {"name": "DravidianCodeMix-2020", "use": "Sentiment (Kannada, Tamil, Malayalam)",
             "licence": "CC BY 4.0", "url": "https://zenodo.org/record/4750858",
             "citation": "Chakravarthi et al. (2022), Language Resources and Evaluation, doi:10.1007/s10579-022-09583-7"},
            {"name": "Dakshina v1.0", "use": "Romanized LID: attested romanization lexicons and romanized sentences "
                                             "(kn, ta, ml, hi)",
             "licence": "CC BY-SA 4.0", "url": "https://github.com/google-research-datasets/dakshina",
             "citation": "Roark et al. (2020), LREC"},
            {"name": "Tatoeba English sentences", "use": "English class for LID",
             "licence": "CC BY 2.0 FR", "url": "https://tatoeba.org/en/downloads", "citation": "Tatoeba Project"},
        ],
        "audit": res["dataset_audit"],
        "lid_data": {k: (res["lid_results"] or {}).get(k) for k in ("word_data", "sentence_data")},
    }


@router.get("/datasets/samples")
def samples(request: Request, lang: Literal["kn", "ta", "ml"] | None = None,
            label: Literal["positive", "negative", "neutral"] | None = None,
            text_type: Literal["romanized", "native", "mixed_script"] | None = None,
            split: Literal["train", "dev", "test"] | None = None, q: str | None = Query(None, max_length=100),
            offset: int = Query(0, ge=0), limit: int = Query(25, ge=1, le=100)):
    return nlp(request).samples(lang, label, text_type, split, q, offset, limit)
