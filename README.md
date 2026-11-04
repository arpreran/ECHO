---
title: ECHO
emoji: 🗣️
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Romanized Dravidian language ID, code-mixing and sentiment
---

# ECHO — Intelligence, in Two Modes

> **Live demo:** https://huggingface.co/spaces/arpreran/ECHO · **Code:** https://github.com/arpreran/ECHO
>
> In the hosted demo the **NLP Lab** is fully working. The **Companion** needs a chat model (local Ollama by default), which the free host does not provide, so it shows "Ollama not running" there; run ECHO locally to use it.

ECHO is one application with two independent products:

| Mode | What it is | AI used |
|---|---|---|
| **ECHO Companion** (`/companion`) | Voice-first multilingual AI companion (Kannada, English, Hindi, Tamil, Malayalam; Romanized, native-script and code-mixed input) with SQLite history and user-approved memory | Conversational model: local Ollama (`qwen3:8b`, default, no key) or Gemini, OpenAI or Anthropic (configurable) |
| **ECHO NLP Lab** (`/lab`) | Research lab: Romanized Dravidian language identification, code-mixing analysis, explainable sentiment analysis | **Classical ML only** (scikit-learn). Never calls an AI API |

The research paper for the NLP Lab is in [`research_paper/ECHO_NLP_Lab_paper.md`](research_paper/ECHO_NLP_Lab_paper.md).

## Architecture

```
Browser (React, :5173) ──/api proxy──► FastAPI backend (:8000)
                                          ├── /api/companion/* ─► CompanionService ─► Gemini | OpenAI | Anthropic (HTTPS, key from backend/.env)
                                          │                         └─► SQLite (database/echo.sqlite3): conversations, messages, memories, preferences
                                          └── /api/nlp/*       ─► NLPService (scikit-learn models in models/, results in experiments/results/)
```

* **Separate services.** The Companion and the NLP Lab are separate packages.
* **One-way sharing.** The Companion reuses only the Lab's classical language-ID model, to label message languages.
* **No AI APIs in the Lab.** The Lab never calls an external AI API.
* **Voice in the browser by default.** The Web Speech API handles recognition and `speechSynthesis` handles speech. Server-side speech-to-text and text-to-speech are optional.

## Project layout

```
backend/            FastAPI app
  app/              entry point (main.py) and settings (config.py)
  companion/        chat/voice providers, conversation service, routes
  nlp_lab/          script ID, LID, normalisation, code-mixing, sentiment, explanations, routes
  database/         SQLite layer with numbered migrations
  tests/            pytest suite
frontend/           React + TypeScript + Vite + Tailwind + Framer Motion + Recharts
experiments/        train_lid.py, run_sentiment.py, results/*.json
datasets/raw/       downloaded corpora (not committed)
models/             trained .joblib models
database/           echo.sqlite3 (created on first run)
research_paper/     paper + verified references.bib
```

## Requirements (Windows)

* Python 3.11+ and Node.js 20+
* Chrome or Edge for browser speech recognition
* For the Companion, either [Ollama](https://ollama.com) with a model installed (default: `ollama pull qwen3:8b`), or one API key for Gemini, OpenAI or Anthropic. The NLP Lab needs neither.

## Setup

Run these from `E:\ECHO` in PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\pip install -r backend\requirements.txt
npm --prefix frontend install
copy backend\.env.example backend\.env
```

The default configuration uses local Ollama and needs no key: make sure Ollama is running and `qwen3:8b` is installed (`ollama list`). For a cloud provider, edit `backend\.env` instead (see *Switching providers*). Keys are read only by the backend; the browser never receives them, and `.env` is git-ignored.

### Data and models

Trained models and experiment results are included. To rebuild them from scratch:

1. Download the datasets into `datasets\raw\`:
   * DravidianCodeMix-2020 (CC BY 4.0): `DravidianCodeMix-2020.zip` from https://github.com/bharathichezhiyan/DravidianCodeMix-Dataset, unzipped into `datasets\raw\dravidiancodemix\`.
   * Dakshina v1.0 (CC BY-SA 4.0): https://github.com/google-research-datasets/dakshina. Extract the `kn`, `ta`, `ml` and `hi` `lexicons/*.sampled.*` and `romanized/*.rejoined.*.roman.txt` files into `datasets\raw\dakshina\dakshina_dataset_v1.0\`.
   * Tatoeba English sentences (CC BY 2.0 FR): `eng_sentences.tsv.bz2` from https://downloads.tatoeba.org/exports/per_language/eng/, saved in `datasets\raw\tatoeba\`.
2. Train, in this order (roughly 4 + 2 + 15 minutes on a laptop CPU):

```powershell
.venv\Scripts\python experiments\train_lid.py
.venv\Scripts\python experiments\short_text_lid.py
.venv\Scripts\python experiments\run_sentiment.py
```

The runs are deterministic (seed 42). Two consecutive `train_lid.py` runs produce identical results.

## Run

Use two terminals from `E:\ECHO`:

```powershell
.venv\Scripts\python -m uvicorn app.main:app --app-dir backend --port 8000
```

```powershell
npm --prefix frontend run dev
```

Open http://localhost:5173. The dev server proxies `/api` to the backend. Health check: http://localhost:8000/api/health.

## Configuration (`backend/.env`)

| Variable | Values | Default |
|---|---|---|
| `ECHO_CHAT_PROVIDER` | `ollama`, `gemini`, `openai`, `anthropic` | `ollama` |
| `OLLAMA_BASE_URL` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | installed Ollama model | `qwen3:8b` |
| `OLLAMA_TIMEOUT` | seconds to wait for a reply | `180` |
| `ECHO_STT_PROVIDER` | `browser` (Web Speech API), `gemini`, `openai` | `browser` |
| `ECHO_TTS_PROVIDER` | `browser` (speechSynthesis), `gemini`, `openai` | `browser` |
| `GEMINI_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | your key | — |
| `OPENAI_BASE_URL` | any OpenAI-compatible server (Groq, OpenRouter, Sarvam AI, …) | `https://api.openai.com/v1` |
| `*_CHAT_MODEL`, `*_STT_MODEL`, `*_TTS_MODEL` | model ids | see `.env.example` |
| `ECHO_DATABASE_PATH` | SQLite file | `database/echo.sqlite3` |

### Switching providers

Edit `backend\.env`, then restart the backend (the server does not reload `.env` by itself):

* **Ollama (default, local, free):** `ECHO_CHAT_PROVIDER=ollama`, plus `OLLAMA_MODEL=qwen3:8b` or any model shown by `ollama list`. ECHO never downloads models.
* **Gemini:** `ECHO_CHAT_PROVIDER=gemini` and `GEMINI_API_KEY=...`. The project behind the key must have free-tier access or billing enabled.
* **OpenAI / Anthropic:** `ECHO_CHAT_PROVIDER=openai` or `anthropic`, with `OPENAI_API_KEY` or `ANTHROPIC_API_KEY`.
* **Groq, OpenRouter, Sarvam AI and other OpenAI-compatible servers:** `ECHO_CHAT_PROVIDER=openai`, `OPENAI_BASE_URL` set to the server (examples in `.env.example`), `OPENAI_API_KEY` set to *that* service's key, and `OPENAI_CHAT_MODEL` set to one of its model ids. If the server rejects JSON mode, ECHO retries once without it. Free tiers and model ids are set by each provider and change over time.

The current provider is shown in the Companion header and under Settings → Providers.

**Choosing a voice pipeline.** ECHO uses a speech-to-text → chat → text-to-speech pipeline rather than a single realtime voice API, because each stage can be swapped and code-mixed text can be language-tagged between stages.
* Browser STT is free, but recognises one locale at a time; pick Kannada (`kn-IN`) in Settings for Kannada-English speech.
* `gemini` or `openai` STT accepts mixed-language audio but costs money and sends your audio to that provider.
* Browser TTS depends on the voices installed in the operating system. Server TTS (`gemini`, `openai`) gives more natural voices.

**What the Companion sends to the provider** (with Ollama, nothing leaves your computer):
* the system prompt;
* up to 20 recent messages;
* your approved memories;
* up to 3 matching snippets from earlier conversations (each can be turned off);
* the new message;
* your audio and the reply text, only if server STT or TTS is enabled.

This list is also shown in Settings.

## Tests

```powershell
.venv\Scripts\python -m pytest -c backend\pytest.ini --rootdir backend backend\tests
npm --prefix frontend test
npm --prefix frontend run build
```

* The Companion tests replace the provider with an `httpx.MockTransport`, so they need no key and make no network calls.
* The NLP tests use the trained models, and include a check that each explanation exactly reproduces the model score.

## API overview

* **General:** `GET /api/health`
* **Companion:**
  * `POST /api/companion/chat`
  * `GET/POST /api/companion/conversations`, `GET/PATCH/DELETE /api/companion/conversations/{id}`
  * `GET /api/companion/conversations/search?q=`, `GET /api/companion/conversations/{id}/export?format=json|markdown`
  * `GET/POST /api/companion/preferences`
  * `GET/POST/PATCH/DELETE /api/companion/memories`
  * `POST /api/companion/voice/transcribe`, `POST /api/companion/voice/speak`
  * `GET /api/companion/status`
* **NLP Lab:**
  * `POST /api/nlp/analyze`, `/language-identification`, `/code-mixing`, `/sentiment`, `/compare`
  * `GET /api/nlp/models`, `/models/{id}/features`, `/datasets`, `/datasets/samples`, `/results`

## Troubleshooting

| Symptom | Fix |
|---|---|
| Landing page says **Backend offline** | Start the backend (see Run). Check `http://localhost:8000/api/health`. |
| Companion says **Chat API not configured** | Put the key for `ECHO_CHAT_PROVIDER` in `backend\.env` and restart the backend. |
| Companion says **Ollama not running** | Start the Ollama app (or run `ollama serve`), then reload the page. |
| Companion says **qwen3:8b not installed** | Run `ollama pull qwen3:8b` yourself, or set `OLLAMA_MODEL` to an installed model (`ollama list`). |
| Ollama replies time out | The first reply loads the model into memory. Wait and retry, or raise `OLLAMA_TIMEOUT`. |
| Companion error: `rejected the API key (401)` | The key is wrong. Fix it in `backend\.env`. |
| Companion error: `denied the request (403)` … `free tier is not available` | The Google Cloud project behind the key needs billing enabled (https://aistudio.google.com/billing), or use a key from a project that has free-tier access. |
| Companion error: `Gemini error 404` … `no longer available` | Set `GEMINI_CHAT_MODEL` to a model your key can use (the default is `gemini-3.8-flash`). |
| Companion error: `rate limit or quota exceeded` | Wait, or check your provider's quota or billing. |
| **Talk** button disabled, “no speech recognition” | Use Chrome or Edge, or set `ECHO_STT_PROVIDER=gemini` or `openai`. |
| “Microphone permission was denied” | Allow the microphone for `localhost:5173` in the browser's site settings, then press **Try again**. |
| “this browser has no kn-IN voice” (reply shown as text only) | Windows and Chrome usually have no Kannada, Tamil or Malayalam voice. Open ECHO in Microsoft Edge, which offers online voices for these languages, or set `ECHO_TTS_PROVIDER=gemini` or `openai`. |
| NLP Lab banner “models missing” | Run the two training commands in *Data and models*. |
| `InconsistentVersionWarning` from scikit-learn | Install the pinned version: `.venv\Scripts\pip install -r backend\requirements.txt`. |
| `npm` build fails with `esbuild ... ENOENT` | The project path is inside a virtualised or very long folder. Keep the project in a normal folder such as `E:\ECHO`. |
| Port already in use | Stop the other process, or run uvicorn with `--port 8001` and change the proxy target in `frontend\vite.config.ts`. |

## Database

* SQLite file: `database\echo.sqlite3`, created on first start. Override it with `ECHO_DATABASE_PATH`.
* The schema is versioned in `backend\database\db.py` (`MIGRATIONS`) and applied automatically on startup.
* Tables:
  * `conversations`, `messages` (with an FTS5 index for search);
  * `preferences`, `memories`;
  * `schema_version`.
* To reset, stop the backend and delete `database\echo.sqlite3`.

## Models and results

| File | Contents |
|---|---|
| `models\lid\sentence_lid.joblib`, `word_lid.joblib` | Romanized LID models |
| `models\sentiment\{kn,ta,ml}_{logreg,linear_svm}.joblib` | Deployed sentiment models (feature set and C chosen on dev) |
| `experiments\results\lid_results.json` | LID metrics, langid.py baseline, length analysis |
| `experiments\results\short_text_lid.json` | Short-input fusion experiment; its selected weight is used by the app |
| `experiments\results\sentiment_results.json` | All 36 sentiment runs, bootstrap tests, routing experiment |
| `experiments\results\dataset_audit.json` | Dataset audit and cleaning statistics |
| `experiments\results\first_run\` | Results of the first full run, kept for the reproducibility comparison |

## Known limitations

* **Short inputs:** sentence-level language ID reaches 96.6–99.7% on inputs of four or more words, but 72.8% on 1–3-word inputs.
  * The app adds word-model evidence for short inputs, with the weight chosen on validation data (`experiments/short_text_lid.py`). This raises held-out short-input accuracy to 74.2%.
  * Informal Kannada such as "Yeno hengidiyaa" is still identified as Romanized Tamil, because those spellings are absent from the training data.
* **Word-level LID:** 72% accuracy on held-out dictionary words. No gold word-level annotation of code-mixed sentences was available, so in-context word tagging is not evaluated.
* **Sentiment domain and data:** the models are trained on YouTube movie comments and are not evaluated on chat. Kannada has only 561 test comments.
* **Companion with local `qwen3:8b`:** live conversations were tested in English, Kannada–English and Hindi–English, including context and storage.
  * English and Hindi–English work well. Hindi–English replies tend to restate the user's line first.
  * Romanized Kannada is only partly understood. The model gets the gist ("busy", "tired", "exam stress") but usually replies in English, mislabels its own replies as "Kannada-English", and can fail on Kannada-only questions; for example, it may read *bagge* ("about") as "bag".
  * Expect better Kannada from a larger model (e.g. `OLLAMA_MODEL=qwen3:14b`, slower) or a cloud provider. Gemini was not testable: the available keys returned HTTP 403 (free tier unavailable for the project).
  * Gemini, OpenAI and Anthropic paths are covered by automated tests with a mocked provider only.
* **Voice not verified:** microphone input, speech recognition, text-to-speech audio, interrupt and stop-speaking have not been tested with a real microphone and speakers. During testing, the browser's speech-synthesis events did drive the waveform. No streaming: each reply arrives in one piece.
* **No authentication:** ECHO is a single-user local application. Anyone with access to the machine can read `database/echo.sqlite3`.
