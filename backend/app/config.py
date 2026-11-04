"""Settings from environment variables (backend/.env). Secrets never leave the backend."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
ROOT = BACKEND_DIR.parent
load_dotenv(BACKEND_DIR / ".env")


def _get(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True)
class Settings:
    chat_provider: str = _get("ECHO_CHAT_PROVIDER", "ollama")        # ollama | gemini | openai | anthropic
    stt_provider: str = _get("ECHO_STT_PROVIDER", "browser")         # browser | gemini | openai
    tts_provider: str = _get("ECHO_TTS_PROVIDER", "browser")         # browser | gemini | openai
    gemini_api_key: str = _get("GEMINI_API_KEY")
    gemini_chat_model: str = _get("GEMINI_CHAT_MODEL", "gemini-3.8-flash")
    gemini_tts_model: str = _get("GEMINI_TTS_MODEL", "gemini-3.8-flash-tts")
    openai_api_key: str = _get("OPENAI_API_KEY")
    # Any OpenAI-compatible server: Groq, OpenRouter, Sarvam AI, ... (default: OpenAI itself)
    openai_base_url: str = _get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    openai_chat_model: str = _get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    openai_stt_model: str = _get("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")
    openai_tts_model: str = _get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
    anthropic_api_key: str = _get("ANTHROPIC_API_KEY")
    anthropic_chat_model: str = _get("ANTHROPIC_CHAT_MODEL", "claude-haiku-5-5")
    # Local Ollama server (no API key). Local models can be slow on CPU, hence the longer timeout.
    ollama_base_url: str = _get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    ollama_model: str = _get("OLLAMA_MODEL", "qwen3:8b")
    ollama_timeout: float = float(_get("OLLAMA_TIMEOUT", "180"))
    database_path: str = _get("ECHO_DATABASE_PATH", str(ROOT / "database" / "echo.sqlite3"))
    cors_origins: tuple[str, ...] = tuple(
        o.strip() for o in _get("ECHO_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip())
    request_timeout: float = float(_get("ECHO_REQUEST_TIMEOUT", "60"))


settings = Settings()
