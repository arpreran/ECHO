"""Configurable conversational / speech providers for ECHO Companion.

Only REST calls over httpx; API keys come from backend environment variables
and are never returned to the client.

Chat:  ollama | gemini | openai | anthropic
       (openai also covers any OpenAI-compatible server via OPENAI_BASE_URL: Groq, OpenRouter, Sarvam AI, ...)
STT:   browser (Web Speech API in the client) | gemini | openai
TTS:   browser (speechSynthesis in the client) | gemini | openai
"""
from __future__ import annotations

import base64
import io
import json
import wave
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from app.config import Settings


class ProviderError(Exception):
    def __init__(self, message: str, status: int = 502, retryable: bool = True):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


@dataclass
class ChatTurn:
    role: str      # "user" | "assistant"
    content: str


def _raise_for(resp: httpx.Response, provider: str) -> None:
    if resp.status_code < 400:
        return
    try:
        detail = resp.json()
        detail = detail.get("error", detail)
        if isinstance(detail, dict):
            detail = detail.get("message") or json.dumps(detail)[:300]
    except Exception:
        detail = resp.text[:300]
    if resp.status_code == 401:
        raise ProviderError(f"{provider} rejected the API key (401): {detail}", 502, retryable=False)
    if resp.status_code == 403:
        # Also used for billing / project-permission blocks, not only bad keys.
        raise ProviderError(f"{provider} denied the request (403): {detail}", 502, retryable=False)
    if resp.status_code == 429:
        raise ProviderError(f"{provider} rate limit or quota exceeded: {detail}", 429)
    raise ProviderError(f"{provider} error {resp.status_code}: {detail}")


class Providers:
    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None):
        self.s = settings
        self.client = httpx.Client(timeout=settings.request_timeout, transport=transport)

    # ------------------------------------------------------------ status
    def key_for(self, provider: str) -> str:
        return {"gemini": self.s.gemini_api_key, "openai": self.s.openai_api_key,
                "anthropic": self.s.anthropic_api_key}.get(provider, "")

    def chat_model(self) -> str:
        return {"gemini": self.s.gemini_chat_model, "openai": self.s.openai_chat_model,
                "anthropic": self.s.anthropic_chat_model, "ollama": self.s.ollama_model}.get(self.s.chat_provider, "unknown")

    def openai_label(self) -> str:
        """'OpenAI', or the host of the OpenAI-compatible server in use (e.g. api.groq.com)."""
        host = urlparse(self.s.openai_base_url).hostname or self.s.openai_base_url
        return "OpenAI" if host == "api.openai.com" else host

    def status(self) -> dict:
        def configured(p: str) -> bool:
            return p == "browser" or bool(self.key_for(p))
        chat = {"provider": self.s.chat_provider, "model": self.chat_model(),
                "configured": self.s.chat_provider == "ollama" or bool(self.key_for(self.s.chat_provider))}
        if self.s.chat_provider == "ollama":
            chat.update(self._ollama_status())
        elif self.s.chat_provider == "openai":
            chat["label"] = self.openai_label()
        return {
            "chat": chat,
            "stt": {"provider": self.s.stt_provider, "configured": configured(self.s.stt_provider)},
            "tts": {"provider": self.s.tts_provider, "configured": configured(self.s.tts_provider),
                    "model": self.s.gemini_tts_model if self.s.tts_provider == "gemini"
                    else self.s.openai_tts_model if self.s.tts_provider == "openai" else None},
        }

    def _need_key(self, provider: str) -> str:
        key = self.key_for(provider)
        if not key:
            env = {"gemini": "GEMINI_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}[provider]
            raise ProviderError(f"{provider} is selected but {env} is not set in backend/.env", 503, retryable=False)
        return key

    # -------------------------------------------------------------- chat
    def chat_json(self, system: str, turns: list[ChatTurn], temperature: float = 0.8) -> dict:
        """Returns the model's JSON object reply (the system prompt asks for JSON)."""
        p = self.s.chat_provider
        try:
            if p == "gemini":
                text = self._gemini_chat(system, turns, temperature)
            elif p == "openai":
                text = self._openai_chat(system, turns, temperature)
            elif p == "anthropic":
                text = self._anthropic_chat(system, turns, temperature)
            elif p == "ollama":
                text = self._ollama_chat(system, turns, temperature)
            else:
                raise ProviderError(f"Unknown ECHO_CHAT_PROVIDER '{p}'", 500, retryable=False)
        except httpx.TimeoutException as e:
            raise ProviderError(f"{p} timed out", 504) from e
        except httpx.ConnectError as e:
            if p == "ollama":
                raise ProviderError(f"Ollama is not running at {self.s.ollama_base_url}. Start it (e.g. open the "
                                    "Ollama app or run `ollama serve`) and try again.", 503) from e
            raise ProviderError(f"Could not reach {p}: {e.__class__.__name__}", 502) from e
        except httpx.HTTPError as e:
            raise ProviderError(f"Could not reach {p}: {e.__class__.__name__}", 502) from e
        return parse_json_reply(text)

    # ------------------------------------------------------------ ollama
    def _ollama_status(self) -> dict:
        """Reachability and model availability, so the UI can show a precise state."""
        try:
            r = self.client.get(f"{self.s.ollama_base_url}/api/tags", timeout=3)
            names = {m.get("name") for m in r.json().get("models", [])}
            return {"reachable": True, "model_available": self.s.ollama_model in names}
        except (httpx.HTTPError, ValueError):
            return {"reachable": False, "model_available": False}

    def _ollama_chat(self, system, turns, temperature) -> str:
        body = {
            "model": self.s.ollama_model,
            "messages": [{"role": "system", "content": system}] + [{"role": t.role, "content": t.content} for t in turns],
            "format": "json",
            "stream": False,
            "think": False,   # qwen3 is a thinking model; ECHO only needs the final answer
            "options": {"temperature": temperature},
        }
        r = self.client.post(f"{self.s.ollama_base_url}/api/chat", json=body, timeout=self.s.ollama_timeout)
        if r.status_code == 404:
            raise ProviderError(f"Ollama model '{self.s.ollama_model}' is not installed. Install it yourself with "
                                f"`ollama pull {self.s.ollama_model}` or set OLLAMA_MODEL in backend/.env.", 503, False)
        _raise_for(r, "Ollama")
        return r.json().get("message", {}).get("content", "")

    def _gemini_chat(self, system, turns, temperature) -> str:
        key = self._need_key("gemini")
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "model" if t.role == "assistant" else "user", "parts": [{"text": t.content}]}
                         for t in turns],
            "generationConfig": {"temperature": temperature, "responseMimeType": "application/json"},
        }
        r = self.client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.s.gemini_chat_model}:generateContent",
            headers={"x-goog-api-key": key}, json=body)
        _raise_for(r, "Gemini")
        data = r.json()
        try:
            return "".join(p.get("text", "") for p in data["candidates"][0]["content"]["parts"])
        except (KeyError, IndexError) as e:
            reason = data.get("promptFeedback", {}).get("blockReason") or data.get("candidates", [{}])[0].get("finishReason")
            raise ProviderError(f"Gemini returned no text (reason: {reason})") from e

    def _openai_chat(self, system, turns, temperature) -> str:
        key = self._need_key("openai")
        msgs = [{"role": "system", "content": system}] + [{"role": t.role, "content": t.content} for t in turns]
        body = {"model": self.s.openai_chat_model, "messages": msgs, "temperature": temperature,
                "response_format": {"type": "json_object"}}
        url, headers = f"{self.s.openai_base_url}/chat/completions", {"Authorization": f"Bearer {key}"}
        r = self.client.post(url, headers=headers, json=body)
        if r.status_code == 400:
            # Some OpenAI-compatible servers or models reject JSON mode; the prompt still asks for JSON.
            body.pop("response_format")
            r = self.client.post(url, headers=headers, json=body)
        _raise_for(r, self.openai_label())
        return r.json()["choices"][0]["message"]["content"]

    def _anthropic_chat(self, system, turns, temperature) -> str:
        key = self._need_key("anthropic")
        msgs = [{"role": t.role, "content": t.content} for t in turns]
        r = self.client.post("https://api.anthropic.com/v1/messages",
                             headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                             json={"model": self.s.anthropic_chat_model, "system": system, "messages": msgs,
                                   "max_tokens": 1024, "temperature": temperature})
        _raise_for(r, "Anthropic")
        return "".join(b.get("text", "") for b in r.json()["content"] if b.get("type") == "text")

    # --------------------------------------------------------------- STT
    def transcribe(self, audio: bytes, mime: str, language_hint: str | None) -> dict:
        p = self.s.stt_provider
        if p == "browser":
            raise ProviderError("Server speech-to-text is disabled (ECHO_STT_PROVIDER=browser).", 400, False)
        try:
            if p == "openai":
                key = self._need_key("openai")
                prompt = ("Conversational speech that may mix English with Kannada, Hindi, Tamil or Malayalam. "
                          "Write each language in the script the speaker would naturally type.")
                r = self.client.post(f"{self.s.openai_base_url}/audio/transcriptions",
                                     headers={"Authorization": f"Bearer {key}"},
                                     files={"file": ("speech" + _ext(mime), audio, mime)},
                                     data={"model": self.s.openai_stt_model, "prompt": prompt})
                _raise_for(r, self.openai_label())
                return {"text": r.json().get("text", "").strip(), "provider": "openai", "model": self.s.openai_stt_model}
            if p == "gemini":
                key = self._need_key("gemini")
                hint = f" The expected primary language tag is {language_hint}." if language_hint else ""
                body = {"contents": [{"role": "user", "parts": [
                    {"inlineData": {"mimeType": mime.split(";")[0], "data": base64.b64encode(audio).decode()}},
                    {"text": "Transcribe this speech verbatim. It may mix English with Kannada, Hindi, Tamil or "
                             "Malayalam. Keep each word in the language spoken; do not translate." + hint +
                             " Return only the transcript."}]}],
                    "generationConfig": {"temperature": 0}}
                r = self.client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{self.s.gemini_chat_model}:generateContent",
                    headers={"x-goog-api-key": key}, json=body)
                _raise_for(r, "Gemini")
                parts = r.json()["candidates"][0]["content"]["parts"]
                return {"text": "".join(x.get("text", "") for x in parts).strip(), "provider": "gemini",
                        "model": self.s.gemini_chat_model}
        except httpx.HTTPError as e:
            raise ProviderError(f"Could not reach {p} for transcription: {e.__class__.__name__}") from e
        raise ProviderError(f"Unknown ECHO_STT_PROVIDER '{p}'", 500, False)

    # --------------------------------------------------------------- TTS
    def speak(self, text: str, voice: str | None) -> tuple[bytes, str]:
        p = self.s.tts_provider
        if p == "browser":
            raise ProviderError("Server text-to-speech is disabled (ECHO_TTS_PROVIDER=browser).", 400, False)
        try:
            if p == "openai":
                key = self._need_key("openai")
                r = self.client.post(f"{self.s.openai_base_url}/audio/speech",
                                     headers={"Authorization": f"Bearer {key}"},
                                     json={"model": self.s.openai_tts_model, "voice": voice or "alloy",
                                           "input": text, "response_format": "mp3"})
                _raise_for(r, self.openai_label())
                return r.content, "audio/mpeg"
            if p == "gemini":
                key = self._need_key("gemini")
                body = {"contents": [{"parts": [{"text": text}]}],
                        "generationConfig": {"responseModalities": ["AUDIO"], "speechConfig": {
                            "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice or "Kore"}}}}}
                r = self.client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{self.s.gemini_tts_model}:generateContent",
                    headers={"x-goog-api-key": key}, json=body)
                _raise_for(r, "Gemini")
                inline = r.json()["candidates"][0]["content"]["parts"][0]["inlineData"]
                pcm = base64.b64decode(inline["data"])
                return pcm_to_wav(pcm, rate=_rate(inline.get("mimeType", ""))), "audio/wav"
        except httpx.HTTPError as e:
            raise ProviderError(f"Could not reach {p} for speech: {e.__class__.__name__}") from e
        raise ProviderError(f"Unknown ECHO_TTS_PROVIDER '{p}'", 500, False)


def _ext(mime: str) -> str:
    return {"audio/webm": ".webm", "audio/ogg": ".ogg", "audio/wav": ".wav", "audio/mpeg": ".mp3",
            "audio/mp4": ".m4a"}.get(mime.split(";")[0], ".webm")


def _rate(mime: str) -> int:
    for part in mime.split(";"):
        if part.strip().startswith("rate="):
            return int(part.split("=")[1])
    return 24000


def pcm_to_wav(pcm: bytes, rate: int = 24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def parse_json_reply(text: str) -> dict:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        t = t[t.find("{"):]
    start, end = t.find("{"), t.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(t[start:end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    # The model ignored the JSON instruction: keep its text as the reply.
    return {"reply": text.strip(), "_unstructured": True}
