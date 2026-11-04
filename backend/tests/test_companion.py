"""Companion API tests. The external provider is replaced by an httpx MockTransport,
so these tests exercise the real request/response code without network or keys."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from companion.providers import Providers


class FakeGemini:
    def __init__(self):
        self.requests = []
        self.fail_with = None
        self.reply = {"reply": "Chennagiddini maga! Neenu hegiddiya?", "emotion": "happy",
                      "user_mood": "cheerful", "reply_language": "Kannada-English (Romanized)",
                      "memory_suggestions": ["Studies at college", "Card number 4111 1111 1111 1111"]}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.fail_with:
            return httpx.Response(self.fail_with, json={"error": {"message": "boom"}})
        assert request.headers["x-goog-api-key"] == "test-key"
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": json.dumps(self.reply)}]}}]})


@pytest.fixture
def env(tmp_path):
    fake = FakeGemini()
    settings = Settings(chat_provider="gemini", gemini_api_key="test-key",
                        database_path=str(tmp_path / "t.sqlite3"))
    app = create_app(settings, providers=Providers(settings, transport=httpx.MockTransport(fake)))
    with TestClient(app) as client:
        yield client, fake


def body_of(req):
    return json.loads(req.content)


def test_health_reports_schema_and_providers(env):
    c, _ = env
    r = c.get("/api/health").json()
    assert r["status"] == "ok" and r["database_schema_version"] == 2
    assert r["companion"]["chat"]["configured"] is True
    assert "test-key" not in json.dumps(r)


def test_chat_persists_and_uses_context(env):
    c, fake = env
    r = c.post("/api/companion/chat", json={"message": "Yeno maga, hegiddiya?"}).json()
    cid = r["conversation_id"]
    assert r["emotion"] == "happy" and r["persisted"]
    # sensitive suggestion is filtered out
    assert r["memory_suggestions"] == ["Studies at college"]
    c.post("/api/companion/memories", json={"content": "Likes filter coffee"})
    c.post("/api/companion/chat", json={"message": "Bro, ivattu college ge hogbeku", "conversation_id": cid})
    sent = body_of(fake.requests[-1])
    texts = [p["text"] for t in sent["contents"] for p in t["parts"]]
    assert texts[0] == "Yeno maga, hegiddiya?"            # earlier turn is in context
    assert "Likes filter coffee" in sent["systemInstruction"]["parts"][0]["text"]
    conv = c.get(f"/api/companion/conversations/{cid}").json()
    assert [m["sender"] for m in conv["messages"]] == ["user", "assistant", "user", "assistant"]


def test_provider_failure_saves_nothing(env):
    c, fake = env
    fake.fail_with = 429
    r = c.post("/api/companion/chat", json={"message": "hello"})
    assert r.status_code == 429 and r.json()["detail"]["retryable"] is True
    assert c.get("/api/companion/conversations").json() == []


def test_bad_key_is_not_retryable(env):
    c, fake = env
    fake.fail_with = 401
    r = c.post("/api/companion/chat", json={"message": "hello"})
    assert r.status_code == 502 and r.json()["detail"]["retryable"] is False


def test_incognito_is_not_stored(env):
    c, fake = env
    r = c.post("/api/companion/chat", json={"message": "secret chat", "incognito": True,
                                            "history": [{"role": "user", "content": "earlier"},
                                                        {"role": "assistant", "content": "ok"}]}).json()
    assert r["persisted"] is False and r["conversation_id"] is None
    assert c.get("/api/companion/conversations").json() == []
    assert len(body_of(fake.requests[-1])["contents"]) == 3


def test_conversation_crud_search_export(env):
    c, _ = env
    cid = c.post("/api/companion/chat", json={"message": "Aaj college jaana hai but mood nahi hai"}).json()["conversation_id"]
    assert c.patch(f"/api/companion/conversations/{cid}", json={"title": "College"}).json()["title"] == "College"
    hits = c.get("/api/companion/conversations/search", params={"q": "college"}).json()
    assert hits and hits[0]["conversation_id"] == cid
    md = c.get(f"/api/companion/conversations/{cid}/export", params={"format": "markdown"})
    assert md.status_code == 200 and "Aaj college jaana hai" in md.text
    assert c.delete(f"/api/companion/conversations/{cid}").status_code == 204
    assert c.get(f"/api/companion/conversations/{cid}").status_code == 404
    assert c.get("/api/companion/conversations/search", params={"q": "college"}).json() == []


def test_memories_crud_and_sensitive_rejection(env):
    c, _ = env
    m = c.post("/api/companion/memories", json={"content": "Prefers Kannada-English replies"}).json()
    assert c.patch(f"/api/companion/memories/{m['id']}", json={"content": "Prefers Kannada replies"}).json()["content"] \
        == "Prefers Kannada replies"
    assert c.post("/api/companion/memories", json={"content": "My password is hunter2"}).status_code == 422
    assert c.delete(f"/api/companion/memories/{m['id']}").status_code == 204
    assert c.get("/api/companion/memories").json() == []


def test_preferences_validation(env):
    c, _ = env
    assert c.post("/api/companion/preferences", json={"reply_language": "kn"}).json()["reply_language"] == "kn"
    assert c.post("/api/companion/preferences", json={"reply_language": "fr"}).status_code == 422
    assert c.post("/api/companion/preferences", json={"auto_speak": "yes"}).status_code == 422
    assert c.post("/api/companion/preferences", json={"bogus": 1}).status_code == 422


def test_history_off_does_not_persist(env):
    c, _ = env
    c.post("/api/companion/preferences", json={"save_history": False})
    r = c.post("/api/companion/chat", json={"message": "hi"}).json()
    assert r["persisted"] is False
    assert c.get("/api/companion/conversations").json() == []


def test_missing_key_gives_clear_503(tmp_path):
    settings = Settings(chat_provider="openai", openai_api_key="", database_path=str(tmp_path / "x.sqlite3"))
    with TestClient(create_app(settings)) as c:
        r = c.post("/api/companion/chat", json={"message": "hi"})
        assert r.status_code == 503 and "OPENAI_API_KEY" in r.json()["detail"]["message"]


def test_key_is_read_from_environment_and_never_returned(tmp_path):
    """Fresh interpreter, dummy key in the environment: the backend must report the
    provider as configured without echoing the key in any status payload."""
    import os
    import subprocess
    import sys
    code = (
        "import json\n"
        "from fastapi.testclient import TestClient\n"
        "from app.main import create_app\n"
        "from app.config import Settings\n"
        "s = Settings()\n"
        "assert s.openai_api_key == 'dummy-key-123' and s.chat_provider == 'openai'\n"
        "with TestClient(create_app(s)) as c:\n"
        "    status = c.get('/api/companion/status').json()\n"
        "    out = json.dumps(c.get('/api/health').json()) + json.dumps(status)\n"
        "assert status['chat']['configured'] is True and status['chat']['provider'] == 'openai'\n"
        "assert 'dummy-key-123' not in out\n"
    )
    env = {**os.environ, "ECHO_CHAT_PROVIDER": "openai", "OPENAI_API_KEY": "dummy-key-123",
           "ECHO_DATABASE_PATH": str(tmp_path / "env.sqlite3")}
    r = subprocess.run([sys.executable, "-c", code], cwd=os.path.dirname(os.path.dirname(__file__)),
                       env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]


def test_persistence_survives_restart(tmp_path):
    fake = FakeGemini()
    settings = Settings(chat_provider="gemini", gemini_api_key="test-key", database_path=str(tmp_path / "p.sqlite3"))
    with TestClient(create_app(settings, providers=Providers(settings, transport=httpx.MockTransport(fake)))) as c:
        cid = c.post("/api/companion/chat", json={"message": "remember this chat"}).json()["conversation_id"]
    with TestClient(create_app(settings, providers=Providers(settings, transport=httpx.MockTransport(fake)))) as c:
        assert len(c.get(f"/api/companion/conversations/{cid}").json()["messages"]) == 2


# ------------------------------------------------------------------ Ollama (local)
class FakeOllama:
    """Mimics Ollama's /api/tags and /api/chat; `mode` selects the scenario."""

    def __init__(self, mode="ok"):
        self.mode = mode
        self.chat_bodies = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.mode == "down":
            raise httpx.ConnectError("connection refused", request=request)
        assert "authorization" not in request.headers and "x-goog-api-key" not in request.headers
        if request.url.path == "/api/tags":
            models = [{"name": "qwen3:8b"}] if self.mode == "ok" else [{"name": "llama3:8b"}]
            return httpx.Response(200, json={"models": models})
        if request.url.path == "/api/chat":
            body = json.loads(request.content)
            self.chat_bodies.append(body)
            if self.mode == "missing":
                return httpx.Response(404, json={"error": f"model '{body['model']}' not found"})
            reply = {"reply": "Ayyo, full busy day! Swalpa rest maadu.", "emotion": "supportive",
                     "user_mood": "tired", "reply_language": "Kannada-English (Romanized)", "memory_suggestions": []}
            return httpx.Response(200, json={"message": {"role": "assistant", "content": json.dumps(reply)}, "done": True})
        return httpx.Response(404)


def ollama_client(tmp_path, mode="ok"):
    fake = FakeOllama(mode)
    settings = Settings(chat_provider="ollama", gemini_api_key="", ollama_model="qwen3:8b",
                        ollama_base_url="http://ollama.test", database_path=str(tmp_path / "o.sqlite3"))
    return TestClient(create_app(settings, providers=Providers(settings, transport=httpx.MockTransport(fake)))), fake


def test_ollama_chat_needs_no_key_and_keeps_context(tmp_path):
    c, fake = ollama_client(tmp_path)
    with c:
        st = c.get("/api/companion/status").json()["chat"]
        assert st == {"provider": "ollama", "model": "qwen3:8b", "configured": True, "reachable": True, "model_available": True}
        r = c.post("/api/companion/chat", json={"message": "Yeno maga, ivattu college alli full busy idde."}).json()
        assert r["emotion"] == "supportive" and r["meta"]["provider"] == "ollama" and r["meta"]["model"] == "qwen3:8b"
        c.post("/api/companion/chat", json={"message": "Swalpa tired agide.", "conversation_id": r["conversation_id"]})
        body = fake.chat_bodies[-1]
        assert body["model"] == "qwen3:8b" and body["format"] == "json" and body["think"] is False and body["stream"] is False
        assert body["messages"][0]["role"] == "system"
        assert [m["content"] for m in body["messages"][1:]][0] == "Yeno maga, ivattu college alli full busy idde."
        assert len(c.get(f"/api/companion/conversations/{r['conversation_id']}").json()["messages"]) == 4


def test_ollama_not_running_gives_clear_error_and_saves_nothing(tmp_path):
    c, _ = ollama_client(tmp_path, mode="down")
    with c:
        st = c.get("/api/companion/status").json()["chat"]
        assert st["reachable"] is False and st["model_available"] is False
        r = c.post("/api/companion/chat", json={"message": "hello"})
        assert r.status_code == 503 and "Ollama is not running" in r.json()["detail"]["message"]
        assert c.get("/api/companion/conversations").json() == []


def test_ollama_missing_model_gives_install_hint(tmp_path):
    c, _ = ollama_client(tmp_path, mode="missing")
    with c:
        assert c.get("/api/companion/status").json()["chat"]["model_available"] is False
        r = c.post("/api/companion/chat", json={"message": "hello"})
        assert r.status_code == 503 and "ollama pull qwen3:8b" in r.json()["detail"]["message"]
        assert r.json()["detail"]["retryable"] is False


def test_recall_uses_only_earlier_user_messages_and_skips_identical(env):
    c, fake = env
    c.post("/api/companion/chat", json={"message": "My maths exam is on Friday"})
    c.post("/api/companion/chat", json={"message": "Remind me about my maths exam"})
    system = body_of(fake.requests[-1])["systemInstruction"]["parts"][0]["text"]
    assert "user said: My maths exam is on Friday" in system
    assert "Chennagiddini maga" not in system          # the assistant's earlier reply is not recalled
    c.post("/api/companion/chat", json={"message": "My maths exam is on Friday"})
    system = body_of(fake.requests[-1])["systemInstruction"]["parts"][0]["text"]
    assert "user said: My maths exam is on Friday" not in system   # identical message is not recalled


def test_openai_compatible_server_via_base_url(tmp_path):
    """OPENAI_BASE_URL points the openai provider at another server (e.g. Groq); JSON mode falls back once."""
    calls = []

    def fake(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        calls.append((str(request.url), request.headers.get("authorization"), "response_format" in body))
        if "response_format" in body:
            return httpx.Response(400, json={"error": {"message": "response_format is not supported"}})
        reply = {"reply": "Hello there!", "emotion": "happy", "user_mood": None,
                 "reply_language": "English", "memory_suggestions": []}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(reply)}}]})

    settings = Settings(chat_provider="openai", openai_api_key="test-key", openai_chat_model="some-model",
                        openai_base_url="https://api.compatible.test/v1", database_path=str(tmp_path / "c.sqlite3"))
    with TestClient(create_app(settings, providers=Providers(settings, transport=httpx.MockTransport(fake)))) as c:
        st = c.get("/api/companion/status").json()["chat"]
        assert st["label"] == "api.compatible.test" and st["configured"] is True
        r = c.post("/api/companion/chat", json={"message": "Hi"}).json()
        assert r["reply"] == "Hello there!" and r["emotion"] == "happy"
    assert [u for u, _, _ in calls] == ["https://api.compatible.test/v1/chat/completions"] * 2
    assert [jm for _, _, jm in calls] == [True, False] and calls[0][1] == "Bearer test-key"
