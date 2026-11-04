from __future__ import annotations

import json
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, Response, UploadFile
from pydantic import BaseModel, Field

from .providers import ProviderError

router = APIRouter(prefix="/api/companion", tags=["companion"])

MAX_AUDIO_BYTES = 15 * 1024 * 1024


def svc(request: Request):
    return request.app.state.companion


def db(request: Request):
    return request.app.state.db


def provider_http_error(e: ProviderError) -> HTTPException:
    return HTTPException(status_code=e.status, detail={"message": str(e), "retryable": e.retryable})


# ------------------------------------------------------------------ models
class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = None
    input_mode: Literal["text", "voice"] = "text"
    incognito: bool = False
    history: list[HistoryItem] | None = None


class ConversationIn(BaseModel):
    title: str = Field(default="New conversation", max_length=120)


class RenameIn(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class MemoryIn(BaseModel):
    content: str = Field(min_length=2, max_length=500)
    category: str = Field(default="general", max_length=40)
    source_message_id: str | None = None


class SpeakIn(BaseModel):
    text: str = Field(min_length=1, max_length=3000)
    voice: str | None = Field(default=None, max_length=60)


# ------------------------------------------------------------------- status
@router.get("/status")
def status(request: Request):
    return {**request.app.state.providers.status(), "lid_available": svc(request).lid is not None,
            "data_sent_to_provider": [
                "System prompt (persona, language and style preferences)",
                "Up to 20 recent messages of the current conversation",
                "Your approved memories (if 'Use memories' is on)",
                "Up to 3 matching snippets from earlier conversations (if enabled)",
                "Your new message",
                "Your recorded audio, only when server speech-to-text is enabled",
                "The reply text, only when server text-to-speech is enabled"]}


# --------------------------------------------------------------------- chat
@router.post("/chat")
def chat(body: ChatIn, request: Request):
    try:
        return svc(request).chat(body.message.strip(), body.conversation_id, body.input_mode, body.incognito,
                                 [h.model_dump() for h in body.history] if body.history else None)
    except LookupError as e:
        raise HTTPException(404, str(e))
    except ProviderError as e:
        raise provider_http_error(e)


@router.post("/detect-language")
def detect_language(body: SpeakIn, request: Request):
    res = svc(request).detect(body.text)
    if res is None:
        raise HTTPException(503, "Language-identification model is not available")
    return res


# -------------------------------------------------------------------- voice
@router.post("/voice/transcribe")
async def transcribe(request: Request, audio: UploadFile = File(...), language_hint: str | None = Form(None)):
    data = await audio.read()
    if not data:
        raise HTTPException(400, "Empty audio upload")
    if len(data) > MAX_AUDIO_BYTES:
        raise HTTPException(413, "Audio too large (max 15 MB)")
    try:
        res = request.app.state.providers.transcribe(data, audio.content_type or "audio/webm", language_hint)
    except ProviderError as e:
        raise provider_http_error(e)
    res["detected_language"] = svc(request).detect(res["text"]) if res["text"] else None
    return res


@router.post("/voice/speak")
def speak(body: SpeakIn, request: Request):
    try:
        audio, mime = request.app.state.providers.speak(body.text, body.voice)
    except ProviderError as e:
        raise provider_http_error(e)
    return Response(content=audio, media_type=mime)


# ------------------------------------------------------------ conversations
@router.get("/conversations")
def list_conversations(request: Request, limit: int = Query(100, ge=1, le=500)):
    return db(request).list_conversations(limit)


@router.post("/conversations", status_code=201)
def create_conversation(body: ConversationIn, request: Request):
    return db(request).create_conversation(body.title)


@router.get("/conversations/search")
def search(request: Request, q: str = Query(..., min_length=1, max_length=200)):
    return db(request).search_messages(q, limit=50)


@router.get("/conversations/{cid}")
def get_conversation(cid: str, request: Request):
    conv = db(request).get_conversation(cid)
    if conv is None:
        raise HTTPException(404, "Conversation not found")
    return {**conv, "messages": db(request).messages(cid)}


@router.patch("/conversations/{cid}")
def rename(cid: str, body: RenameIn, request: Request):
    if not db(request).rename_conversation(cid, body.title.strip()):
        raise HTTPException(404, "Conversation not found")
    return db(request).get_conversation(cid)


@router.delete("/conversations/{cid}", status_code=204)
def delete_conversation(cid: str, request: Request):
    if not db(request).delete_conversation(cid):
        raise HTTPException(404, "Conversation not found")
    return Response(status_code=204)


@router.delete("/conversations", status_code=200)
def delete_all(request: Request, confirm: bool = Query(False)):
    if not confirm:
        raise HTTPException(400, "Pass confirm=true to delete all conversations")
    return {"deleted": db(request).delete_all_conversations()}


@router.get("/conversations/{cid}/export")
def export(cid: str, request: Request, format: Literal["json", "markdown"] = "json"):
    conv = db(request).get_conversation(cid)
    if conv is None:
        raise HTTPException(404, "Conversation not found")
    msgs = db(request).messages(cid)
    safe = "".join(c if c.isalnum() else "_" for c in conv["title"])[:40] or "conversation"
    if format == "json":
        body = json.dumps({**conv, "messages": msgs}, ensure_ascii=False, indent=2)
        return Response(body, media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="{safe}.json"'})
    lines = [f"# {conv['title']}", "", f"Created {conv['created_at']}", ""]
    for m in msgs:
        who = "You" if m["sender"] == "user" else "ECHO"
        lines += [f"**{who}** · {m['created_at']}", "", m["content"], ""]
    return Response("\n".join(lines), media_type="text/markdown",
                    headers={"Content-Disposition": f'attachment; filename="{safe}.md"'})


# -------------------------------------------------------------- preferences
@router.get("/preferences")
def get_prefs(request: Request):
    return db(request).get_preferences()


@router.post("/preferences")
def set_prefs(body: dict, request: Request):
    allowed = {
        "display_name": str, "reply_language": str, "reply_script": str, "style": str, "voice_input_lang": str,
        "voice_name": str, "speech_rate": (int, float), "auto_speak": bool, "save_history": bool,
        "use_memories": bool, "use_past_conversations": bool, "suggest_memories": bool}
    enums = {"reply_language": {"auto", "en", "kn", "hi", "ta", "ml"}, "reply_script": {"match", "romanized", "native"},
             "style": {"friendly", "concise", "detailed", "playful"}}
    clean = {}
    for k, v in body.items():
        if k not in allowed:
            raise HTTPException(422, f"Unknown preference '{k}'")
        if not isinstance(v, allowed[k]) or (allowed[k] is not bool and isinstance(v, bool)):
            raise HTTPException(422, f"Invalid type for '{k}'")
        if k in enums and v not in enums[k]:
            raise HTTPException(422, f"Invalid value for '{k}'")
        if isinstance(v, str):
            v = v[:80]
        if k == "speech_rate":
            v = max(0.5, min(2.0, float(v)))
        clean[k] = v
    return db(request).set_preferences(clean)


# ----------------------------------------------------------------- memories
@router.get("/memories")
def list_memories(request: Request):
    return db(request).list_memories()


@router.post("/memories", status_code=201)
def add_memory(body: MemoryIn, request: Request):
    from .service import is_sensitive
    if is_sensitive(body.content):
        raise HTTPException(422, "This looks like sensitive information (numbers, credentials, contact or "
                                 "financial details). ECHO does not store it as a memory.")
    return db(request).add_memory(body.content, body.category, body.source_message_id)


@router.patch("/memories/{mid}")
def update_memory(mid: str, body: MemoryIn, request: Request):
    from .service import is_sensitive
    if is_sensitive(body.content):
        raise HTTPException(422, "This looks like sensitive information; not stored.")
    m = db(request).update_memory(mid, body.content, body.category)
    if m is None:
        raise HTTPException(404, "Memory not found")
    return m


@router.delete("/memories/{mid}", status_code=204)
def delete_memory(mid: str, request: Request):
    if not db(request).delete_memory(mid):
        raise HTTPException(404, "Memory not found")
    return Response(status_code=204)


@router.delete("/memories", status_code=200)
def delete_all_memories(request: Request, confirm: bool = Query(False)):
    if not confirm:
        raise HTTPException(400, "Pass confirm=true to delete all memories")
    return {"deleted": db(request).delete_all_memories()}
