"""ECHO Companion conversation logic.

What is sent to the external chat provider on each turn (also shown in the UI):
  * the system prompt (persona, language preference, style),
  * up to RECENT_TURNS recent messages of the current conversation,
  * the user's approved memories (if enabled),
  * up to 3 short snippets from earlier conversations that match the message
    (if enabled),
  * the new message.
Nothing else (no device data, no API keys) is sent.
"""
from __future__ import annotations

import re

from database.db import Database

from .providers import ChatTurn, ProviderError, Providers

RECENT_TURNS = 20
EMOTIONS = {"neutral", "happy", "supportive", "curious"}
LANG_NAMES = {"en": "English", "kn": "Kannada", "hi": "Hindi", "ta": "Tamil", "ml": "Malayalam"}

# Never suggest storing these, even if the model proposes it.
SENSITIVE_PATTERNS = [
    re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{1,4}\b"),        # card-like numbers
    re.compile(r"\b\d{10,}\b"),                                       # phone / account numbers
    re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),                           # PAN
    re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),                         # Aadhaar-like
    re.compile(r"password|passcode|otp|pin\b|cvv|ifsc|aadhaar|aadhar|pan card|bank account", re.I),
    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),                           # e-mail address
]


def is_sensitive(text: str) -> bool:
    return any(p.search(text) for p in SENSITIVE_PATTERNS)


def build_system_prompt(prefs: dict, memories: list[dict], recalled: list[dict], detected: dict | None) -> str:
    lang = prefs.get("reply_language", "auto")
    script = prefs.get("reply_script", "match")
    if lang == "auto":
        lang_rule = ("Reply in the same language mix the user is using. If they code-mix (for example Kannada "
                     "and English in one sentence, written in English letters), code-mix back naturally in the "
                     "same proportion. If they switch languages, switch with them.")
    else:
        lang_rule = f"Reply mainly in {LANG_NAMES.get(lang, lang)}; casual English words are fine where natural."
    script_rule = {
        "match": "Use the same script the user uses (Romanized text gets Romanized replies; native script gets native script).",
        "romanized": "Write Indian languages in Romanized (English) letters.",
        "native": "Write Indian languages in their native script (ಕನ್ನಡ, हिन्दी, தமிழ், മലയാളം).",
    }.get(script, "")
    style = {
        "friendly": "Warm, friendly and natural, like a real conversation; concise by default (usually one to "
                    "three sentences) unless the user asks for more.",
        "concise": "Brief and direct: one to three sentences.",
        "detailed": "Thorough and explanatory when the topic needs it.",
        "playful": "Light, playful and humorous where appropriate, never mocking.",
    }.get(prefs.get("style", "friendly"), "")
    name = prefs.get("display_name") or ""

    parts = [
        "You are ECHO, an AI companion. You are an AI, not a human: never claim human experiences, a body, "
        "feelings you cannot have, or memories you were not given. Be supportive, curious and honest.",
        # No example phrases here: in testing, qwen3:8b copied them into replies or echoed the user verbatim.
        "You converse in English, Kannada, Hindi, Tamil and Malayalam as well as you can, including Romanized "
        "spellings (Indian languages typed in English letters) and mixing languages within one sentence. "
        "Always write your own reply; never repeat the user's message back.",
        "If you are not sure what a word or phrase means, say so briefly or ask, instead of guessing or inventing "
        "a translation. Never claim fluency you do not have in a particular phrase.",
        lang_rule, script_rule, f"Style: {style}",
        "Ask a natural follow-up question when it helps the conversation, but not every turn. Avoid repeating "
        "phrases you used recently. Notice the user's mood (frustration, excitement, sadness, happiness) and "
        "respond to it. Replies will often be read aloud: avoid markdown, lists and emoji unless asked.",
        "If the user is in crisis or mentions self-harm, respond with care and encourage contacting local "
        "emergency services or a trusted person (India: Tele-MANAS 14416).",
    ]
    if name:
        parts.append(f"The user likes to be called {name}.")
    if detected and detected.get("summary"):
        parts.append(f"A classical language-ID model tagged the latest message as: {detected['summary']}. "
                     "It can be wrong on short messages; trust the text itself.")
    if memories:
        parts.append("Facts the user explicitly approved for you to remember (use only when relevant; you may "
                     "say you remember them):\n" + "\n".join(f"- {m['content']}" for m in memories))
    else:
        parts.append("You have no saved long-term memories about this user. Do not pretend to remember things "
                     "outside this conversation.")
    if recalled:
        parts.append("Things the user said in EARLIER, separate conversations (background only; refer to them only "
                     "if clearly relevant, and never copy them as your reply):\n" + "\n".join(
                         f"- [{r['created_at'][:10]}] user said: {r['content'][:200]}" for r in recalled))
    parts.append(
        'Respond ONLY with a JSON object: {"reply": string, "emotion": one of "neutral","happy","supportive",'
        '"curious" (the tone of YOUR reply), "user_mood": short string or null, "reply_language": short '
        'description such as "Kannada-English (Romanized)", "memory_suggestions": array of at most 2 short '
        'third-person facts worth remembering long-term that the user stated about themselves in their LATEST '
        'message (preferences, goals, names of pets, etc.). Never suggest health, financial, identity-number, '
        'password, address or other sensitive details. Use [] when nothing qualifies.}')
    return "\n\n".join(p for p in parts if p)


def summarize_detection(lid_result: dict | None) -> dict | None:
    if not lid_result:
        return None
    counts = lid_result.get("word_language_counts", {})
    names = {"en": "English", "kn_rom": "Romanized Kannada", "ta_rom": "Romanized Tamil",
             "ml_rom": "Romanized Malayalam", "hi_rom": "Romanized Hindi", "kn": "Kannada script",
             "ta": "Tamil script", "ml": "Malayalam script", "hi": "Devanagari"}
    langs = [names[k] for k, _ in sorted(counts.items(), key=lambda kv: -kv[1]) if k in names]
    return {
        "primary": lid_result["primary_language"],
        "primary_name": lid_result["primary_language_name"],
        "confidence": lid_result["confidence"],
        "uncertain": lid_result["uncertain"],
        "languages": langs,
        "code_mixed": len({k.replace("_rom", "") for k in counts if k in names}) > 1,
        "text_type": lid_result["script_profile"]["text_type"],
        "summary": " + ".join(langs) if langs else None,
        "model": "ECHO NLP Lab classical LID (char n-gram TF-IDF + Logistic Regression)",
    }


class CompanionService:
    def __init__(self, db: Database, providers: Providers, lid=None):
        self.db = db
        self.providers = providers
        self.lid = lid

    def detect(self, text: str) -> dict | None:
        if self.lid is None:
            return None
        try:
            return summarize_detection(self.lid.identify(text))
        except Exception:
            return None

    def chat(self, message: str, conversation_id: str | None, input_mode: str = "text",
             incognito: bool = False, history: list[dict] | None = None) -> dict:
        prefs = self.db.get_preferences()
        persist = prefs.get("save_history", True) and not incognito
        detected = self.detect(message)

        conv = None
        if persist:
            if conversation_id:
                conv = self.db.get_conversation(conversation_id)
                if conv is None:
                    raise LookupError("Conversation not found")
            else:
                conv = self.db.create_conversation(title=_title_from(message))
            past = self.db.messages(conv["id"], limit=RECENT_TURNS)
            turns = [ChatTurn(m["sender"], m["content"]) for m in past]
        else:
            turns = [ChatTurn(h["role"], h["content"]) for h in (history or [])[-RECENT_TURNS:]
                     if h.get("role") in ("user", "assistant") and h.get("content")]

        memories = self.db.list_memories() if prefs.get("use_memories", True) else []
        recalled = []
        if prefs.get("use_past_conversations", True):
            # Only the user's own earlier words, and never a copy of the current message: in testing, small
            # models copied recalled assistant replies (or the identical earlier message) instead of answering.
            current = " ".join(message.lower().split())
            hits = self.db.search_messages(message, limit=12, exclude_conversation=conv["id"] if conv else None)
            recalled = [h for h in hits if h["sender"] == "user" and " ".join(h["content"].lower().split()) != current][:3]

        system = build_system_prompt(prefs, memories, recalled, detected)
        turns.append(ChatTurn("user", message))
        # Providers require alternating roles starting with the user.
        turns = _normalise_turns(turns)

        # Raises ProviderError; nothing is saved for a failed turn so a retry does not duplicate it.
        # A conversation created for this turn is removed again if it is still empty.
        try:
            out = self.providers.chat_json(system, turns)
        except ProviderError:
            if persist and not conversation_id:
                self.db.delete_conversation(conv["id"])
            raise

        user_msg = None
        if persist:
            user_msg = self.db.add_message(conv["id"], "user", message, input_mode=input_mode, language=detected)

        reply = str(out.get("reply") or "").strip() or "Sorry, I didn't get a reply from the model. Could you say that again?"
        emotion = out.get("emotion") if out.get("emotion") in EMOTIONS else "neutral"
        suggestions = []
        if prefs.get("suggest_memories", True):
            for s in (out.get("memory_suggestions") or [])[:2]:
                if isinstance(s, str) and 3 < len(s) < 200 and not is_sensitive(s):
                    suggestions.append(s.strip())

        meta = {"provider": self.providers.s.chat_provider, "model": self.providers.chat_model(),
                "user_mood": out.get("user_mood"), "reply_language": out.get("reply_language"),
                "memories_used": len(memories), "past_snippets_used": len(recalled),
                "unstructured_reply": bool(out.get("_unstructured"))}
        assistant_msg = None
        if persist:
            assistant_msg = self.db.add_message(conv["id"], "assistant", reply, emotion=emotion, metadata=meta)
        return {
            "conversation_id": conv["id"] if conv else None,
            "conversation_title": conv["title"] if conv else None,
            "persisted": persist,
            "user_message": user_msg,
            "assistant_message": assistant_msg,
            "reply": reply,
            "emotion": emotion,
            "detected_language": detected,
            "memory_suggestions": suggestions,
            "context": {"recent_turns": len(turns) - 1, "memories": len(memories),
                        "past_snippets": [{"conversation_id": r["conversation_id"], "title": r["title"],
                                           "created_at": r["created_at"]} for r in recalled]},
            "meta": meta,
        }


def _title_from(message: str) -> str:
    words = message.strip().split()
    t = " ".join(words[:7])
    return (t + ("…" if len(words) > 7 else "")) or "New conversation"


def _normalise_turns(turns: list[ChatTurn]) -> list[ChatTurn]:
    out: list[ChatTurn] = []
    for t in turns:
        if out and out[-1].role == t.role:
            out[-1] = ChatTurn(t.role, out[-1].content + "\n" + t.content)
        else:
            out.append(t)
    while out and out[0].role != "user":
        out.pop(0)
    return out
