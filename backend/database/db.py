"""SQLite persistence for ECHO Companion, with numbered schema migrations.

All statements use parameter binding. Foreign keys are enforced so deleting a
conversation deletes its messages.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

MIGRATIONS: list[tuple[int, str]] = [
    (1, """
    CREATE TABLE conversations (
        id            TEXT PRIMARY KEY,
        title         TEXT NOT NULL,
        created_at    TEXT NOT NULL,
        updated_at    TEXT NOT NULL,
        language_pref TEXT,
        metadata_json TEXT NOT NULL DEFAULT '{}'
    );
    CREATE TABLE messages (
        id              TEXT PRIMARY KEY,
        conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        sender          TEXT NOT NULL CHECK (sender IN ('user', 'assistant')),
        content         TEXT NOT NULL,
        created_at      TEXT NOT NULL,
        input_mode      TEXT NOT NULL DEFAULT 'text' CHECK (input_mode IN ('text', 'voice')),
        language_json   TEXT,
        emotion         TEXT,
        metadata_json   TEXT NOT NULL DEFAULT '{}'
    );
    CREATE INDEX idx_messages_conv ON messages(conversation_id, created_at);
    CREATE TABLE preferences (
        key        TEXT PRIMARY KEY,
        value_json TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    CREATE TABLE memories (
        id                TEXT PRIMARY KEY,
        content           TEXT NOT NULL,
        category          TEXT NOT NULL DEFAULT 'general',
        source_message_id TEXT REFERENCES messages(id) ON DELETE SET NULL,
        created_at        TEXT NOT NULL,
        updated_at        TEXT NOT NULL
    );
    """),
    (2, """
    CREATE VIRTUAL TABLE messages_fts USING fts5(content, message_id UNINDEXED, conversation_id UNINDEXED,
                                                 tokenize = 'unicode61 remove_diacritics 2');
    CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN
        INSERT INTO messages_fts(content, message_id, conversation_id) VALUES (new.content, new.id, new.conversation_id);
    END;
    CREATE TRIGGER messages_ad AFTER DELETE ON messages BEGIN
        DELETE FROM messages_fts WHERE message_id = old.id;
    END;
    """),
]

DEFAULT_PREFERENCES = {
    "display_name": "",
    "reply_language": "auto",       # auto | en | kn | hi | ta | ml
    "reply_script": "match",        # match | romanized | native
    "style": "friendly",            # friendly | concise | detailed | playful
    "voice_input_lang": "en-IN",    # BCP-47 tag for speech recognition
    "voice_name": "",
    "speech_rate": 1.0,
    "auto_speak": True,
    "save_history": True,
    "use_memories": True,
    "use_past_conversations": True,
    "suggest_memories": True,
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: str):
        self.path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.migrate()

    # ------------------------------------------------------------ schema
    def migrate(self) -> None:
        with self._lock:
            self.conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY, applied_at TEXT)")
            done = {r[0] for r in self.conn.execute("SELECT version FROM schema_version")}
            for version, sql in MIGRATIONS:
                if version in done:
                    continue
                with self.conn:
                    self.conn.executescript(sql)
                    self.conn.execute("INSERT INTO schema_version VALUES (?, ?)", (version, now()))

    def schema_version(self) -> int:
        return self.conn.execute("SELECT MAX(version) FROM schema_version").fetchone()[0]

    def _q(self, sql: str, args: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def _x(self, sql: str, args: tuple = ()) -> int:
        with self._lock, self.conn:
            return self.conn.execute(sql, args).rowcount

    # ----------------------------------------------------- conversations
    def create_conversation(self, title: str = "New conversation", language_pref: str | None = None) -> dict:
        cid, ts = str(uuid.uuid4()), now()
        self._x("INSERT INTO conversations (id, title, created_at, updated_at, language_pref) VALUES (?,?,?,?,?)",
                (cid, title[:120], ts, ts, language_pref))
        return self.get_conversation(cid)

    def get_conversation(self, cid: str) -> dict | None:
        rows = self._q("""SELECT c.*, (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count
                          FROM conversations c WHERE c.id = ?""", (cid,))
        return rows[0] if rows else None

    def list_conversations(self, limit: int = 100) -> list[dict]:
        return self._q("""SELECT c.*, (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count,
                                 (SELECT content FROM messages m WHERE m.conversation_id = c.id
                                  ORDER BY created_at DESC, rowid DESC LIMIT 1) AS last_message
                          FROM conversations c ORDER BY updated_at DESC LIMIT ?""", (limit,))

    def rename_conversation(self, cid: str, title: str) -> bool:
        return self._x("UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?", (title[:120], now(), cid)) > 0

    def delete_conversation(self, cid: str) -> bool:
        return self._x("DELETE FROM conversations WHERE id = ?", (cid,)) > 0

    def delete_all_conversations(self) -> int:
        return self._x("DELETE FROM conversations")

    # ---------------------------------------------------------- messages
    def add_message(self, cid: str, sender: str, content: str, input_mode: str = "text",
                    language: dict | None = None, emotion: str | None = None, metadata: dict | None = None) -> dict:
        mid, ts = str(uuid.uuid4()), now()
        with self._lock, self.conn:
            self.conn.execute(
                """INSERT INTO messages (id, conversation_id, sender, content, created_at, input_mode,
                                         language_json, emotion, metadata_json) VALUES (?,?,?,?,?,?,?,?,?)""",
                (mid, cid, sender, content, ts, input_mode, json.dumps(language, ensure_ascii=False) if language else None,
                 emotion, json.dumps(metadata or {}, ensure_ascii=False)))
            self.conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (ts, cid))
        return self.get_message(mid)

    def get_message(self, mid: str) -> dict | None:
        rows = self._q("SELECT * FROM messages WHERE id = ?", (mid,))
        return _decode_msg(rows[0]) if rows else None

    def messages(self, cid: str, limit: int | None = None) -> list[dict]:
        if limit:
            rows = self._q("""SELECT * FROM (SELECT *, rowid AS r FROM messages WHERE conversation_id = ?
                              ORDER BY created_at DESC, rowid DESC LIMIT ?) ORDER BY created_at, r""", (cid, limit))
        else:
            rows = self._q("SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at, rowid", (cid,))
        return [_decode_msg(r) for r in rows]

    def search_messages(self, query: str, limit: int = 30, exclude_conversation: str | None = None) -> list[dict]:
        terms = [t for t in "".join(c if c.isalnum() or c.isspace() else " " for c in query).split() if len(t) > 1]
        if not terms:
            return []
        fts_query = " OR ".join(f'"{t}"' for t in terms[:12])
        rows = self._q(
            """SELECT m.id, m.conversation_id, m.sender, m.content, m.created_at, c.title,
                      bm25(messages_fts) AS score
               FROM messages_fts JOIN messages m ON m.id = messages_fts.message_id
               JOIN conversations c ON c.id = m.conversation_id
               WHERE messages_fts MATCH ? AND (? IS NULL OR m.conversation_id != ?)
               ORDER BY score LIMIT ?""", (fts_query, exclude_conversation, exclude_conversation, limit))
        return rows

    # ------------------------------------------------------- preferences
    def get_preferences(self) -> dict:
        prefs = dict(DEFAULT_PREFERENCES)
        for r in self._q("SELECT key, value_json FROM preferences"):
            prefs[r["key"]] = json.loads(r["value_json"])
        return prefs

    def set_preferences(self, values: dict) -> dict:
        ts = now()
        with self._lock, self.conn:
            for k, v in values.items():
                if k not in DEFAULT_PREFERENCES:
                    continue
                self.conn.execute("INSERT INTO preferences (key, value_json, updated_at) VALUES (?,?,?) "
                                  "ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json, "
                                  "updated_at = excluded.updated_at", (k, json.dumps(v), ts))
        return self.get_preferences()

    # ---------------------------------------------------------- memories
    def list_memories(self) -> list[dict]:
        return self._q("SELECT * FROM memories ORDER BY updated_at DESC")

    def add_memory(self, content: str, category: str = "general", source_message_id: str | None = None) -> dict:
        mid, ts = str(uuid.uuid4()), now()
        if source_message_id and not self.get_message(source_message_id):
            source_message_id = None
        self._x("INSERT INTO memories (id, content, category, source_message_id, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?)", (mid, content.strip()[:500], category[:40], source_message_id, ts, ts))
        return self._q("SELECT * FROM memories WHERE id = ?", (mid,))[0]

    def update_memory(self, mid: str, content: str, category: str | None = None) -> dict | None:
        n = self._x("UPDATE memories SET content = ?, category = COALESCE(?, category), updated_at = ? WHERE id = ?",
                    (content.strip()[:500], category, now(), mid))
        return self._q("SELECT * FROM memories WHERE id = ?", (mid,))[0] if n else None

    def delete_memory(self, mid: str) -> bool:
        return self._x("DELETE FROM memories WHERE id = ?", (mid,)) > 0

    def delete_all_memories(self) -> int:
        return self._x("DELETE FROM memories")


def _decode_msg(r: dict) -> dict:
    r = dict(r)
    r.pop("r", None)
    r["language"] = json.loads(r.pop("language_json")) if r.get("language_json") else None
    r["metadata"] = json.loads(r.pop("metadata_json") or "{}")
    return r
