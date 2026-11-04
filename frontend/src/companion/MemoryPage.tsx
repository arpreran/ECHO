import { Brain, Pencil, Plus, Trash2 } from 'lucide-react'
import { useState } from 'react'
import { Badge, Button, Empty, ErrorState, Loading, Panel, Toggle, useAsync } from '../components/ui'
import { api, type Memory } from '../lib/api'
import { useCompanion } from './CompanionLayout'

export function MemoryPage() {
  const { prefs, savePrefs } = useCompanion()
  const mems = useAsync(() => api.companion.memories())
  const [draft, setDraft] = useState('')
  const [editing, setEditing] = useState<string | null>(null)
  const [editText, setEditText] = useState('')
  const [err, setErr] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [confirmAll, setConfirmAll] = useState(false)

  const add = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!draft.trim()) return
    setSaving(true); setErr(null)
    try { const m = await api.companion.addMemory(draft.trim(), 'manual'); mems.setData((l) => [m, ...(l ?? [])]); setDraft('') }
    catch (e2) { setErr((e2 as Error).message) } finally { setSaving(false) }
  }
  const update = async (m: Memory) => {
    setErr(null)
    try { const u = await api.companion.updateMemory(m.id, editText.trim(), m.category); mems.setData((l) => l?.map((x) => (x.id === m.id ? u : x)) ?? null); setEditing(null) }
    catch (e) { setErr((e as Error).message) }
  }
  const remove = async (id: string) => {
    try { await api.companion.deleteMemory(id); mems.setData((l) => l?.filter((x) => x.id !== id) ?? null) } catch (e) { setErr((e as Error).message) }
  }
  const removeAll = async () => {
    try { await api.companion.deleteAllMemories(); mems.setData([]); setConfirmAll(false) } catch (e) { setErr((e as Error).message) }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
      <div className="space-y-6">
        <div><div className="eyebrow">Companion</div><h1 className="h-display text-2xl font-semibold text-white">Long-term memory</h1>
          <p className="mt-2 max-w-xl text-sm text-mist">These are the only facts ECHO remembers across conversations. Nothing is stored here without your approval. Conversation history is separate: ECHO can search it for relevant snippets, which you can turn off in the panel on the right.</p></div>
        <Panel title="Add a memory">
          <form onSubmit={add} className="flex flex-col gap-2 sm:flex-row">
            <input value={draft} onChange={(e) => setDraft(e.target.value)} maxLength={500} placeholder="e.g. Prefers replies in Kannada-English mix"
              className="focus-ring h-10 flex-1 rounded-xl border border-line bg-ink-850 px-3 text-sm text-white" />
            <Button type="submit" variant="primary" icon={<Plus className="size-4" />} loading={saving} disabled={!draft.trim()}>Remember</Button>
          </form>
          {err && <p className="mt-2 text-sm text-neg">{err}</p>}
        </Panel>
        <Panel title="Saved memories" eyebrow={mems.data ? `${mems.data.length} items` : undefined} actions={
          mems.data && mems.data.length > 0 && (confirmAll
            ? <><Button size="sm" variant="danger" onClick={() => void removeAll()}>Confirm delete all</Button><Button size="sm" onClick={() => setConfirmAll(false)}>Cancel</Button></>
            : <Button size="sm" variant="danger" icon={<Trash2 className="size-3.5" />} onClick={() => setConfirmAll(true)}>Forget everything</Button>)}>
          {mems.loading ? <Loading /> : mems.error ? <ErrorState error={mems.error} onRetry={mems.reload} /> : !mems.data?.length ? (
            <Empty title="ECHO has no long-term memories." hint="Approve a suggestion after chatting, or add one above." />
          ) : (
            <ul className="divide-y divide-line">
              {mems.data.map((m) => (
                <li key={m.id} className="flex items-start gap-3 py-3">
                  <Brain className="mt-0.5 size-4 shrink-0 text-violet-300" />
                  <div className="min-w-0 flex-1">
                    {editing === m.id ? (
                      <form onSubmit={(e) => { e.preventDefault(); void update(m) }} className="flex gap-2">
                        <input autoFocus value={editText} onChange={(e) => setEditText(e.target.value)} maxLength={500} className="focus-ring h-9 flex-1 rounded-lg border border-line bg-ink-850 px-3 text-sm text-white" />
                        <Button size="sm" type="submit" variant="primary">Save</Button><Button size="sm" type="button" onClick={() => setEditing(null)}>Cancel</Button>
                      </form>
                    ) : (
                      <>
                        <p className="text-sm text-white">{m.content}</p>
                        <p className="mt-0.5 flex items-center gap-2 text-[11px] text-mist"><Badge>{m.category}</Badge>updated {new Date(m.updated_at).toLocaleString()}</p>
                      </>
                    )}
                  </div>
                  {editing !== m.id && (
                    <div className="flex shrink-0 gap-1">
                      <Button size="sm" variant="ghost" aria-label="Edit" onClick={() => { setEditing(m.id); setEditText(m.content) }}><Pencil className="size-4" /></Button>
                      <Button size="sm" variant="ghost" aria-label="Delete" onClick={() => void remove(m.id)}><Trash2 className="size-4 text-neg" /></Button>
                    </div>
                  )}
                </li>
              ))}
            </ul>
          )}
        </Panel>
      </div>
      <div className="space-y-6">
        <Panel title="Memory controls">
          {!prefs ? <Loading /> : (
            <div className="divide-y divide-line">
              <Toggle checked={prefs.use_memories} onChange={(v) => void savePrefs({ use_memories: v })} label="Use saved memories" hint="Send the memories above to the chat model as context." />
              <Toggle checked={prefs.suggest_memories} onChange={(v) => void savePrefs({ suggest_memories: v })} label="Suggest new memories" hint="ECHO may propose facts to remember; each needs your approval." />
              <Toggle checked={prefs.use_past_conversations} onChange={(v) => void savePrefs({ use_past_conversations: v })} label="Recall earlier conversations" hint="Search saved history for up to 3 relevant snippets per message." />
              <Toggle checked={prefs.save_history} onChange={(v) => void savePrefs({ save_history: v })} label="Save conversation history" hint="When off, new messages are not written to the database." />
            </div>
          )}
        </Panel>
        <Panel title="What is never stored">
          <p className="text-sm text-mist">Memory suggestions and manual memories that look like passwords, OTPs, card, bank, phone, Aadhaar or PAN numbers, or e-mail addresses are rejected by the backend. Health and financial details are excluded from suggestions by instruction to the model.</p>
        </Panel>
      </div>
    </div>
  )
}
