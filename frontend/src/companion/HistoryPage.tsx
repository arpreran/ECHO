import { Download, MessageCircle, Pencil, Search, Trash2, X } from 'lucide-react'
import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Button, Empty, ErrorState, Loading, Panel, useAsync } from '../components/ui'
import { api, type Conversation } from '../lib/api'
import { useCompanion } from './CompanionLayout'

export function HistoryPage() {
  const { setConversationId, conversationId } = useCompanion()
  const nav = useNavigate()
  const list = useAsync(() => api.companion.conversations())
  const [q, setQ] = useState('')
  const [hits, setHits] = useState<Awaited<ReturnType<typeof api.companion.search>> | null>(null)
  const [searching, setSearching] = useState(false)
  const [searchErr, setSearchErr] = useState<string | null>(null)
  const [editing, setEditing] = useState<string | null>(null)
  const [editTitle, setEditTitle] = useState('')
  const [confirmAll, setConfirmAll] = useState(false)
  const [actionErr, setActionErr] = useState<string | null>(null)

  const open = (id: string) => { setConversationId(id); nav('/companion') }

  const search = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!q.trim()) { setHits(null); return }
    setSearching(true); setSearchErr(null)
    try { setHits(await api.companion.search(q.trim())) } catch (err) { setSearchErr((err as Error).message) } finally { setSearching(false) }
  }

  const rename = async (c: Conversation) => {
    if (!editTitle.trim()) return
    try {
      const u = await api.companion.rename(c.id, editTitle.trim())
      list.setData((l) => l?.map((x) => (x.id === c.id ? { ...x, title: u.title } : x)) ?? null)
      setEditing(null)
    } catch (e) { setActionErr((e as Error).message) }
  }

  const remove = async (id: string) => {
    try {
      await api.companion.remove(id)
      list.setData((l) => l?.filter((x) => x.id !== id) ?? null)
      if (id === conversationId) setConversationId(null)
    } catch (e) { setActionErr((e as Error).message) }
  }

  const removeAll = async () => {
    try { await api.companion.removeAll(); setConversationId(null); list.setData([]); setConfirmAll(false) } catch (e) { setActionErr((e as Error).message) }
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div><div className="eyebrow">Companion</div><h1 className="h-display text-2xl font-semibold text-white">Conversation history</h1></div>
        <form onSubmit={search} className="flex w-full gap-2 sm:w-auto">
          <div className="relative flex-1 sm:w-72">
            <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-mist" />
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search all messages…" maxLength={200}
              className="focus-ring h-10 w-full rounded-xl border border-line bg-ink-850 pl-9 pr-3 text-sm text-white" />
          </div>
          <Button type="submit" loading={searching}>Search</Button>
          {hits && <Button type="button" variant="ghost" aria-label="Clear search" onClick={() => { setHits(null); setQ('') }}><X className="size-4" /></Button>}
        </form>
      </div>
      {actionErr && <p className="text-sm text-neg">{actionErr}</p>}

      {hits && (
        <Panel title={`Search results for “${q}”`} eyebrow={`${hits.length} matching messages`}>
          {searchErr ? <ErrorState error={searchErr} /> : hits.length === 0 ? <Empty title="No messages match." /> : (
            <ul className="divide-y divide-line">
              {hits.map((h) => (
                <li key={h.id}>
                  <button onClick={() => open(h.conversation_id)} className="focus-ring w-full rounded-lg px-2 py-3 text-left hover:bg-white/[0.03]">
                    <div className="text-xs text-mist">{h.title} · {new Date(h.created_at).toLocaleString()} · {h.sender === 'user' ? 'You' : 'ECHO'}</div>
                    <div className="mt-1 line-clamp-2 text-sm text-fog">{h.content}</div>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </Panel>
      )}

      <Panel title="All conversations" actions={
        list.data && list.data.length > 0 && (confirmAll ? (
          <><span className="text-xs text-neg">Delete every conversation permanently?</span>
            <Button size="sm" variant="danger" onClick={() => void removeAll()}>Delete all</Button>
            <Button size="sm" onClick={() => setConfirmAll(false)}>Cancel</Button></>
        ) : <Button size="sm" variant="danger" icon={<Trash2 className="size-3.5" />} onClick={() => setConfirmAll(true)}>Delete all</Button>)
      }>
        {list.loading ? <Loading /> : list.error ? <ErrorState error={list.error} onRetry={list.reload} /> : !list.data?.length ? (
          <Empty title="No saved conversations yet." hint="Conversations are saved automatically while “Save history” is on in Settings." action={<Button size="sm" variant="primary" onClick={() => nav('/companion')}>Start talking</Button>} />
        ) : (
          <ul className="divide-y divide-line">
            {list.data.map((c) => (
              <li key={c.id} className="flex flex-wrap items-center gap-3 py-3">
                <div className="min-w-0 flex-1">
                  {editing === c.id ? (
                    <form onSubmit={(e) => { e.preventDefault(); void rename(c) }} className="flex gap-2">
                      <input autoFocus value={editTitle} onChange={(e) => setEditTitle(e.target.value)} maxLength={120}
                        className="focus-ring h-9 flex-1 rounded-lg border border-line bg-ink-850 px-3 text-sm text-white" />
                      <Button size="sm" type="submit" variant="primary">Save</Button>
                      <Button size="sm" type="button" onClick={() => setEditing(null)}>Cancel</Button>
                    </form>
                  ) : (
                    <button onClick={() => open(c.id)} className="focus-ring block w-full rounded-lg text-left">
                      <div className="flex items-center gap-2 text-sm font-medium text-white">{c.title}{c.id === conversationId && <span className="text-[10px] text-cyan-soft">active</span>}</div>
                      <div className="truncate text-xs text-mist">{c.message_count} messages · {new Date(c.updated_at).toLocaleString()}{c.last_message ? ` · ${c.last_message}` : ''}</div>
                    </button>
                  )}
                </div>
                {editing !== c.id && (
                  <div className="flex shrink-0 gap-1">
                    <Button size="sm" variant="ghost" aria-label="Continue" title="Continue" onClick={() => open(c.id)}><MessageCircle className="size-4" /></Button>
                    <Button size="sm" variant="ghost" aria-label="Rename" title="Rename" onClick={() => { setEditing(c.id); setEditTitle(c.title) }}><Pencil className="size-4" /></Button>
                    <a className="focus-ring inline-flex h-8 items-center rounded-xl px-2 text-xs text-fog hover:bg-white/5" href={api.companion.exportUrl(c.id, 'markdown')} title="Export Markdown"><Download className="mr-1 size-4" />MD</a>
                    <a className="focus-ring inline-flex h-8 items-center rounded-xl px-2 text-xs text-fog hover:bg-white/5" href={api.companion.exportUrl(c.id, 'json')} title="Export JSON">JSON</a>
                    <Button size="sm" variant="ghost" aria-label="Delete" title="Delete" onClick={() => void remove(c.id)}><Trash2 className="size-4 text-neg" /></Button>
                  </div>
                )}
              </li>
            ))}
          </ul>
        )}
      </Panel>
    </div>
  )
}
