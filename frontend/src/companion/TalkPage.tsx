import { AnimatePresence, motion } from 'framer-motion'
import { AlertTriangle, BookmarkPlus, Check, EyeOff, Hand, Mic, MicOff, Plus, Send, Square, Volume2, VolumeX, X } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { Orb, Waveform } from '../components/Orb'
import { Badge, Button, cx, Toggle } from '../components/ui'
import { api, type ChatResponse, type Message } from '../lib/api'
import { brokenSpeechRecognitionBrowser, speechRecognitionCtor } from '../lib/audio'
import { useCompanion } from './CompanionLayout'
import { useVoiceLoop } from './useVoiceLoop'

const STATE_LABEL: Record<string, string> = {
  idle: 'Idle', listening: 'Listening', processing: 'Thinking', speaking: 'Speaking', happy: 'Happy', supportive: 'Supportive',
  curious: 'Curious', interrupted: 'Interrupted', error: 'Error',
}

interface Suggestion { text: string; sourceId: string | null; state: 'pending' | 'saved' | 'dismissed' | 'error'; error?: string }

export function TalkPage() {
  const { prefs, status, online, conversationId, setConversationId } = useCompanion()
  const [messages, setMessages] = useState<Message[]>([])
  const [title, setTitle] = useState<string | null>(null)
  const [draft, setDraft] = useState('')
  const [incognito, setIncognito] = useState(false)
  const [suggestions, setSuggestions] = useState<Suggestion[]>([])
  const [lastContext, setLastContext] = useState<ChatResponse['context'] | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const scroller = useRef<HTMLDivElement>(null)

  const voice = useVoiceLoop({
    prefs, status, conversationId, setConversationId, incognito,
    history: () => messages.map((m) => ({ role: m.sender, content: m.content })),
    onReply: (r, userText, mode) => {
      const now = new Date().toISOString()
      const user: Message = r.user_message ?? { id: `local-u-${now}`, conversation_id: '', sender: 'user', content: userText, created_at: now, input_mode: mode, language: r.detected_language, emotion: null, metadata: {} }
      const bot: Message = r.assistant_message ?? { id: `local-a-${now}`, conversation_id: '', sender: 'assistant', content: r.reply, created_at: now, input_mode: 'text', language: null, emotion: r.emotion, metadata: r.meta }
      setMessages((m) => [...m, user, bot])
      if (r.conversation_title) setTitle(r.conversation_title)
      setLastContext(r.context)
      if (r.memory_suggestions.length)
        setSuggestions((s) => [...r.memory_suggestions.map((t) => ({ text: t, sourceId: r.user_message?.id ?? null, state: 'pending' as const })), ...s].slice(0, 4))
    },
  })

  // Load the active conversation (persisted across reloads).
  useEffect(() => {
    if (!conversationId || incognito) return
    if (messages.some((m) => m.conversation_id === conversationId)) return
    setLoadError(null)
    api.companion.conversation(conversationId)
      .then((c) => { setMessages(c.messages); setTitle(c.title) })
      .catch((e) => { if (e.status === 404) setConversationId(null); else setLoadError(e.message) })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [conversationId, incognito])

  useEffect(() => { scroller.current?.scrollTo({ top: scroller.current.scrollHeight, behavior: 'smooth' }) }, [messages, voice.interim])

  const newConversation = () => {
    voice.stopSpeaking()
    setConversationId(null); setMessages([]); setTitle(null); setSuggestions([]); setLastContext(null)
  }
  const toggleIncognito = (v: boolean) => { newConversation(); setIncognito(v) }

  const submit = (e?: React.FormEvent) => {
    e?.preventDefault()
    if (!draft.trim() || voice.mode === 'processing') return
    const t = draft
    setDraft('')
    void voice.send(t, 'text')
  }

  const saveSuggestion = async (i: number) => {
    const s = suggestions[i]
    try {
      await api.companion.addMemory(s.text, 'conversation', s.sourceId)
      setSuggestions((all) => all.map((x, j) => (j === i ? { ...x, state: 'saved' } : x)))
    } catch (e) {
      setSuggestions((all) => all.map((x, j) => (j === i ? { ...x, state: 'error', error: (e as Error).message } : x)))
    }
  }

  const busy = voice.mode === 'processing'
  const listening = voice.mode === 'listening'
  const speaking = voice.mode === 'speaking'
  const chatReady = online !== false && (status?.chat.configured ?? true)
  const browserSTT = !status || status.stt.provider === 'browser'
  const sttAvailable = !browserSTT || !!speechRecognitionCtor()
  const brokenBrowser = browserSTT ? brokenSpeechRecognitionBrowser() : null

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.05fr)]">
      {/* ---------------------------------------------------------- orb column */}
      <section className="glass relative flex flex-col items-center rounded-3xl px-4 pb-6 pt-4">
        <div className="flex w-full flex-wrap items-center justify-between gap-2">
          <Badge tone={voice.mode === 'error' ? 'neg' : voice.mode === 'idle' ? 'neutral' : 'cyan'}>
            <span className="size-1.5 rounded-full bg-current" />{STATE_LABEL[voice.orbState]}
          </Badge>
          {incognito && <Badge tone="violet"><EyeOff className="size-3" />Incognito · not saved</Badge>}
        </div>

        <div className="relative my-2 grid place-items-center">
          <button onClick={() => (speaking ? voice.interrupt() : listening ? voice.stopListening() : voice.mode !== 'processing' && sttAvailable && chatReady && voice.listen())}
            className="focus-ring rounded-full" aria-label={speaking ? 'Interrupt ECHO' : listening ? 'Stop listening' : 'Start talking'}>
            <Orb state={voice.orbState} levelRef={voice.level} size={300} />
          </button>
        </div>
        <Waveform levelRef={voice.level} active={listening || speaking} color={listening ? '#67e8f9' : '#a78bfa'} />

        <div className="mt-3 min-h-12 max-w-md text-center text-sm">
          {listening && <p className="text-cyan-soft">{voice.interim || (browserSTT ? `Listening (${prefs?.voice_input_lang || 'en-IN'})…` : 'Listening… stop talking to send')}</p>}
          {busy && <p className="text-violet-300">{voice.interim ? `“${voice.interim}”` : 'ECHO is thinking…'}</p>}
          {voice.mode === 'idle' && <p className="text-mist">{chatReady ? 'Tap the orb or the mic to talk. Tap again while ECHO speaks to interrupt.' : ''}</p>}
        </div>

        {/* controls */}
        <div className="mt-2 flex flex-wrap items-center justify-center gap-2">
          {listening ? (
            <Button variant="primary" size="lg" icon={<MicOff className="size-5" />} onClick={voice.stopListening}>Stop</Button>
          ) : (
            <Button variant="primary" size="lg" icon={<Mic className="size-5" />} disabled={busy || speaking || !chatReady || !sttAvailable}
              onClick={() => void voice.listen()}>Talk</Button>
          )}
          <Button icon={<Hand className="size-4" />} disabled={!speaking} onClick={voice.interrupt} title="Stop ECHO and start listening">Interrupt</Button>
          <Button icon={<Square className="size-4" />} disabled={!speaking} onClick={voice.stopSpeaking}>Stop speaking</Button>
          <Button icon={voice.muted ? <VolumeX className="size-4" /> : <Volume2 className="size-4" />} onClick={voice.toggleMute} aria-pressed={voice.muted}>
            {voice.muted ? 'Unmute' : 'Mute'}
          </Button>
        </div>
        <div className="mt-4 w-full max-w-sm">
          <Toggle checked={voice.handsFree} onChange={voice.setHandsFree} label="Hands-free" hint="After ECHO finishes speaking a voice reply, listen again automatically." />
        </div>

        {!sttAvailable && <p className="mt-3 max-w-sm text-center text-xs text-warn">This browser has no speech recognition. Use Chrome or Edge, or set ECHO_STT_PROVIDER in backend/.env. Text chat still works.</p>}
        {sttAvailable && brokenBrowser && <p className="mt-3 max-w-sm text-center text-xs text-warn">Voice input does not work in {brokenBrowser}: its built-in speech recognition returns no words. Open http://localhost:5173 in Google Chrome or Microsoft Edge to talk. Typing works here.</p>}
        {status && !status.chat.configured && (
          <div className="mt-4 w-full rounded-xl border border-warn/30 bg-warn/5 p-3 text-xs text-fog">
            <p className="font-medium text-warn">Conversational API not configured</p>
            <p className="mt-1">Set <code className="font-mono text-cyan-soft">{status.chat.provider === 'gemini' ? 'GEMINI_API_KEY' : status.chat.provider === 'openai' ? 'OPENAI_API_KEY' : 'ANTHROPIC_API_KEY'}</code> in <code className="font-mono">backend/.env</code> and restart the backend.</p>
          </div>
        )}
        {status?.chat.provider === 'ollama' && (!status.chat.reachable || !status.chat.model_available) && (
          <div className="mt-4 w-full rounded-xl border border-warn/30 bg-warn/5 p-3 text-xs text-fog">
            <p className="font-medium text-warn">{!status.chat.reachable ? 'Ollama is not running' : `Model ${status.chat.model} is not installed`}</p>
            <p className="mt-1">{!status.chat.reachable
              ? 'Start the Ollama app (or run ollama serve), then reload this page.'
              : <>Install it with <code className="font-mono text-cyan-soft">ollama pull {status.chat.model}</code> or change <code className="font-mono">OLLAMA_MODEL</code> in <code className="font-mono">backend/.env</code>.</>}</p>
          </div>
        )}
        <AnimatePresence>
          {voice.error && (
            <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }} role="alert"
              className="mt-4 w-full rounded-xl border border-neg/30 bg-neg/5 p-3 text-sm">
              <div className="flex items-start gap-2"><AlertTriangle className="mt-0.5 size-4 shrink-0 text-neg" /><p className="text-fog">{voice.error.message}</p></div>
              <div className="mt-3 flex gap-2">
                {voice.error.retry && <Button size="sm" variant="primary" onClick={voice.error.retry}>Try again</Button>}
                <Button size="sm" onClick={voice.dismissError}>Dismiss</Button>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </section>

      {/* ----------------------------------------------------- transcript column */}
      <section className="panel flex min-h-[560px] flex-col rounded-3xl lg:h-[calc(100vh-140px)]">
        <header className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-5 py-3">
          <div className="min-w-0">
            <div className="eyebrow">{incognito ? 'Incognito conversation' : 'Conversation'}</div>
            <h2 className="h-display truncate text-sm font-semibold text-white">{title || 'New conversation'}</h2>
          </div>
          <div className="flex items-center gap-2">
            <Button size="sm" variant={incognito ? 'primary' : 'subtle'} icon={<EyeOff className="size-3.5" />} onClick={() => toggleIncognito(!incognito)} aria-pressed={incognito}>Incognito</Button>
            <Button size="sm" icon={<Plus className="size-3.5" />} onClick={newConversation}>New</Button>
          </div>
        </header>

        <div ref={scroller} className="scroll-thin flex-1 space-y-4 overflow-y-auto px-5 py-4" aria-live="polite">
          {loadError && <p className="text-sm text-neg">{loadError}</p>}
          {messages.length === 0 && !loadError && (
            <div className="py-10 text-center">
              <p className="text-sm text-white">Say hello in any language you like.</p>
              <div className="mt-4 flex flex-wrap justify-center gap-2">
                {['Yeno maga, hegiddiya?', 'Aaj college jaana hai but mood nahi hai.', 'Neenga enna panreenga?', 'ನಾನು today ತುಂಬಾ busy ಇದ್ದೀನಿ.'].map((s) => (
                  <button key={s} disabled={!chatReady || busy} onClick={() => void voice.send(s, 'text')}
                    className="focus-ring rounded-full border border-line px-3 py-1.5 text-xs text-fog hover:border-cyan-glow/40 hover:text-white disabled:opacity-40">{s}</button>
                ))}
              </div>
            </div>
          )}
          {messages.map((m) => <Bubble key={m.id} m={m} />)}
          {listening && voice.interim && <div className="ml-auto max-w-[85%] rounded-2xl border border-dashed border-cyan-glow/30 px-4 py-2.5 text-sm text-cyan-soft/80">{voice.interim}</div>}
        </div>

        {suggestions.some((s) => s.state !== 'dismissed') && (
          <div className="border-t border-line px-5 py-3">
            <div className="eyebrow mb-2">Remember this? Only saved if you approve.</div>
            <div className="space-y-2">
              {suggestions.map((s, i) => s.state !== 'dismissed' && (
                <div key={i} className="flex items-center justify-between gap-2 rounded-xl bg-white/[0.03] px-3 py-2 text-sm">
                  <span className="min-w-0 truncate text-fog">{s.text}</span>
                  {s.state === 'saved' ? <Badge tone="pos"><Check className="size-3" />Saved</Badge> : (
                    <span className="flex shrink-0 gap-1">
                      {s.state === 'error' && <span className="text-xs text-neg" title={s.error}>Failed</span>}
                      <Button size="sm" icon={<BookmarkPlus className="size-3.5" />} onClick={() => void saveSuggestion(i)}>Save</Button>
                      <Button size="sm" variant="ghost" aria-label="Dismiss" onClick={() => setSuggestions((a) => a.map((x, j) => (j === i ? { ...x, state: 'dismissed' } : x)))}><X className="size-3.5" /></Button>
                    </span>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        <form onSubmit={submit} className="border-t border-line p-3">
          <div className="flex items-end gap-2">
            <textarea value={draft} onChange={(e) => setDraft(e.target.value)} rows={1} maxLength={4000}
              onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit() } }}
              placeholder={chatReady ? 'Type a message — any language, any script…' : 'Chat API not configured'}
              disabled={!chatReady}
              className="focus-ring max-h-40 min-h-11 flex-1 resize-none rounded-xl border border-line bg-ink-850 px-4 py-2.5 text-sm text-white placeholder:text-mist/70" />
            <Button type="submit" variant="primary" icon={<Send className="size-4" />} disabled={!draft.trim() || busy || !chatReady} loading={busy} aria-label="Send" />
          </div>
          {lastContext && (
            <p className="mt-2 text-[11px] text-mist">
              Context sent: {lastContext.recent_turns} recent messages · {lastContext.memories} memories · {lastContext.past_snippets.length} earlier-conversation snippets.{' '}
              <Link to="/companion/settings" className="underline hover:text-white">Privacy</Link>
            </p>
          )}
        </form>
      </section>
    </div>
  )
}

function Bubble({ m }: { m: Message }) {
  const mine = m.sender === 'user'
  const lang = m.language
  return (
    <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className={cx('flex flex-col', mine ? 'items-end' : 'items-start')}>
      <div className={cx('max-w-[85%] whitespace-pre-wrap rounded-2xl px-4 py-2.5 text-[15px] leading-relaxed',
        mine ? 'rounded-br-md bg-cyan-glow/12 text-white border border-cyan-glow/20' : 'rounded-bl-md bg-white/[0.04] text-fog border border-line')}>
        {m.content}
      </div>
      <div className="mt-1 flex flex-wrap items-center gap-1.5 px-1 text-[10px] text-mist">
        {m.input_mode === 'voice' && <span className="inline-flex items-center gap-0.5"><Mic className="size-3" />voice</span>}
        {mine && lang?.summary && (
          <span title={`${lang.model}${lang.confidence != null ? ` · confidence ${(lang.confidence * 100).toFixed(0)}%` : ''}`}>
            {lang.summary}{lang.code_mixed ? ' · code-mixed' : ''}{lang.uncertain ? ' · uncertain' : ''}
          </span>
        )}
        {!mine && m.emotion && m.emotion !== 'neutral' && <span>{m.emotion}</span>}
        {!mine && typeof m.metadata?.reply_language === 'string' && <span>· {m.metadata.reply_language as string}</span>}
      </div>
    </motion.div>
  )
}
