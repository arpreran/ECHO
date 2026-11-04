import { Play } from 'lucide-react'
import { useEffect, useState } from 'react'
import { Badge, Button, Loading, Panel, Select, Toggle } from '../components/ui'
import { api, type Preferences } from '../lib/api'
import { browserVoices, pickVoice, speakBrowser } from '../lib/audio'
import { useCompanion } from './CompanionLayout'

const INPUT_LANGS = [
  { value: 'en-IN', label: 'English (India)' }, { value: 'kn-IN', label: 'Kannada' }, { value: 'hi-IN', label: 'Hindi' },
  { value: 'ta-IN', label: 'Tamil' }, { value: 'ml-IN', label: 'Malayalam' }, { value: 'en-US', label: 'English (US)' },
]

export function SettingsPage() {
  const { prefs, savePrefs, status } = useCompanion()
  const [voices, setVoices] = useState<SpeechSynthesisVoice[]>(browserVoices())
  const [saved, setSaved] = useState<string | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [name, setName] = useState('')
  const [testing, setTesting] = useState(false)

  useEffect(() => {
    if (typeof speechSynthesis === 'undefined') return
    const on = () => setVoices(browserVoices())
    speechSynthesis.addEventListener('voiceschanged', on)
    return () => speechSynthesis.removeEventListener('voiceschanged', on)
  }, [])
  useEffect(() => { if (prefs) setName(prefs.display_name) }, [prefs])

  if (!prefs) return <Loading label="Loading preferences…" />

  const save = async (p: Partial<Preferences>) => {
    setErr(null)
    try { await savePrefs(p); setSaved('Saved'); setTimeout(() => setSaved(null), 1500) } catch (e) { setErr((e as Error).message) }
  }
  const indianVoices = voices.filter((v) => /-(IN)$/i.test(v.lang) || /^(kn|hi|ta|ml)/.test(v.lang))
  const otherVoices = voices.filter((v) => !indianVoices.includes(v))
  const serverTTS = status && status.tts.provider !== 'browser'

  const testVoice = async () => {
    setTesting(true); setErr(null)
    const sample = 'Hello! Naanu ECHO. Hegiddira?'
    try {
      if (serverTTS) {
        const buf = await api.companion.speak(sample, prefs.voice_name || undefined)
        const ctx = new AudioContext(); const src = ctx.createBufferSource(); src.buffer = await ctx.decodeAudioData(buf); src.connect(ctx.destination); src.start()
        await new Promise((r) => (src.onended = r))
      } else {
        await speakBrowser(sample, { voice: pickVoice(prefs.voice_input_lang, prefs.voice_name), rate: prefs.speech_rate, lang: prefs.voice_input_lang, muted: false, onLevel: () => {} }).done
      }
    } catch (e) { setErr((e as Error).message) } finally { setTesting(false) }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-end justify-between">
        <div><div className="eyebrow">Companion</div><h1 className="h-display text-2xl font-semibold text-white">Voice, language & privacy</h1></div>
        {saved && <Badge tone="pos">{saved}</Badge>}
      </div>
      {err && <p className="text-sm text-neg">{err}</p>}
      <div className="grid gap-6 lg:grid-cols-2">
        <Panel title="Language">
          <div className="grid gap-4 sm:grid-cols-2">
            <Select label="Reply language" value={prefs.reply_language} onChange={(v) => void save({ reply_language: v })} options={[
              { value: 'auto', label: 'Match me (incl. code-mixing)' }, { value: 'en', label: 'English' }, { value: 'kn', label: 'Kannada' },
              { value: 'hi', label: 'Hindi' }, { value: 'ta', label: 'Tamil' }, { value: 'ml', label: 'Malayalam' }]} />
            <Select label="Script" value={prefs.reply_script} onChange={(v) => void save({ reply_script: v })} options={[
              { value: 'match', label: 'Same as mine' }, { value: 'romanized', label: 'Romanized (English letters)' }, { value: 'native', label: 'Native script' }]} />
            <Select label="Style" value={prefs.style} onChange={(v) => void save({ style: v })} options={[
              { value: 'friendly', label: 'Friendly' }, { value: 'concise', label: 'Concise' }, { value: 'detailed', label: 'Detailed' }, { value: 'playful', label: 'Playful' }]} />
            <label className="block"><span className="eyebrow mb-1.5 block">Call me</span>
              <input value={name} maxLength={80} onChange={(e) => setName(e.target.value)} onBlur={() => name !== prefs.display_name && void save({ display_name: name })}
                placeholder="Optional name" className="focus-ring h-10 w-full rounded-xl border border-line bg-ink-850 px-3 text-sm text-white" /></label>
          </div>
        </Panel>

        <Panel title="Voice" actions={<Button size="sm" icon={<Play className="size-3.5" />} loading={testing} onClick={() => void testVoice()}>Test voice</Button>}>
          <div className="grid gap-4 sm:grid-cols-2">
            <Select label="Speech recognition language" value={prefs.voice_input_lang} onChange={(v) => void save({ voice_input_lang: v })} options={INPUT_LANGS} />
            {serverTTS ? (
              <label className="block"><span className="eyebrow mb-1.5 block">Server voice name</span>
                <input defaultValue={prefs.voice_name} maxLength={60} onBlur={(e) => void save({ voice_name: e.target.value })}
                  placeholder={status?.tts.provider === 'gemini' ? 'e.g. Kore, Puck, Aoede' : 'e.g. alloy, nova, shimmer'}
                  className="focus-ring h-10 w-full rounded-xl border border-line bg-ink-850 px-3 text-sm text-white" /></label>
            ) : (
              <label className="block"><span className="eyebrow mb-1.5 block">Browser voice</span>
                <select value={prefs.voice_name} onChange={(e) => void save({ voice_name: e.target.value })} className="focus-ring h-10 w-full rounded-xl border border-line bg-ink-850 px-3 text-sm text-white">
                  <option value="">Automatic (match reply script)</option>
                  {indianVoices.length > 0 && <optgroup label="Indian languages">{indianVoices.map((v) => <option key={v.name} value={v.name}>{v.name} ({v.lang})</option>)}</optgroup>}
                  <optgroup label="Other">{otherVoices.map((v) => <option key={v.name} value={v.name}>{v.name} ({v.lang})</option>)}</optgroup>
                </select></label>
            )}
            <label className="block sm:col-span-2"><span className="eyebrow mb-1.5 block">Speech rate · {prefs.speech_rate.toFixed(2)}×</span>
              <input type="range" min={0.5} max={2} step={0.05} defaultValue={prefs.speech_rate} onChange={(e) => void save({ speech_rate: Number(e.target.value) })} className="w-full accent-cyan-400" /></label>
          </div>
          <div className="mt-2 divide-y divide-line"><Toggle checked={prefs.auto_speak} onChange={(v) => void save({ auto_speak: v })} label="Speak replies aloud" /></div>
          <p className="mt-3 text-xs text-mist">
            Browser speech recognition handles one language per session, so code-mixed speech is transcribed best with the matching Indian locale (for example Kannada for Kannada-English). {!serverTTS && indianVoices.length === 0 && 'No Indian-language browser voices are installed on this device; replies in native scripts may be read by an English voice.'}
          </p>
        </Panel>

        <Panel title="Providers (configured on the backend)">
          {!status ? <Loading /> : (
            <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-sm">
              <dt className="text-mist">Conversation</dt><dd className="text-white">{status.chat.label ?? status.chat.provider} · {status.chat.model}{' '}
                {status.chat.provider === 'ollama'
                  ? (!status.chat.reachable ? <Badge tone="warn">not running</Badge> : !status.chat.model_available ? <Badge tone="warn">model not installed</Badge> : <Badge tone="pos">local · no key</Badge>)
                  : status.chat.configured ? <Badge tone="pos">key set</Badge> : <Badge tone="warn">no key</Badge>}</dd>
              <dt className="text-mist">Speech-to-text</dt><dd className="text-white">{status.stt.provider === 'browser' ? 'Browser Web Speech API' : status.stt.provider} {!status.stt.configured && <Badge tone="warn">no key</Badge>}</dd>
              <dt className="text-mist">Text-to-speech</dt><dd className="text-white">{status.tts.provider === 'browser' ? 'Browser speechSynthesis' : `${status.tts.provider} · ${status.tts.model}`} {!status.tts.configured && <Badge tone="warn">no key</Badge>}</dd>
              <dt className="text-mist">Language tags</dt><dd className="text-white">{status.lid_available ? 'ECHO NLP Lab classical LID' : 'unavailable'}</dd>
            </dl>
          )}
          <p className="mt-3 text-xs text-mist">Providers and keys are set in <code className="font-mono">backend/.env</code>. Keys never reach the browser.</p>
        </Panel>

        <Panel title="Privacy: what is sent to the AI provider">
          <ul className="list-disc space-y-1 pl-5 text-sm text-fog">{(status?.data_sent_to_provider ?? []).map((d) => <li key={d}>{d}</li>)}</ul>
          <p className="mt-3 text-xs text-mist">Browser speech recognition is processed by your browser vendor’s service. Conversations and memories are stored locally in SQLite (<code className="font-mono">database/echo.sqlite3</code>).</p>
        </Panel>
      </div>
    </div>
  )
}
