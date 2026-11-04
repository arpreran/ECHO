import { useCallback, useEffect, useRef, useState } from 'react'
import type { OrbState } from '../components/Orb'
import { api, ApiError, type ChatResponse, type Emotion, type Message, type Preferences, type ProviderStatus } from '../lib/api'
import { AudioPlayback, classifyMicError, MicSession, micErrorText, pickVoice, recognizeOnce, speakBrowser } from '../lib/audio'

export type Mode = 'idle' | 'listening' | 'processing' | 'speaking' | 'interrupted' | 'error'

// Peak mic level (0-1, see lib/audio rmsLoop) below which we assume the mic captured no sound.
const NO_SOUND_LEVEL = 0.03

/** BCP-47 tag for speaking a reply, chosen from the script actually used in it. */
export function replyLangTag(text: string, fallback: string): string {
  if (/[ಀ-೿]/.test(text)) return 'kn-IN'
  if (/[஀-௿]/.test(text)) return 'ta-IN'
  if (/[ഀ-ൿ]/.test(text)) return 'ml-IN'
  if (/[ऀ-ॿ]/.test(text)) return 'hi-IN'
  return fallback.startsWith('en') ? fallback : 'en-IN'
}

interface Opts {
  prefs: Preferences | null
  status: ProviderStatus | null
  conversationId: string | null
  setConversationId: (id: string | null) => void
  incognito: boolean
  onReply: (r: ChatResponse, userText: string, mode: 'text' | 'voice') => void
  history: () => { role: string; content: string }[]
}

export function useVoiceLoop(o: Opts) {
  const [mode, setMode] = useState<Mode>('idle')
  const [emotion, setEmotion] = useState<Emotion>('neutral')
  const [error, setError] = useState<{ message: string; retry?: () => void } | null>(null)
  const [interim, setInterim] = useState('')
  const [muted, setMuted] = useState(false)
  const [handsFree, setHandsFree] = useState(false)
  const level = useRef(0)
  const mic = useRef<MicSession | null>(null)
  const playback = useRef(new AudioPlayback())
  const stopSpeech = useRef<(() => void) | null>(null)
  const abortListen = useRef<(() => void) | null>(null)
  const cancelled = useRef(false)
  const emotionTimer = useRef<number | undefined>(undefined)
  const optsRef = useRef(o)
  optsRef.current = o
  const handsFreeRef = useRef(handsFree)
  handsFreeRef.current = handsFree
  const mutedRef = useRef(muted)
  mutedRef.current = muted
  const modeRef = useRef(mode)
  modeRef.current = mode

  const releaseMic = () => { mic.current?.stop(); mic.current = null; level.current = 0 }

  const fail = useCallback((message: string, retry?: () => void) => {
    releaseMic()
    setMode('error')
    setError({ message, retry })
  }, [])

  const speak = useCallback(async (text: string, emo: Emotion): Promise<'ended' | 'stopped'> => {
    const { prefs, status } = optsRef.current
    setEmotion(emo)
    setMode('speaking')
    const onLevel = (l: number) => { level.current = l }
    try {
      if (status?.tts.provider && status.tts.provider !== 'browser' && status.tts.configured) {
        const audio = await api.companion.speak(text, prefs?.voice_name || undefined)
        if (modeRef.current !== 'speaking') return 'stopped'
        stopSpeech.current = () => playback.current.stop()
        return await playback.current.play(audio, onLevel, mutedRef.current)
      }
      if (typeof speechSynthesis === 'undefined') throw new Error('This browser cannot synthesise speech.')
      const tag = replyLangTag(text, prefs?.voice_input_lang || 'en-IN')
      const voice = pickVoice(tag, prefs?.voice_name || undefined)
      // pickVoice falls back to an English voice, which cannot read Indic script: say so instead.
      if (!tag.startsWith('en') && !voice?.lang.startsWith(tag.split('-')[0])) {
        throw new Error(`this browser has no ${tag} voice, so the reply is shown as text only. Microsoft Edge provides voices for Kannada, Hindi, Tamil and Malayalam: open ECHO in Edge to hear them.`)
      }
      const s = speakBrowser(text, { voice, rate: prefs?.speech_rate || 1, lang: tag, muted: mutedRef.current, onLevel })
      stopSpeech.current = s.stop
      return await s.done
    } finally {
      stopSpeech.current = null
      level.current = 0
    }
  }, [])

  const settleAfterReply = useCallback((emo: Emotion) => {
    setMode('idle')
    setEmotion(emo)
    window.clearTimeout(emotionTimer.current)
    emotionTimer.current = window.setTimeout(() => setEmotion('neutral'), 4000)
  }, [])

  // eslint-disable-next-line prefer-const
  let listen: () => Promise<void>

  const send = useCallback(async (text: string, inputMode: 'text' | 'voice') => {
    const msg = text.trim()
    if (!msg) { setMode('idle'); return }
    const { conversationId, incognito, prefs } = optsRef.current
    setError(null)
    setMode('processing')
    let res: ChatResponse
    try {
      res = await api.companion.chat({ message: msg, conversation_id: incognito ? null : conversationId, input_mode: inputMode, incognito,
        history: incognito ? optsRef.current.history() : undefined })
    } catch (e) {
      const err = e as ApiError
      fail(err.message, err.retryable ? () => void send(msg, inputMode) : undefined)
      return
    }
    if (res.conversation_id && res.conversation_id !== conversationId) optsRef.current.setConversationId(res.conversation_id)
    optsRef.current.onReply(res, msg, inputMode)
    const shouldSpeak = (prefs?.auto_speak ?? true) && !mutedRef.current
    if (!shouldSpeak) { settleAfterReply(res.emotion); return }
    let outcome: 'ended' | 'stopped' = 'stopped'
    try {
      outcome = await speak(res.reply, res.emotion)
    } catch (e) {
      fail(`Could not play the reply: ${(e as Error).message}`)
      return
    }
    if (outcome === 'stopped') return // interrupt()/stopSpeaking() already set the next state
    if (handsFreeRef.current && inputMode === 'voice') void listen()
    else settleAfterReply(res.emotion)
  }, [fail, speak, settleAfterReply])

  listen = useCallback(async () => {
    const { prefs, status } = optsRef.current
    setError(null)
    setInterim('')
    releaseMic()
    cancelled.current = false
    const m = new MicSession()
    mic.current = m
    let peak = 0   // loudest mic level seen, to tell "no sound" apart from "sound but no text"
    try {
      await m.start((l) => { level.current = l; if (l > peak) peak = l })
    } catch (e) {
      mic.current = null
      fail(micErrorText[classifyMicError(e)], () => void listen())
      return
    }
    setMode('listening')
    const lang = prefs?.voice_input_lang || 'en-IN'
    let transcript = ''
    try {
      if (status?.stt.provider && status.stt.provider !== 'browser' && status.stt.configured) {
        abortListen.current = () => m.stopRecording()
        const blob = await m.record({ getLevel: () => level.current })
        releaseMic()
        if (blob.size >= 2000) {   // smaller recordings carry no speech; handled as "no words" below
          setMode('processing')
          const r = await api.companion.transcribe(blob, lang)
          transcript = r.text
          setInterim(transcript)
        }
      } else {
        transcript = await recognizeOnce(lang, setInterim, (abort) => { abortListen.current = abort })
        releaseMic()
      }
    } catch (e) {
      const err = e as Error & { name?: string }
      fail(err.name === 'NotAllowedError' ? micErrorText.denied : err.message, () => void listen())
      return
    } finally {
      abortListen.current = null
    }
    if (cancelled.current) return // user pressed Stop
    if (!transcript.trim()) {
      setInterim('')
      // Never fail silently: say whether the mic produced sound at all.
      fail(peak < NO_SOUND_LEVEL
        ? 'Your microphone is not sending any sound. Check that the right microphone is selected and not muted (Windows: Settings → System → Sound → Input), then try again.'
        : `I heard sound but the speech recogniser returned no words (language: ${lang}). Speak a little longer and closer to the mic, check the "Speech recognition language" in Settings matches what you speak, and use Chrome or Edge with an internet connection.`,
      () => void listen())
      return
    }
    await send(transcript, 'voice')
    setInterim('')
  }, [fail, send])

  const stopListening = useCallback(() => {
    cancelled.current = true
    abortListen.current?.()
    releaseMic()
    setMode('idle')
  }, [])

  const stopSpeaking = useCallback(() => {
    stopSpeech.current?.()
    setMode('idle')
  }, [])

  /** Barge-in: cut playback immediately and hand the turn back to the user. */
  const interrupt = useCallback(() => {
    stopSpeech.current?.()
    setMode('interrupted')
    window.setTimeout(() => void listen(), 450)
  }, [listen])

  const toggleMute = useCallback(() => {
    setMuted((m) => {
      if (!m) stopSpeech.current?.()
      return !m
    })
    if (modeRef.current === 'speaking') setMode('idle')
  }, [])

  const dismissError = useCallback(() => { setError(null); setMode('idle') }, [])

  useEffect(() => () => { releaseMic(); stopSpeech.current?.() }, [])

  const orbState: OrbState =
    mode === 'speaking' ? (emotion === 'neutral' ? 'speaking' : emotion)
      : mode === 'idle' ? (emotion === 'neutral' ? 'idle' : emotion)
        : mode

  return { mode, orbState, emotion, error, interim, muted, handsFree, setHandsFree, level, listen, stopListening, send, stopSpeaking, interrupt, toggleMute, dismissError }
}

export type { Message }
