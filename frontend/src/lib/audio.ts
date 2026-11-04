// Audio primitives for ECHO Companion. Every level value reported here comes
// from a real signal: microphone RMS, playback RMS, or speech-synthesis
// word-boundary events. Nothing is simulated.

type LevelFn = (level: number) => void

let sharedCtx: AudioContext | null = null
export function audioContext(): AudioContext {
  if (!sharedCtx || sharedCtx.state === 'closed') sharedCtx = new AudioContext()
  if (sharedCtx.state === 'suspended') void sharedCtx.resume()
  return sharedCtx
}

function rmsLoop(analyser: AnalyserNode, onLevel: LevelFn): () => void {
  const buf = new Float32Array(analyser.fftSize)
  let raf = 0
  let smooth = 0
  const tick = () => {
    analyser.getFloatTimeDomainData(buf)
    let s = 0
    for (let i = 0; i < buf.length; i++) s += buf[i] * buf[i]
    const rms = Math.sqrt(s / buf.length)
    const lvl = Math.min(1, rms * 6) // speech RMS is ~0.02-0.2
    smooth = smooth * 0.7 + lvl * 0.3
    onLevel(smooth)
    raf = requestAnimationFrame(tick)
  }
  raf = requestAnimationFrame(tick)
  return () => cancelAnimationFrame(raf)
}

export type MicError = 'denied' | 'no-device' | 'insecure' | 'unknown'

export function classifyMicError(e: unknown): MicError {
  const name = (e as { name?: string })?.name
  if (!window.isSecureContext) return 'insecure'
  if (name === 'NotAllowedError' || name === 'SecurityError') return 'denied'
  if (name === 'NotFoundError' || name === 'OverconstrainedError') return 'no-device'
  return 'unknown'
}

export const micErrorText: Record<MicError, string> = {
  denied: 'Microphone permission was denied. Allow microphone access in your browser’s site settings, then try again.',
  'no-device': 'No microphone was found. Connect one and try again.',
  insecure: 'Microphone access needs a secure context (http://localhost or https).',
  unknown: 'The microphone could not be started.',
}

/** Live microphone level monitor (and optional recorder with simple VAD). */
export class MicSession {
  stream: MediaStream | null = null
  private stopLoop: (() => void) | null = null
  private recorder: MediaRecorder | null = null
  private chunks: Blob[] = []

  async start(onLevel: LevelFn): Promise<void> {
    this.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } })
    const ctx = audioContext()
    const src = ctx.createMediaStreamSource(this.stream)
    const analyser = ctx.createAnalyser()
    analyser.fftSize = 1024
    src.connect(analyser)
    this.stopLoop = rmsLoop(analyser, onLevel)
  }

  /** Record until `silenceMs` of silence after speech, or until stopRecording(). */
  record(opts: { silenceMs?: number; maxMs?: number; getLevel: () => number }): Promise<Blob> {
    if (!this.stream) return Promise.reject(new Error('Mic not started'))
    const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : ''
    this.recorder = new MediaRecorder(this.stream, mime ? { mimeType: mime } : undefined)
    this.chunks = []
    const silenceMs = opts.silenceMs ?? 1300
    const maxMs = opts.maxMs ?? 30000
    return new Promise((resolve, reject) => {
      const r = this.recorder!
      let spoke = false
      let lastLoud = performance.now()
      const started = performance.now()
      const iv = window.setInterval(() => {
        const l = opts.getLevel()
        if (l > 0.08) { spoke = true; lastLoud = performance.now() }
        const now = performance.now()
        if ((spoke && now - lastLoud > silenceMs) || now - started > maxMs) r.state !== 'inactive' && r.stop()
      }, 100)
      r.ondataavailable = (e) => e.data.size && this.chunks.push(e.data)
      r.onerror = () => { clearInterval(iv); reject(new Error('Recording failed')) }
      r.onstop = () => { clearInterval(iv); resolve(new Blob(this.chunks, { type: r.mimeType || 'audio/webm' })) }
      r.start(250)
    })
  }

  stopRecording() { if (this.recorder && this.recorder.state !== 'inactive') this.recorder.stop() }

  stop() {
    this.stopLoop?.()
    this.stopLoop = null
    this.stopRecording()
    this.stream?.getTracks().forEach((t) => t.stop())
    this.stream = null
  }
}

/** Plays server-synthesised audio and reports its real output level. */
export class AudioPlayback {
  private src: AudioBufferSourceNode | null = null
  private stopLoop: (() => void) | null = null

  async play(data: ArrayBuffer, onLevel: LevelFn, muted: boolean): Promise<'ended' | 'stopped'> {
    this.stop()
    const ctx = audioContext()
    const buffer = await ctx.decodeAudioData(data.slice(0))
    const src = ctx.createBufferSource()
    src.buffer = buffer
    const analyser = ctx.createAnalyser()
    analyser.fftSize = 1024
    const gain = ctx.createGain()
    gain.gain.value = muted ? 0 : 1
    src.connect(analyser)
    analyser.connect(gain)
    gain.connect(ctx.destination)
    this.src = src
    this.stopLoop = rmsLoop(analyser, onLevel)
    return new Promise((resolve) => {
      let stopped = false
      src.onended = () => {
        this.stopLoop?.(); this.stopLoop = null; onLevel(0)
        resolve(stopped ? 'stopped' : 'ended')
      }
      ;(src as AudioBufferSourceNode & { _stop?: () => void })._stop = () => { stopped = true }
      src.start()
    })
  }

  stop() {
    if (this.src) {
      ;(this.src as AudioBufferSourceNode & { _stop?: () => void })._stop?.()
      try { this.src.stop() } catch { /* already stopped */ }
      this.src = null
    }
  }
}

// ----------------------------------------------------- browser speech APIs
interface SRResultEvent { resultIndex: number; results: ArrayLike<{ isFinal: boolean; 0: { transcript: string } }> }
interface SR {
  lang: string; continuous: boolean; interimResults: boolean
  onresult: ((e: SRResultEvent) => void) | null
  onerror: ((e: { error: string }) => void) | null
  onend: (() => void) | null
  start(): void; stop(): void; abort(): void
}
type SRCtor = new () => SR

export function speechRecognitionCtor(): SRCtor | null {
  const w = window as unknown as { SpeechRecognition?: SRCtor; webkitSpeechRecognition?: SRCtor }
  return w.SpeechRecognition || w.webkitSpeechRecognition || null
}

/** Browsers that expose SpeechRecognition but whose speech service does not work
 *  (it silently returns nothing or a network error). Returns the browser name, or null. */
export function brokenSpeechRecognitionBrowser(): string | null {
  const ua = navigator.userAgent
  if ((navigator as Navigator & { brave?: unknown }).brave) return 'Brave'
  if (/\bOPR\/|Opera/.test(ua)) return 'Opera'
  if (/Vivaldi/.test(ua)) return 'Vivaldi'
  return null
}

/** One utterance of browser speech recognition. Resolves with the final transcript. */
export function recognizeOnce(lang: string, onInterim: (t: string) => void, register: (abort: () => void) => void): Promise<string> {
  const Ctor = speechRecognitionCtor()
  if (!Ctor) return Promise.reject(new Error('This browser has no built-in speech recognition. Use Chrome or Edge, or enable server speech-to-text.'))
  return new Promise((resolve, reject) => {
    const rec = new Ctor()
    rec.lang = lang
    rec.continuous = false
    rec.interimResults = true
    let finalText = ''
    let settled = false
    register(() => { settled = true; rec.abort(); resolve('') })
    rec.onresult = (e) => {
      let interim = ''
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const r = e.results[i]
        if (r.isFinal) finalText += r[0].transcript
        else interim += r[0].transcript
      }
      onInterim((finalText + ' ' + interim).trim())
    }
    rec.onerror = (e) => {
      if (settled) return
      settled = true
      if (e.error === 'no-speech' || e.error === 'aborted') resolve('')
      else if (e.error === 'not-allowed') reject(Object.assign(new Error(micErrorText.denied), { name: 'NotAllowedError' }))
      else if (e.error === 'language-not-supported') reject(new Error(`Speech recognition does not support ${lang} in this browser.`))
      else if (e.error === 'network') reject(new Error('Browser speech recognition needs an internet connection (it uses the browser vendor’s service).'))
      else reject(new Error(`Speech recognition error: ${e.error}`))
    }
    rec.onend = () => { if (!settled) { settled = true; resolve(finalText.trim()) } }
    rec.start()
  })
}

export function browserVoices(): SpeechSynthesisVoice[] {
  return typeof speechSynthesis === 'undefined' ? [] : speechSynthesis.getVoices()
}

export function pickVoice(langTag: string, preferredName?: string): SpeechSynthesisVoice | null {
  const voices = browserVoices()
  if (preferredName) {
    const v = voices.find((x) => x.name === preferredName)
    if (v) return v
  }
  const base = langTag.split('-')[0]
  return voices.find((v) => v.lang === langTag) || voices.find((v) => v.lang.startsWith(base)) || voices.find((v) => v.lang === 'en-IN') || null
}

/** Browser TTS. Level pulses come from real `boundary` (word) events. */
export function speakBrowser(text: string, opts: { voice: SpeechSynthesisVoice | null; rate: number; lang: string; muted: boolean; onLevel: LevelFn }): { done: Promise<'ended' | 'stopped'>; stop: () => void } {
  let stopped = false
  let decay = 0
  const utter = new SpeechSynthesisUtterance(text)
  if (opts.voice) utter.voice = opts.voice
  utter.lang = opts.voice?.lang || opts.lang
  utter.rate = opts.rate
  utter.volume = opts.muted ? 0 : 1
  let level = 0
  const tick = () => { level *= 0.88; opts.onLevel(level); decay = requestAnimationFrame(tick) }
  const done = new Promise<'ended' | 'stopped'>((resolve) => {
    utter.onstart = () => { level = 0.6; decay = requestAnimationFrame(tick) }
    utter.onboundary = () => { level = 0.55 + 0.35 * Math.min(1, 1) }
    utter.onend = () => { cancelAnimationFrame(decay); opts.onLevel(0); resolve(stopped ? 'stopped' : 'ended') }
    utter.onerror = () => { cancelAnimationFrame(decay); opts.onLevel(0); resolve('stopped') }
  })
  speechSynthesis.cancel()
  speechSynthesis.speak(utter)
  return { done, stop: () => { stopped = true; speechSynthesis.cancel() } }
}
