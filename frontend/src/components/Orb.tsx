import { useEffect, useRef } from 'react'

export type OrbState =
  | 'idle' | 'listening' | 'processing' | 'speaking'
  | 'happy' | 'supportive' | 'curious' | 'interrupted' | 'error'

interface Palette { core: string; mid: string; edge: string; glow: string }
const PALETTES: Record<OrbState, Palette> = {
  idle:        { core: '#a5f3fc', mid: '#22d3ee', edge: '#1e3a8a', glow: 'rgba(34,211,238,0.28)' },
  listening:   { core: '#ecfeff', mid: '#22d3ee', edge: '#0e7490', glow: 'rgba(34,211,238,0.55)' },
  processing:  { core: '#e9d5ff', mid: '#8b5cf6', edge: '#1e1b4b', glow: 'rgba(139,92,246,0.45)' },
  speaking:    { core: '#f0fdff', mid: '#38bdf8', edge: '#4c1d95', glow: 'rgba(56,189,248,0.55)' },
  happy:       { core: '#fefce8', mid: '#2dd4bf', edge: '#0891b2', glow: 'rgba(45,212,191,0.6)' },
  supportive:  { core: '#ede9fe', mid: '#a78bfa', edge: '#312e81', glow: 'rgba(167,139,250,0.4)' },
  curious:     { core: '#e0f2fe', mid: '#22d3ee', edge: '#6d28d9', glow: 'rgba(99,102,241,0.45)' },
  interrupted: { core: '#fef3c7', mid: '#f59e0b', edge: '#78350f', glow: 'rgba(245,158,11,0.45)' },
  error:       { core: '#ffe4e6', mid: '#fb7185', edge: '#4c0519', glow: 'rgba(251,113,133,0.45)' },
}

// How each state moves. `react` = how strongly the outline follows the real audio level.
const MOTION: Record<OrbState, { breathe: number; speed: number; react: number; wobble: number }> = {
  idle:        { breathe: 0.018, speed: 0.35, react: 0,    wobble: 0.012 },
  listening:   { breathe: 0.006, speed: 1.0,  react: 0.28, wobble: 0.02 },
  processing:  { breathe: 0.01,  speed: 0.8,  react: 0,    wobble: 0.015 },
  speaking:    { breathe: 0.004, speed: 1.3,  react: 0.32, wobble: 0.025 },
  happy:       { breathe: 0.03,  speed: 1.6,  react: 0.3,  wobble: 0.035 },
  supportive:  { breathe: 0.02,  speed: 0.45, react: 0.18, wobble: 0.012 },
  curious:     { breathe: 0.015, speed: 0.9,  react: 0.25, wobble: 0.02 },
  interrupted: { breathe: 0,     speed: 2.2,  react: 0,    wobble: 0.04 },
  error:       { breathe: 0.008, speed: 0.25, react: 0,    wobble: 0.006 },
}

function mix(a: number, b: number, t: number) { return a + (b - a) * t }

export function Orb({ state, levelRef, size = 320 }: { state: OrbState; levelRef: React.RefObject<number>; size?: number }) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const stateRef = useRef(state)
  const changedAt = useRef(performance.now())
  useEffect(() => { stateRef.current = state; changedAt.current = performance.now() }, [state])

  useEffect(() => {
    const c = canvas.current!
    const ctx = c.getContext('2d')!
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    c.width = size * dpr
    c.height = size * dpr
    ctx.scale(dpr, dpr)
    const reduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches
    let raf = 0
    let smoothLevel = 0
    let reactAmt = 0
    const start = performance.now()

    const draw = (now: number) => {
      const s = stateRef.current
      const m = MOTION[s]
      const p = PALETTES[s]
      const t = (now - start) / 1000
      const since = (now - changedAt.current) / 1000
      const lvl = Math.max(0, Math.min(1, levelRef.current ?? 0))
      smoothLevel = mix(smoothLevel, lvl, 0.25)
      reactAmt = mix(reactAmt, m.react, 0.08)

      const cx = size / 2, cy = size / 2
      const base = size * 0.3
      const breathe = reduce ? 0 : Math.sin(t * (s === 'supportive' ? 0.9 : 1.6)) * m.breathe
      let r = base * (1 + breathe + smoothLevel * reactAmt * 0.6)
      if (s === 'interrupted') r *= 1 - 0.12 * Math.exp(-since * 6) // quick contraction
      ctx.clearRect(0, 0, size, size)

      // outer glow
      const g = ctx.createRadialGradient(cx, cy, r * 0.6, cx, cy, r * 1.9)
      g.addColorStop(0, p.glow)
      g.addColorStop(1, 'rgba(0,0,0,0)')
      ctx.fillStyle = g
      ctx.beginPath(); ctx.arc(cx, cy, r * 1.9, 0, Math.PI * 2); ctx.fill()

      // deformed body: outline driven by real audio level
      const pts = 96
      ctx.beginPath()
      for (let i = 0; i <= pts; i++) {
        const a = (i / pts) * Math.PI * 2
        const n = reduce ? 0 :
          Math.sin(a * 3 + t * m.speed * 2) * m.wobble +
          Math.sin(a * 5 - t * m.speed * 1.3) * m.wobble * 0.6 +
          Math.sin(a * 7 + t * m.speed * 3.1) * smoothLevel * reactAmt * 0.22
        const rr = r * (1 + n)
        const x = cx + Math.cos(a) * rr, y = cy + Math.sin(a) * rr
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y)
      }
      ctx.closePath()
      const body = ctx.createRadialGradient(cx - r * 0.35, cy - r * 0.4, r * 0.05, cx, cy, r * 1.1)
      body.addColorStop(0, p.core)
      body.addColorStop(0.45, p.mid)
      body.addColorStop(1, p.edge)
      ctx.fillStyle = body
      ctx.fill()

      // inner sheen
      ctx.save()
      ctx.clip()
      const sheen = ctx.createLinearGradient(cx - r, cy - r, cx + r, cy + r)
      sheen.addColorStop(0, 'rgba(255,255,255,0.18)')
      sheen.addColorStop(0.5, 'rgba(255,255,255,0)')
      ctx.fillStyle = sheen
      ctx.fillRect(cx - r * 1.2, cy - r * 1.2, r * 2.4, r * 2.4)
      ctx.restore()

      // processing: orbiting arcs (a thinking indicator, not an audio level)
      if (s === 'processing' && !reduce) {
        for (let k = 0; k < 3; k++) {
          const a0 = t * (1.4 + k * 0.5) + k * 2.1
          ctx.strokeStyle = k === 1 ? 'rgba(34,211,238,0.75)' : 'rgba(167,139,250,0.7)'
          ctx.lineWidth = 2
          ctx.beginPath()
          ctx.arc(cx, cy, r * (1.18 + k * 0.1), a0, a0 + 0.9)
          ctx.stroke()
        }
      }
      // curious: a small satellite that tilts around the orb
      if (s === 'curious' && !reduce) {
        const a = t * 1.2
        ctx.fillStyle = 'rgba(224,242,254,0.9)'
        ctx.beginPath(); ctx.arc(cx + Math.cos(a) * r * 1.3, cy + Math.sin(a) * r * 0.5 - r * 0.2, 4, 0, Math.PI * 2); ctx.fill()
      }
      // listening / speaking: concentric ring whose radius follows the level
      if ((s === 'listening' || s === 'speaking' || s === 'happy') && smoothLevel > 0.02) {
        ctx.strokeStyle = s === 'listening' ? 'rgba(103,232,249,0.5)' : 'rgba(139,92,246,0.45)'
        ctx.lineWidth = 1.5
        ctx.beginPath(); ctx.arc(cx, cy, r * (1.12 + smoothLevel * 0.35), 0, Math.PI * 2); ctx.stroke()
      }
      // error: thin broken ring
      if (s === 'error') {
        ctx.setLineDash([6, 10])
        ctx.strokeStyle = 'rgba(251,113,133,0.6)'
        ctx.beginPath(); ctx.arc(cx, cy, r * 1.2, 0, Math.PI * 2); ctx.stroke()
        ctx.setLineDash([])
      }
      raf = requestAnimationFrame(draw)
    }
    raf = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(raf)
  }, [size, levelRef])

  return <canvas ref={canvas} style={{ width: size, height: size }} className="max-w-full" aria-label={`ECHO is ${state}`} role="img" />
}

/** Bar waveform driven by the same real level signal (history of recent values). */
export function Waveform({ levelRef, active, color = '#22d3ee', bars = 48 }: { levelRef: React.RefObject<number>; active: boolean; color?: string; bars?: number }) {
  const canvas = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = canvas.current!
    const ctx = c.getContext('2d')!
    const w = 320, h = 48
    const dpr = Math.min(window.devicePixelRatio || 1, 2)
    c.width = w * dpr; c.height = h * dpr; ctx.scale(dpr, dpr)
    const hist = new Array(bars).fill(0)
    let raf = 0, frame = 0
    const draw = () => {
      frame++
      if (frame % 3 === 0) { hist.shift(); hist.push(active ? Math.max(0, Math.min(1, levelRef.current ?? 0)) : 0) }
      ctx.clearRect(0, 0, w, h)
      const bw = w / bars
      hist.forEach((v, i) => {
        const bh = Math.max(2, v * h * 0.95)
        ctx.fillStyle = color
        ctx.globalAlpha = 0.25 + (i / bars) * 0.75
        ctx.fillRect(i * bw + 1, (h - bh) / 2, bw - 2, bh)
      })
      ctx.globalAlpha = 1
      raf = requestAnimationFrame(draw)
    }
    raf = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(raf)
  }, [active, color, bars, levelRef])
  return <canvas ref={canvas} style={{ width: 320, height: 48 }} className="max-w-full" aria-hidden />
}
