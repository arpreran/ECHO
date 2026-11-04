import { Link } from 'react-router-dom'
import { useEffect, useState } from 'react'
import { api, type Health } from '../lib/api'
import { cx } from './ui'

export function Logo({ to = '/', sub }: { to?: string; sub?: string }) {
  return (
    <Link to={to} className="focus-ring group flex items-center gap-2.5 rounded-lg">
      <span className="relative grid size-8 place-items-center">
        <span className="absolute inset-0 rounded-full bg-cyan-glow/20 blur-md transition group-hover:bg-cyan-glow/35" />
        <span className="relative size-5 rounded-full bg-[radial-gradient(circle_at_35%_30%,#ecfeff,#22d3ee_45%,#4c1d95)]" />
      </span>
      <span className="leading-tight">
        <span className="h-display block text-[15px] font-semibold tracking-[0.28em] text-white">ECHO</span>
        {sub && <span className="block text-[10px] uppercase tracking-[0.2em] text-mist">{sub}</span>}
      </span>
    </Link>
  )
}

/** Polls /api/health so connection state shown in the UI is real. */
export function useHealth(intervalMs = 15000) {
  const [health, setHealth] = useState<Health | null>(null)
  const [online, setOnline] = useState<boolean | null>(null)
  useEffect(() => {
    let alive = true
    const check = () => api.health().then((h) => { if (alive) { setHealth(h); setOnline(true) } }).catch(() => alive && setOnline(false))
    check()
    const id = setInterval(check, intervalMs)
    return () => { alive = false; clearInterval(id) }
  }, [intervalMs])
  return { health, online }
}

export function StatusDot({ ok, label }: { ok: boolean | null; label: string }) {
  return (
    <span className="inline-flex items-center gap-2 text-xs text-mist">
      <span className={cx('size-2 rounded-full', ok === null ? 'bg-mist animate-pulse' : ok ? 'bg-pos shadow-[0_0_8px_#34d399]' : 'bg-neg shadow-[0_0_8px_#fb7185]')} />
      {label}
    </span>
  )
}
