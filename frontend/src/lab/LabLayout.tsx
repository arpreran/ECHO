import { BarChart3, BookOpen, Database, FlaskConical, Gauge, Languages, Layers, Lightbulb, Scale, Smile, Menu, X } from 'lucide-react'
import { useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import { Logo, StatusDot, useHealth } from '../components/brand'
import { cx } from '../components/ui'

const NAV = [
  { to: '/lab', end: true, label: 'Overview', icon: Gauge },
  { to: '/lab/workbench', label: 'Analysis Workbench', icon: FlaskConical },
  { to: '/lab/language', label: 'Romanized Language Detector', icon: Languages },
  { to: '/lab/code-mixing', label: 'Code-Mixing Intelligence', icon: Layers },
  { to: '/lab/sentiment', label: 'Sentiment Analysis', icon: Smile },
  { to: '/lab/explainability', label: 'Explainability Explorer', icon: Lightbulb },
  { to: '/lab/models', label: 'Model Comparison', icon: Scale },
  { to: '/lab/datasets', label: 'Dataset Explorer', icon: Database },
  { to: '/lab/analytics', label: 'Research Analytics', icon: BarChart3 },
  { to: '/lab/methodology', label: 'Methodology', icon: BookOpen },
]

export function LabLayout() {
  const { health, online } = useHealth()
  const [open, setOpen] = useState(false)
  const ready = health?.nlp_lab.lid_ready
  return (
    <div className="min-h-screen lg:grid lg:grid-cols-[260px_1fr]">
      <aside className={cx('fixed inset-y-0 left-0 z-30 w-[260px] border-r border-line bg-ink-900/95 backdrop-blur-xl transition-transform lg:sticky lg:top-0 lg:h-screen lg:translate-x-0', open ? 'translate-x-0' : '-translate-x-full')}>
        <div className="flex items-center justify-between px-5 py-5"><Logo sub="NLP Lab" /><button className="lg:hidden" aria-label="Close menu" onClick={() => setOpen(false)}><X className="size-5" /></button></div>
        <nav className="scroll-thin space-y-0.5 overflow-y-auto px-3 pb-6" style={{ maxHeight: 'calc(100vh - 140px)' }}>
          {NAV.map(({ to, end, label, icon: Icon }) => (
            <NavLink key={to} to={to} end={end} onClick={() => setOpen(false)}
              className={({ isActive }) => cx('focus-ring flex items-center gap-3 rounded-xl px-3 py-2 text-sm transition-colors', isActive ? 'bg-violet-glow/12 text-white shadow-[inset_2px_0_0_#8b5cf6]' : 'text-mist hover:bg-white/[0.03] hover:text-white')}>
              <Icon className="size-4 shrink-0" />{label}
            </NavLink>
          ))}
        </nav>
        <div className="absolute inset-x-0 bottom-0 border-t border-line px-5 py-4">
          <StatusDot ok={online === false ? false : ready ?? null} label={online === false ? 'Backend offline' : ready ? `${health?.nlp_lab.sentiment_models.length} models loaded` : health ? 'Models not trained' : 'Connecting…'} />
          <p className="mt-1 text-[10px] text-mist">Classical ML only · no external AI APIs</p>
        </div>
      </aside>
      {open && <div className="fixed inset-0 z-20 bg-black/50 lg:hidden" onClick={() => setOpen(false)} />}
      <div className="min-w-0">
        <div className="sticky top-0 z-10 flex items-center gap-3 border-b border-line bg-ink-950/80 px-4 py-3 backdrop-blur-xl lg:hidden">
          <button aria-label="Open menu" onClick={() => setOpen(true)} className="focus-ring rounded-lg p-1"><Menu className="size-5" /></button>
          <span className="h-display text-sm font-semibold tracking-[0.2em] text-white">ECHO · NLP LAB</span>
        </div>
        <main className="mx-auto max-w-6xl px-4 py-8 sm:px-8">
          {health && health.nlp_lab.errors.length > 0 && (
            <div className="mb-6 rounded-xl border border-warn/30 bg-warn/5 p-3 text-sm text-warn">{health.nlp_lab.errors.join(' · ')}</div>
          )}
          <Outlet />
        </main>
      </div>
    </div>
  )
}

export function PageHeader({ eyebrow, title, children }: { eyebrow: string; title: string; children?: React.ReactNode }) {
  return (
    <div className="mb-8">
      <div className="eyebrow text-violet-300">{eyebrow}</div>
      <h1 className="h-display mt-1 text-3xl font-semibold text-white">{title}</h1>
      {children && <div className="mt-3 max-w-3xl text-sm leading-relaxed text-mist">{children}</div>}
    </div>
  )
}
