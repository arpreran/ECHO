import { AlertTriangle, Inbox, Loader2, RotateCcw } from 'lucide-react'
import { useCallback, useEffect, useRef, useState, type ButtonHTMLAttributes, type ReactNode } from 'react'

export function cx(...c: (string | false | null | undefined)[]) { return c.filter(Boolean).join(' ') }

type Variant = 'primary' | 'ghost' | 'subtle' | 'danger'
export function Button({ variant = 'subtle', size = 'md', loading, icon, children, className, ...rest }:
  ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; size?: 'sm' | 'md' | 'lg'; loading?: boolean; icon?: ReactNode }) {
  const v = {
    primary: 'bg-cyan-glow text-ink-950 hover:bg-cyan-soft shadow-[0_0_24px_-6px_rgba(34,211,238,0.7)]',
    ghost: 'text-fog hover:bg-white/5',
    subtle: 'bg-white/[0.04] text-fog border border-line hover:border-line-strong hover:bg-white/[0.07]',
    danger: 'bg-neg/10 text-neg border border-neg/30 hover:bg-neg/20',
  }[variant]
  const s = { sm: 'h-8 px-3 text-xs gap-1.5', md: 'h-10 px-4 text-sm gap-2', lg: 'h-12 px-6 text-base gap-2' }[size]
  return (
    <button {...rest} disabled={rest.disabled || loading}
      className={cx('focus-ring inline-flex items-center justify-center rounded-xl font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-45', v, s, className)}>
      {loading ? <Loader2 className="size-4 animate-spin" /> : icon}
      {children}
    </button>
  )
}

export function Panel({ title, eyebrow, actions, children, className, glass }: { title?: ReactNode; eyebrow?: string; actions?: ReactNode; children: ReactNode; className?: string; glass?: boolean }) {
  return (
    <section className={cx(glass ? 'glass' : 'panel', 'min-w-0 rounded-2xl p-5', className)}>
      {(title || actions || eyebrow) && (
        <header className="mb-4 flex flex-wrap items-start justify-between gap-3">
          <div>
            {eyebrow && <div className="eyebrow mb-1">{eyebrow}</div>}
            {title && <h3 className="h-display text-base font-semibold text-white">{title}</h3>}
          </div>
          {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
        </header>
      )}
      {children}
    </section>
  )
}

export function Badge({ children, tone = 'neutral', className }: { children: ReactNode; tone?: 'neutral' | 'cyan' | 'violet' | 'pos' | 'neg' | 'neu' | 'warn'; className?: string }) {
  const t = {
    neutral: 'bg-white/5 text-fog border-line',
    cyan: 'bg-cyan-glow/10 text-cyan-soft border-cyan-glow/30',
    violet: 'bg-violet-glow/10 text-violet-300 border-violet-glow/30',
    pos: 'bg-pos/10 text-pos border-pos/30',
    neg: 'bg-neg/10 text-neg border-neg/30',
    neu: 'bg-neu/10 text-neu border-neu/30',
    warn: 'bg-warn/10 text-warn border-warn/30',
  }[tone]
  return <span className={cx('inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-[11px] font-medium', t, className)}>{children}</span>
}

export function Toggle({ checked, onChange, label, hint, disabled }: { checked: boolean; onChange: (v: boolean) => void; label: string; hint?: string; disabled?: boolean }) {
  return (
    <label className={cx('flex items-start justify-between gap-4 py-2', disabled && 'opacity-50')}>
      <span>
        <span className="block text-sm text-white">{label}</span>
        {hint && <span className="block text-xs text-mist">{hint}</span>}
      </span>
      <button type="button" role="switch" aria-checked={checked} disabled={disabled} onClick={() => onChange(!checked)}
        className={cx('focus-ring relative mt-0.5 h-6 w-11 shrink-0 rounded-full border transition-colors', checked ? 'border-cyan-glow/50 bg-cyan-glow/30' : 'border-line bg-white/5')}>
        <span className={cx('absolute top-0.5 size-4.5 rounded-full transition-all', checked ? 'left-[22px] bg-cyan-soft' : 'left-0.5 bg-mist')} />
      </button>
    </label>
  )
}

export function Select<T extends string>({ value, onChange, options, label, className }: { value: T; onChange: (v: T) => void; options: { value: T; label: string }[]; label?: string; className?: string }) {
  return (
    <label className={cx('block', className)}>
      {label && <span className="eyebrow mb-1.5 block">{label}</span>}
      <select value={value} onChange={(e) => onChange(e.target.value as T)}
        className="focus-ring h-10 w-full rounded-xl border border-line bg-ink-850 px-3 text-sm text-white">
        {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    </label>
  )
}

export function Segmented<T extends string>({ value, onChange, options }: { value: T; onChange: (v: T) => void; options: { value: T; label: string }[] }) {
  return (
    <div className="inline-flex rounded-xl border border-line bg-ink-850 p-1" role="tablist">
      {options.map((o) => (
        <button key={o.value} role="tab" aria-selected={value === o.value} onClick={() => onChange(o.value)}
          className={cx('focus-ring rounded-lg px-3 py-1.5 text-xs font-medium transition-colors', value === o.value ? 'bg-cyan-glow/15 text-cyan-soft' : 'text-mist hover:text-white')}>
          {o.label}
        </button>
      ))}
    </div>
  )
}

export function Loading({ label = 'Loading…' }: { label?: string }) {
  return <div className="flex items-center gap-2 py-10 text-sm text-mist justify-center"><Loader2 className="size-4 animate-spin text-cyan-glow" />{label}</div>
}

export function ErrorState({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const msg = error instanceof Error ? error.message : String(error)
  return (
    <div className="flex flex-col items-center gap-3 rounded-xl border border-neg/25 bg-neg/5 px-4 py-8 text-center">
      <AlertTriangle className="size-6 text-neg" />
      <p className="max-w-md text-sm text-fog">{msg}</p>
      {onRetry && <Button size="sm" icon={<RotateCcw className="size-3.5" />} onClick={onRetry}>Retry</Button>}
    </div>
  )
}

export function Empty({ title, hint, action }: { title: string; hint?: string; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center gap-2 py-10 text-center">
      <Inbox className="size-6 text-mist" />
      <p className="text-sm text-white">{title}</p>
      {hint && <p className="max-w-sm text-xs text-mist">{hint}</p>}
      {action}
    </div>
  )
}

export function useAsync<T>(fn: () => Promise<T>, deps: unknown[] = []) {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<unknown>(null)
  const [loading, setLoading] = useState(true)
  const fnRef = useRef(fn)
  fnRef.current = fn
  const reload = useCallback(() => {
    setLoading(true); setError(null)
    fnRef.current().then((d) => setData(d)).catch(setError).finally(() => setLoading(false))
  }, [])
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(reload, deps)
  return { data, error, loading, reload, setData }
}

export function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: string }) {
  return (
    <div className="rounded-xl border border-line bg-white/[0.02] p-4">
      <div className="eyebrow">{label}</div>
      <div className="h-display mt-1 text-2xl font-semibold text-white tabular-nums">{value}</div>
      {hint && <div className="mt-1 text-xs text-mist">{hint}</div>}
    </div>
  )
}

export const pct = (v: number | null | undefined, d = 1) => (v == null ? '—' : `${(v * 100).toFixed(d)}%`)
export const num = (v: number | null | undefined, d = 3) => (v == null ? '—' : v.toFixed(d))
