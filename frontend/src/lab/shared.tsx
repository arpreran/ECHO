import { Sparkles } from 'lucide-react'
import { useState, type ReactNode } from 'react'
import { Badge, Button, cx, Segmented } from '../components/ui'
import type { Explanation, ModelKind, SentimentLang, SentimentPrediction, TokenTag } from '../lib/api'

export const LANG_COLOR: Record<string, string> = {
  en: '#a5b4fc', kn: '#22d3ee', kn_rom: '#22d3ee', ta: '#fbbf24', ta_rom: '#fbbf24', ml: '#34d399', ml_rom: '#34d399',
  hi: '#f472b6', hi_rom: '#f472b6', univ: '#64748b', other: '#ef4444',
}
export const LANG_LABEL: Record<string, string> = {
  en: 'English', kn: 'Kannada', kn_rom: 'Kannada (Rom.)', ta: 'Tamil', ta_rom: 'Tamil (Rom.)', ml: 'Malayalam', ml_rom: 'Malayalam (Rom.)',
  hi: 'Hindi', hi_rom: 'Hindi (Rom.)', univ: 'Language-independent', other: 'Unsupported',
}
export const SENT_COLOR: Record<string, string> = { positive: '#34d399', negative: '#fb7185', neutral: '#a5b4fc' }

export const EXAMPLES = [
  'Yeno hengidiyaa',
  'Neenga epdi irukeenga',
  'Sukham ano',
  'How are you?',
  'Naan office ge hogbeku today.',
  'ನಾನು office ಗೆ ಹೋಗಬೇಕು.',
  'Today college ge hogalla.',
  'ಇವತ್ತು class ತುಂಬಾ boring ide.',
  'Trailer semma mass, waiting for the movie',
  'Padam kollam, pakshe climax mosham aayi',
  'Movie thumba chennagide, must watch',
]

export function TextInput({ value, onChange, onSubmit, loading, submitLabel = 'Analyse', children }: { value: string; onChange: (v: string) => void; onSubmit: () => void; loading?: boolean; submitLabel?: string; children?: ReactNode }) {
  return (
    <div className="panel rounded-2xl p-4">
      <textarea value={value} onChange={(e) => onChange(e.target.value)} maxLength={2000} rows={3}
        onKeyDown={(e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) onSubmit() }}
        placeholder="Type Romanized, native-script or code-mixed text…"
        className="focus-ring w-full resize-y rounded-xl border border-line bg-ink-900 px-4 py-3 text-[15px] text-white placeholder:text-mist/60" />
      <div className="mt-3 flex flex-wrap items-center justify-between gap-3">
        <div className="flex flex-wrap gap-1.5">
          {EXAMPLES.slice(0, 8).map((ex) => (
            <button key={ex} onClick={() => onChange(ex)} className="focus-ring rounded-full border border-line px-2.5 py-1 text-[11px] text-mist hover:border-violet-glow/40 hover:text-white">{ex}</button>
          ))}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {children}
          <Button variant="primary" icon={<Sparkles className="size-4" />} loading={loading} disabled={!value.trim()} onClick={onSubmit}>{submitLabel}</Button>
        </div>
      </div>
    </div>
  )
}

export function ModelControls({ model, setModel, lang, setLang }: { model?: ModelKind; setModel?: (m: ModelKind) => void; lang: SentimentLang; setLang: (l: SentimentLang) => void }) {
  return (
    <>
      {setModel && model && <Segmented value={model} onChange={setModel} options={[{ value: 'logreg', label: 'Logistic Regression' }, { value: 'linear_svm', label: 'Linear SVM' }]} />}
      <Segmented value={lang} onChange={setLang} options={[{ value: 'auto', label: 'Auto (LID)' }, { value: 'kn', label: 'kn' }, { value: 'ta', label: 'ta' }, { value: 'ml', label: 'ml' }]} />
    </>
  )
}

export function TokenStrip({ tokens, showConf = true }: { tokens: TokenTag[]; showConf?: boolean }) {
  return (
    <div className="flex flex-wrap gap-2">
      {tokens.map((t, i) => (
        <div key={i} className={cx('rounded-xl border px-3 py-2 text-center', t.uncertain && t.lang !== 'univ' ? 'border-dashed' : '')}
          style={{ borderColor: `${LANG_COLOR[t.lang] ?? '#64748b'}66`, background: `${LANG_COLOR[t.lang] ?? '#64748b'}14` }}
          title={t.probabilities ? Object.entries(t.probabilities).map(([k, v]) => `${k}: ${(v * 100).toFixed(1)}%`).join('\n') : t.script}>
          <div className="text-[15px] text-white">{t.token}</div>
          <div className="mt-0.5 text-[10px] font-medium" style={{ color: LANG_COLOR[t.lang] ?? '#94a3b8' }}>{LANG_LABEL[t.lang] ?? t.lang}</div>
          {showConf && t.confidence != null && t.lang !== 'univ' && <div className="text-[10px] text-mist tabular-nums">{t.script === 'Latin' ? `${(t.confidence * 100).toFixed(0)}%` : 'script'}{t.uncertain ? ' · uncertain' : ''}</div>}
        </div>
      ))}
    </div>
  )
}

export function LangLegend() {
  return (
    <div className="flex flex-wrap gap-3 text-[11px] text-mist">
      {['en', 'kn', 'ta', 'ml', 'hi', 'univ'].map((k) => <span key={k} className="inline-flex items-center gap-1.5"><span className="size-2 rounded-full" style={{ background: LANG_COLOR[k] }} />{LANG_LABEL[k]}</span>)}
      <span className="inline-flex items-center gap-1.5"><span className="size-3 rounded border border-dashed border-mist" />uncertain (&lt;60%)</span>
    </div>
  )
}

export function ProbBars({ probs, colorFor }: { probs: Record<string, number>; colorFor?: (k: string) => string }) {
  const entries = Object.entries(probs).sort((a, b) => b[1] - a[1])
  return (
    <div className="space-y-2">
      {entries.map(([k, v]) => (
        <div key={k} className="grid grid-cols-[140px_1fr_52px] items-center gap-3 text-xs">
          <span className="truncate text-fog">{LANG_LABEL[k] ?? k}</span>
          <div className="h-2 overflow-hidden rounded-full bg-white/5"><div className="h-full rounded-full transition-all" style={{ width: `${v * 100}%`, background: colorFor?.(k) ?? LANG_COLOR[k] ?? '#22d3ee' }} /></div>
          <span className="text-right tabular-nums text-mist">{(v * 100).toFixed(1)}%</span>
        </div>
      ))}
    </div>
  )
}

export function SentimentSummary({ p }: { p: SentimentPrediction }) {
  if (!p.supported) return <div className="rounded-xl border border-warn/30 bg-warn/5 p-4 text-sm text-fog">{p.message}</div>
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <span className="h-display text-3xl font-semibold capitalize" style={{ color: SENT_COLOR[p.label!] }}>{p.label}</span>
        <Badge tone="violet">{p.model}</Badge>
        <Badge>{p.language_name} model · {p.feature_config}</Badge>
        {p.routing.source === 'lid' ? <Badge tone="cyan">routed by LID: {p.routing.lid?.primary_language_name}</Badge> : <Badge>language chosen manually</Badge>}
      </div>
      {p.probabilities ? (
        <div><div className="eyebrow mb-2">{p.confidence_type}</div><ProbBars probs={p.probabilities} colorFor={(k) => SENT_COLOR[k]} /></div>
      ) : (
        <div>
          <div className="eyebrow mb-2">{p.confidence_type} · margin {p.margin?.toFixed(3)}</div>
          <div className="grid grid-cols-3 gap-2">
            {Object.entries(p.decision_scores ?? {}).map(([k, v]) => (
              <div key={k} className="rounded-lg border border-line px-3 py-2 text-center"><div className="text-[11px] capitalize text-mist">{k}</div><div className="font-mono text-sm tabular-nums" style={{ color: SENT_COLOR[k] }}>{v.toFixed(3)}</div></div>
            ))}
          </div>
        </div>
      )}
      <p className="text-xs text-mist">Held-out test macro-F1 of this model: <span className="text-fog">{p.test_macro_f1?.toFixed(3)}</span>. Predictions on very short or out-of-domain text (the data are YouTube movie comments) are less reliable.</p>
    </div>
  )
}

export function ExplanationView({ ex }: { ex: Explanation }) {
  const [view, setView] = useState<'tokens' | 'features'>('tokens')
  const max = Math.max(1e-6, ...ex.tokens.map((t) => Math.abs(t.contribution)))
  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-xs text-mist">Contributions to the <span className="capitalize text-fog">{ex.target_class}</span> score ({ex.score.toFixed(3)} = tokens + other features {ex.non_token_contribution.toFixed(3)} + bias {ex.bias.toFixed(3)}).</p>
        <Segmented value={view} onChange={setView} options={[{ value: 'tokens', label: 'Tokens' }, { value: 'features', label: 'Top features' }]} />
      </div>
      {view === 'tokens' ? (
        <div className="flex flex-wrap gap-1.5">
          {ex.tokens.map((t, i) => {
            const a = Math.abs(t.contribution) / max
            const c = t.contribution >= 0 ? '52,211,153' : '251,113,133'
            return (
              <span key={i} className="rounded-lg px-2 py-1 text-[15px] text-white" style={{ background: `rgba(${c},${0.08 + a * 0.55})`, border: `1px solid rgba(${c},${0.2 + a * 0.5})` }}
                title={`${t.normalized} · ${LANG_LABEL[t.lang] ?? t.lang} · ${t.contribution >= 0 ? '+' : ''}${t.contribution.toFixed(4)}`}>{t.token}</span>
            )
          })}
        </div>
      ) : (
        <div className="space-y-1.5">
          {ex.top_features.map((f, i) => {
            const w = Math.abs(f.contribution) / Math.max(1e-6, Math.abs(ex.top_features[0].contribution))
            return (
              <div key={i} className="grid grid-cols-[1fr_2fr_70px] items-center gap-3 text-xs">
                <span className="truncate font-mono text-fog" title={f.feature}><span className="mr-1 text-mist">{f.block}</span>“{f.feature}”</span>
                <div className="h-2 rounded-full bg-white/5"><div className="h-full rounded-full" style={{ width: `${w * 100}%`, background: f.contribution >= 0 ? '#34d399' : '#fb7185' }} /></div>
                <span className="text-right font-mono tabular-nums text-mist">{f.contribution >= 0 ? '+' : ''}{f.contribution.toFixed(4)}</span>
              </div>
            )
          })}
        </div>
      )}
      <div className="flex gap-4 text-[11px] text-mist"><span className="inline-flex items-center gap-1"><span className="size-2 rounded bg-pos" />supports {ex.target_class}</span><span className="inline-flex items-center gap-1"><span className="size-2 rounded bg-neg" />opposes</span></div>
      <p className="text-[11px] text-mist">{ex.method}</p>
    </div>
  )
}

export function ConfusionMatrix({ labels, matrix }: { labels: string[]; matrix: number[][] }) {
  const max = Math.max(1, ...matrix.flat())
  return (
    <div className="overflow-x-auto">
      <table className="text-xs">
        <thead><tr><th className="p-1 text-left text-mist">true ↓ / pred →</th>{labels.map((l) => <th key={l} className="p-1 font-medium text-mist">{LANG_LABEL[l] ?? l}</th>)}</tr></thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}><th className="p-1 pr-3 text-left font-medium text-fog">{LANG_LABEL[labels[i]] ?? labels[i]}</th>
              {row.map((v, j) => <td key={j} className="p-0.5"><div className="grid h-10 min-w-14 place-items-center rounded-md font-mono tabular-nums text-white" style={{ background: i === j ? `rgba(34,211,238,${0.1 + 0.6 * v / max})` : `rgba(251,113,133,${v ? 0.06 + 0.5 * v / max : 0.02})` }}>{v}</div></td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
