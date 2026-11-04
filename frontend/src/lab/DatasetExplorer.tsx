import { ChevronLeft, ChevronRight, ExternalLink } from 'lucide-react'
import { useState } from 'react'
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Badge, Button, Empty, ErrorState, Loading, Panel, Select, useAsync } from '../components/ui'
import { api } from '../lib/api'
import { chartAxis, chartGrid, chartTooltip } from './charts'
import { PageHeader } from './LabLayout'
import { SENT_COLOR } from './shared'

const LN: Record<string, string> = { kn: 'Kannada', ta: 'Tamil', ml: 'Malayalam' }

export function DatasetExplorer() {
  const ds = useAsync(() => api.nlp.datasets())
  const a = ds.data?.audit
  const labelChart = a ? ['kn', 'ta', 'ml'].map((l) => {
    const r: Record<string, string | number> = { lang: LN[l] }
    ;['train', 'dev', 'test'].forEach((s) => Object.entries(a.clean_label_distribution[`${l}/${s}`] ?? {}).forEach(([k, v]) => { r[k] = ((r[k] as number) || 0) + v }))
    return r
  }) : []
  const typeChart = a ? ['kn', 'ta', 'ml'].map((l) => ({ lang: LN[l], ...a.clean_text_type_distribution[l] })) : []

  return (
    <div>
      <PageHeader eyebrow="Dataset Explorer" title="Data used in the experiments">
        Every source below was downloaded and inspected for this project. Counts are computed from the files, not copied from the original papers.
      </PageHeader>
      {ds.loading ? <Loading /> : ds.error ? <ErrorState error={ds.error} onRetry={ds.reload} /> : ds.data && (
        <div className="space-y-6">
          <div className="grid gap-4 md:grid-cols-3">
            {ds.data.sources.map((s) => (
              <Panel key={s.name} title={s.name} eyebrow={s.licence}>
                <p className="text-sm text-fog">{s.use}</p>
                <p className="mt-2 text-xs text-mist">{s.citation}</p>
                <a href={s.url} target="_blank" rel="noreferrer" className="mt-3 inline-flex items-center gap-1 text-xs text-cyan-soft hover:underline">Source <ExternalLink className="size-3" /></a>
              </Panel>
            ))}
          </div>
          {a && (
            <>
              <div className="grid gap-6 lg:grid-cols-2">
                <Panel title="Sentiment labels after cleaning" eyebrow="train+dev+test">
                  <div className="h-64"><ResponsiveContainer><BarChart data={labelChart} margin={{ left: -8 }}>
                    <CartesianGrid {...chartGrid} /><XAxis dataKey="lang" {...chartAxis} /><YAxis {...chartAxis} /><Tooltip {...chartTooltip} /><Legend wrapperStyle={{ fontSize: 12 }} />
                    {['positive', 'neutral', 'negative'].map((k) => <Bar key={k} dataKey={k} stackId="a" fill={SENT_COLOR[k]} />)}
                  </BarChart></ResponsiveContainer></div>
                </Panel>
                <Panel title="Script type of comments" eyebrow="from Unicode, after cleaning">
                  <div className="h-64"><ResponsiveContainer><BarChart data={typeChart} margin={{ left: -8 }}>
                    <CartesianGrid {...chartGrid} /><XAxis dataKey="lang" {...chartAxis} /><YAxis {...chartAxis} /><Tooltip {...chartTooltip} /><Legend wrapperStyle={{ fontSize: 12 }} />
                    <Bar dataKey="romanized" stackId="a" fill="#a5b4fc" /><Bar dataKey="native" stackId="a" fill="#22d3ee" /><Bar dataKey="mixed_script" stackId="a" fill="#8b5cf6" />
                  </BarChart></ResponsiveContainer></div>
                </Panel>
              </div>
              <Panel title="Cleaning and leakage control (DravidianCodeMix)">
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead className="text-left text-mist"><tr><th className="py-1.5">Language</th><th className="text-right">Raw rows</th><th className="text-right">3-class rows</th><th className="text-right">Conflicting dups</th><th className="text-right">Within-split dups</th><th className="text-right">Cross-split leaks</th><th className="text-right">Final</th><th className="text-right">Code-mixed share*</th><th className="text-right">Mean CMI*</th></tr></thead>
                    <tbody className="divide-y divide-line font-mono text-fog">
                      {['kn', 'ta', 'ml'].map((l) => { const c = a.cleaning[l] as Record<string, number>; return (
                        <tr key={l}><td className="py-2 font-sans">{LN[l]}</td><td className="text-right">{c.input}</td><td className="text-right">{c.after_label_filter}</td><td className="text-right">{c.conflicting_duplicates_removed}</td><td className="text-right">{c.within_split_duplicates_removed}</td><td className="text-right">{c.cross_split_leakage_removed}</td><td className="text-right text-white">{c.final}</td><td className="text-right">{(a.clean_code_mixed_share[l] * 100).toFixed(1)}%</td><td className="text-right">{a.code_mixing_mean_cmi[l]}</td></tr>
                      ) })}
                    </tbody>
                  </table>
                </div>
                <p className="mt-3 text-xs text-mist">Label mapping: Positive → positive, Negative → negative, unknown_state → neutral; Mixed_feelings and not-&lt;language&gt; rows are excluded. *{a.note}</p>
              </Panel>
              <Panel title="Raw label strings (before mapping)">
                <div className="grid gap-4 md:grid-cols-3">
                  {['kn', 'ta', 'ml'].map((l) => (
                    <div key={l}><div className="eyebrow mb-2">{LN[l]} · {a.raw_audit[l].rows} rows</div>
                      <ul className="space-y-1 text-xs">{Object.entries(a.raw_audit[l].raw_labels).map(([k, v]) => <li key={k} className="flex justify-between font-mono text-fog"><span>{k || '(empty)'}</span><span>{v}</span></li>)}</ul></div>
                  ))}
                </div>
              </Panel>
            </>
          )}
          <SampleBrowser />
        </div>
      )}
    </div>
  )
}

function SampleBrowser() {
  const [f, setF] = useState({ lang: '', label: '', text_type: '', split: '', q: '' })
  const [offset, setOffset] = useState(0)
  const limit = 20
  const s = useAsync(() => api.nlp.samples({ ...f, offset, limit }), [f.lang, f.label, f.text_type, f.split, f.q, offset])
  const set = (k: keyof typeof f) => (v: string) => { setF({ ...f, [k]: v }); setOffset(0) }
  return (
    <Panel title="Browse cleaned DravidianCodeMix examples" eyebrow={s.data ? `${s.data.total.toLocaleString()} matching` : undefined}>
      <div className="mb-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-5">
        <Select label="Language" value={f.lang} onChange={set('lang')} options={[{ value: '', label: 'All' }, { value: 'kn', label: 'Kannada' }, { value: 'ta', label: 'Tamil' }, { value: 'ml', label: 'Malayalam' }]} />
        <Select label="Label" value={f.label} onChange={set('label')} options={[{ value: '', label: 'All' }, { value: 'positive', label: 'Positive' }, { value: 'neutral', label: 'Neutral' }, { value: 'negative', label: 'Negative' }]} />
        <Select label="Script" value={f.text_type} onChange={set('text_type')} options={[{ value: '', label: 'All' }, { value: 'romanized', label: 'Romanized' }, { value: 'native', label: 'Native' }, { value: 'mixed_script', label: 'Mixed script' }]} />
        <Select label="Split" value={f.split} onChange={set('split')} options={[{ value: '', label: 'All' }, { value: 'train', label: 'Train' }, { value: 'dev', label: 'Dev' }, { value: 'test', label: 'Test' }]} />
        <label className="block"><span className="eyebrow mb-1.5 block">Contains</span>
          <input defaultValue={f.q} maxLength={100} onKeyDown={(e) => e.key === 'Enter' && set('q')((e.target as HTMLInputElement).value)} onBlur={(e) => e.target.value !== f.q && set('q')(e.target.value)}
            placeholder="Enter to search" className="focus-ring h-10 w-full rounded-xl border border-line bg-ink-850 px-3 text-sm text-white" /></label>
      </div>
      {s.loading ? <Loading /> : s.error ? <ErrorState error={s.error} onRetry={s.reload} /> : !s.data?.rows.length ? <Empty title="No examples match these filters." /> : (
        <>
          <ul className="divide-y divide-line">
            {s.data.rows.map((r, i) => (
              <li key={i} className="flex flex-wrap items-start justify-between gap-3 py-2.5">
                <p className="min-w-0 flex-1 text-sm text-fog">{r.text}</p>
                <div className="flex shrink-0 gap-1.5"><Badge>{r.lang}</Badge><Badge>{r.split}</Badge><Badge>{r.text_type}</Badge>
                  <Badge tone={r.label === 'positive' ? 'pos' : r.label === 'negative' ? 'neg' : 'neu'}>{r.label}</Badge></div>
              </li>
            ))}
          </ul>
          <div className="mt-4 flex items-center justify-between text-xs text-mist">
            <span>{offset + 1}–{Math.min(offset + limit, s.data.total)} of {s.data.total.toLocaleString()}</span>
            <div className="flex gap-2">
              <Button size="sm" icon={<ChevronLeft className="size-4" />} disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - limit))}>Prev</Button>
              <Button size="sm" disabled={offset + limit >= s.data.total} onClick={() => setOffset(offset + limit)}>Next<ChevronRight className="size-4" /></Button>
            </div>
          </div>
          <p className="mt-3 text-[11px] text-mist">Comments are user-generated YouTube text from DravidianCodeMix (CC BY 4.0) and may contain offensive language.</p>
        </>
      )}
    </Panel>
  )
}
