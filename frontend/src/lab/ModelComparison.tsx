import { useState } from 'react'
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Badge, ErrorState, Loading, Panel, useAsync } from '../components/ui'
import { api, type SentimentLang } from '../lib/api'
import { PageHeader } from './LabLayout'
import { ConfusionMatrix, ExplanationView, ModelControls, SentimentSummary, TextInput } from './shared'
import { chartAxis, chartGrid, chartTooltip } from './charts'

export function ModelComparison() {
  const models = useAsync(() => api.nlp.models())
  const [text, setText] = useState('Trailer semma mass, waiting for the movie')
  const [lang, setLang] = useState<SentimentLang>('auto')
  const [res, setRes] = useState<Awaited<ReturnType<typeof api.nlp.compare>> | null>(null)
  const [err, setErr] = useState<unknown>(null)
  const [loading, setLoading] = useState(false)
  const run = async () => {
    setLoading(true); setErr(null)
    try { setRes(await api.nlp.compare(text, lang)) } catch (e) { setErr(e) } finally { setLoading(false) }
  }
  const chart = models.data ? ['kn', 'ta', 'ml'].map((l) => {
    const row: Record<string, string | number> = { lang: { kn: 'Kannada', ta: 'Tamil', ml: 'Malayalam' }[l]! }
    models.data!.filter((m) => m.language === l).forEach((m) => { row[m.model] = m.test_metrics.macro_f1 })
    return row
  }) : []

  return (
    <div>
      <PageHeader eyebrow="Model Comparison" title="Logistic Regression vs Linear SVM">
        The deployed models: for each language and classifier, the feature configuration and C were selected on the dev set, and the scores below are from
        the untouched test split.
      </PageHeader>
      {models.loading ? <Loading /> : models.error ? <ErrorState error={models.error} onRetry={models.reload} /> : models.data && (
        <div className="grid gap-6 lg:grid-cols-[1fr_1.2fr]">
          <Panel title="Test macro-F1 of deployed models">
            <div className="h-64">
              <ResponsiveContainer>
                <BarChart data={chart} margin={{ left: -16 }}>
                  <CartesianGrid {...chartGrid} /><XAxis dataKey="lang" {...chartAxis} /><YAxis domain={[0, 1]} {...chartAxis} />
                  <Tooltip {...chartTooltip} formatter={(v: number) => v.toFixed(4)} /><Legend wrapperStyle={{ fontSize: 12 }} />
                  <Bar dataKey="Logistic Regression" fill="#22d3ee" radius={[4, 4, 0, 0]} /><Bar dataKey="Linear SVM" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Panel>
          <Panel title="Details">
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead className="text-left text-mist"><tr><th className="py-1.5">Model</th><th>Features</th><th>C</th><th className="text-right">Acc</th><th className="text-right">Macro-P</th><th className="text-right">Macro-R</th><th className="text-right">Macro-F1</th></tr></thead>
                <tbody className="divide-y divide-line">
                  {models.data.map((m) => (
                    <tr key={m.id} className="text-fog"><td className="py-2">{m.language_name} · {m.model}</td><td className="font-mono">{m.feature_config}</td><td className="font-mono">{m.C}</td>
                      <td className="text-right font-mono">{m.test_metrics.accuracy.toFixed(3)}</td><td className="text-right font-mono">{m.test_metrics.macro_precision.toFixed(3)}</td>
                      <td className="text-right font-mono">{m.test_metrics.macro_recall.toFixed(3)}</td><td className="text-right font-mono text-white">{m.test_metrics.macro_f1.toFixed(3)}</td></tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Panel>
          <Panel className="lg:col-span-2" title="Confusion matrices (test)">
            <div className="grid gap-6 md:grid-cols-2 xl:grid-cols-3">
              {models.data.map((m) => <div key={m.id}><div className="mb-2 text-xs text-fog">{m.language_name} · {m.model} <span className="text-mist">(n={m.test_metrics.n})</span></div><ConfusionMatrix labels={['negative', 'neutral', 'positive']} matrix={m.test_metrics.confusion_matrix} /></div>)}
            </div>
          </Panel>
        </div>
      )}

      <h2 className="h-display mb-4 mt-10 text-lg font-semibold text-white">Compare both models on your text</h2>
      <TextInput value={text} onChange={setText} onSubmit={run} loading={loading} submitLabel="Compare"><ModelControls lang={lang} setLang={setLang} /></TextInput>
      {err != null && <div className="mt-6"><ErrorState error={err} onRetry={run} /></div>}
      {res && !res.supported && <p className="mt-6 text-sm text-warn">Detected {res.routing.lid?.primary_language_name}; choose kn, ta or ml to compare anyway.</p>}
      {res?.supported && (
        <>
          <div className="mt-4">{res.agree ? <Badge tone="pos">Both models agree</Badge> : <Badge tone="warn">Models disagree</Badge>}</div>
          <div className="mt-4 grid gap-6 lg:grid-cols-2">
            {res.results.map((p) => (
              <Panel key={p.model_kind} title={p.model}>
                <SentimentSummary p={{ ...p, supported: true, routing: res.routing }} />
                {p.explanation && <div className="mt-5 border-t border-line pt-4"><ExplanationView ex={p.explanation} /></div>}
              </Panel>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
