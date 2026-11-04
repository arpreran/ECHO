import { useState } from 'react'
import { Badge, ErrorState, Loading, Panel, Segmented, Select, useAsync } from '../components/ui'
import { api, type ModelKind, type SentimentLang, type SentimentPrediction } from '../lib/api'
import { PageHeader } from './LabLayout'
import { ExplanationView, ModelControls, SENT_COLOR, TextInput } from './shared'

export function Explainability() {
  const [text, setText] = useState('Padam kollam, pakshe climax mosham aayi')
  const [model, setModel] = useState<ModelKind>('logreg')
  const [lang, setLang] = useState<SentimentLang>('auto')
  const [res, setRes] = useState<SentimentPrediction | null>(null)
  const [err, setErr] = useState<unknown>(null)
  const [loading, setLoading] = useState(false)
  const run = async () => {
    setLoading(true); setErr(null)
    try { setRes(await api.nlp.sentiment(text, model, lang)) } catch (e) { setErr(e) } finally { setLoading(false) }
  }
  return (
    <div>
      <PageHeader eyebrow="Explainability Explorer" title="Local and global explanations">
        Both classifiers are linear, so a class score is exactly the sum over active features of TF-IDF value × weight, plus a bias. Local explanations
        decompose that sum; global explanations list the largest weights per class. No surrogate model or sampling is involved.
      </PageHeader>
      <TextInput value={text} onChange={setText} onSubmit={run} loading={loading} submitLabel="Explain"><ModelControls model={model} setModel={setModel} lang={lang} setLang={setLang} /></TextInput>
      {err != null && <div className="mt-6"><ErrorState error={err} onRetry={run} /></div>}
      {res && !res.supported && <p className="mt-6 text-sm text-warn">{res.message}</p>}
      {res?.explanation && (
        <Panel className="mt-6" title={<span>Local explanation · predicted <span className="capitalize" style={{ color: SENT_COLOR[res.label!] }}>{res.label}</span></span>}
          actions={<><Badge tone="violet">{res.model}</Badge><Badge>{res.language_name} · {res.feature_config}</Badge></>}>
          <ExplanationView ex={res.explanation} />
        </Panel>
      )}
      <GlobalFeatures />
    </div>
  )
}

function GlobalFeatures() {
  const models = useAsync(() => api.nlp.models())
  const [id, setId] = useState('kn_logreg')
  const [cls, setCls] = useState<'positive' | 'negative' | 'neutral'>('positive')
  const feats = useAsync(() => api.nlp.features(id, 20), [id])
  return (
    <Panel className="mt-6" title="Global explanation: strongest features per class" actions={
      models.data && <Select value={id} onChange={setId} options={models.data.map((m) => ({ value: m.id, label: `${m.language_name} · ${m.model}` }))} />}>
      <div className="mb-4"><Segmented value={cls} onChange={setCls} options={[{ value: 'positive', label: 'Positive' }, { value: 'neutral', label: 'Neutral' }, { value: 'negative', label: 'Negative' }]} /></div>
      {feats.loading ? <Loading /> : feats.error ? <ErrorState error={feats.error} onRetry={feats.reload} /> : feats.data && (
        <div className="grid gap-6 md:grid-cols-2">
          {(['positive', 'negative'] as const).map((dir) => (
            <div key={dir}>
              <div className="eyebrow mb-2">{dir === 'positive' ? `Pushes towards ${cls}` : `Pushes away from ${cls}`}</div>
              <ol className="space-y-1">
                {feats.data!.features[cls][dir].map((f, i) => (
                  <li key={i} className="flex items-center justify-between gap-3 rounded-lg bg-white/[0.02] px-3 py-1.5 text-xs">
                    <span className="truncate font-mono text-fog"><span className="mr-2 text-mist">{f.block}</span>“{f.feature}”</span>
                    <span className="font-mono tabular-nums" style={{ color: f.weight >= 0 ? '#34d399' : '#fb7185' }}>{f.weight.toFixed(3)}</span>
                  </li>
                ))}
              </ol>
            </div>
          ))}
        </div>
      )}
      <p className="mt-4 text-[11px] text-mist">Character n-grams appear with their word-boundary spaces. Features with large weights can still be rare in real text, and weights are not causal claims about language.</p>
    </Panel>
  )
}
