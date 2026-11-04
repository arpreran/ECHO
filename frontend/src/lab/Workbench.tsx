import { useState } from 'react'
import { Badge, ErrorState, Panel } from '../components/ui'
import { api, type ModelKind, type SentimentLang } from '../lib/api'
import { PageHeader } from './LabLayout'
import { ExplanationView, LangLegend, ModelControls, ProbBars, SentimentSummary, TextInput, TokenStrip } from './shared'

type Result = Awaited<ReturnType<typeof api.nlp.analyze>>

export function Workbench() {
  const [text, setText] = useState('Naan office ge hogbeku today.')
  const [model, setModel] = useState<ModelKind>('logreg')
  const [lang, setLang] = useState<SentimentLang>('auto')
  const [res, setRes] = useState<Result | null>(null)
  const [err, setErr] = useState<unknown>(null)
  const [loading, setLoading] = useState(false)
  const run = async () => {
    setLoading(true); setErr(null)
    try { setRes(await api.nlp.analyze(text, model, lang)) } catch (e) { setErr(e) } finally { setLoading(false) }
  }
  return (
    <div>
      <PageHeader eyebrow="Analysis Workbench" title="Full pipeline on one input">Script profile → language identification → word-level tagging and code-mixing statistics → language-aware normalisation → sentiment with explanation.</PageHeader>
      <TextInput value={text} onChange={setText} onSubmit={run} loading={loading}><ModelControls model={model} setModel={setModel} lang={lang} setLang={setLang} /></TextInput>
      {err != null && <div className="mt-6"><ErrorState error={err} onRetry={run} /></div>}
      {res && (
        <div className="mt-6 grid gap-6 lg:grid-cols-2">
          <Panel title="1 · Language identification" eyebrow={res.language_identification.decision_source}>
            <div className="flex flex-wrap items-center gap-2">
              <span className="h-display text-xl font-semibold text-white">{res.language_identification.primary_language_name}</span>
              {res.language_identification.confidence != null && <Badge tone="cyan">{(res.language_identification.confidence * 100).toFixed(1)}%</Badge>}
              <Badge>{res.language_identification.script_profile.text_type.replace('_', ' ')}</Badge>
              {res.language_identification.uncertain && <Badge tone="warn">uncertain</Badge>}
            </div>
            {res.language_identification.sentence_probabilities && <div className="mt-4"><ProbBars probs={res.language_identification.sentence_probabilities} /></div>}
          </Panel>
          <Panel title="2 · Code-mixing" eyebrow={res.code_mixing.stats.is_code_mixed ? 'code-mixed' : 'single language'}>
            <TokenStrip tokens={res.code_mixing.tokens} />
            <div className="mt-3"><LangLegend /></div>
            <div className="mt-4 grid grid-cols-3 gap-2 text-center text-xs">
              <div className="rounded-lg border border-line p-2"><div className="text-mist">CMI</div><div className="font-mono text-white">{res.code_mixing.stats.cmi}</div></div>
              <div className="rounded-lg border border-line p-2"><div className="text-mist">Switch points</div><div className="font-mono text-white">{res.code_mixing.stats.switch_points}</div></div>
              <div className="rounded-lg border border-line p-2"><div className="text-mist">I-index</div><div className="font-mono text-white">{res.code_mixing.stats.i_index}</div></div>
            </div>
          </Panel>
          <Panel title="3 · Normalisation">
            <dl className="space-y-3 text-sm">
              <div><dt className="eyebrow">Basic</dt><dd className="mt-1 font-mono text-fog">{res.code_mixing.normalized.basic}</dd></div>
              <div><dt className="eyebrow">Language-aware</dt><dd className="mt-1 font-mono text-cyan-soft">{res.code_mixing.normalized.language_aware}</dd></div>
            </dl>
          </Panel>
          <Panel title="4 · Sentiment"><SentimentSummary p={res.sentiment} /></Panel>
          {res.sentiment.explanation && <Panel className="lg:col-span-2" title="5 · Explanation"><ExplanationView ex={res.sentiment.explanation} /></Panel>}
        </div>
      )}
    </div>
  )
}
