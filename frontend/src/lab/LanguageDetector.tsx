import { useState } from 'react'
import { Badge, ErrorState, Panel } from '../components/ui'
import { api, type LIDResult } from '../lib/api'
import { PageHeader } from './LabLayout'
import { LANG_COLOR, LangLegend, ProbBars, TextInput, TokenStrip } from './shared'

const SCRIPT_COLOR: Record<string, string> = { Latin: '#a5b4fc', Kannada: '#22d3ee', Tamil: '#fbbf24', Malayalam: '#34d399', Devanagari: '#f472b6' }

export function LanguageDetector() {
  const [text, setText] = useState('Yeno hengidiyaa')
  const [res, setRes] = useState<LIDResult | null>(null)
  const [err, setErr] = useState<unknown>(null)
  const [loading, setLoading] = useState(false)
  const run = async () => {
    setLoading(true); setErr(null)
    try { setRes(await api.nlp.lid(text)) } catch (e) { setErr(e) } finally { setLoading(false) }
  }
  return (
    <div>
      <PageHeader eyebrow="Pillar A" title="Romanized Language Detector">
        Native scripts are identified deterministically from Unicode blocks. Latin-script text is classified by a character 1–4-gram TF-IDF + Logistic
        Regression model into English or Romanized Kannada, Tamil, Malayalam or Hindi. Probabilities are the model’s softmax outputs; predictions
        below 60% are flagged as uncertain. Short inputs (1–3 words) are much less reliable; see Research Analytics.
      </PageHeader>
      <TextInput value={text} onChange={setText} onSubmit={run} loading={loading} submitLabel="Identify" />
      {err != null && <div className="mt-6"><ErrorState error={err} onRetry={run} /></div>}
      {res && (
        <div className="mt-6 grid gap-6 lg:grid-cols-[1fr_1.2fr]">
          <Panel title="Sentence-level decision">
            <div className="flex items-baseline gap-3">
              <span className="h-display text-2xl font-semibold" style={{ color: LANG_COLOR[res.primary_language] ?? '#fff' }}>{res.primary_language_name}</span>
              {res.confidence != null && <span className="font-mono text-sm text-fog">{(res.confidence * 100).toFixed(1)}%</span>}
            </div>
            <div className="mt-2 flex flex-wrap gap-2">
              <Badge>decided by: {res.decision_source}</Badge>
              <Badge>{res.script_profile.text_type.replace('_', ' ')}</Badge>
              {res.uncertain && <Badge tone="warn">uncertain</Badge>}
              {res.unsupported && <Badge tone="neg">unsupported script</Badge>}
            </div>
            {res.sentence_probabilities ? (
              <div className="mt-5"><div className="eyebrow mb-2">Model probabilities (Latin-script portion)</div><ProbBars probs={res.sentence_probabilities} /></div>
            ) : <p className="mt-5 text-sm text-mist">No Latin-script words, so no model was needed.</p>}
          </Panel>
          <Panel title="Script profile" eyebrow={`${res.script_profile.letter_count} letters`}>
            <div className="flex h-4 overflow-hidden rounded-full">
              {Object.entries(res.script_profile.distribution).map(([s, v]) => <div key={s} style={{ width: `${v * 100}%`, background: SCRIPT_COLOR[s] ?? '#64748b' }} title={`${s} ${(v * 100).toFixed(1)}%`} />)}
            </div>
            <div className="mt-3 flex flex-wrap gap-3 text-xs text-fog">
              {Object.entries(res.script_profile.distribution).map(([s, v]) => <span key={s} className="inline-flex items-center gap-1.5"><span className="size-2 rounded-full" style={{ background: SCRIPT_COLOR[s] }} />{s} {(v * 100).toFixed(1)}%</span>)}
            </div>
            <div className="eyebrow mb-2 mt-6">Word-level tags</div>
            <TokenStrip tokens={res.tokens} />
            <div className="mt-3"><LangLegend /></div>
            <p className="mt-3 text-[11px] text-mist">Each Latin word chooses between English and the sentence’s Indic language (sentence-constrained decoding). Hover a word for the unconstrained word-model probabilities.</p>
          </Panel>
        </div>
      )}
    </div>
  )
}
