import { useState } from 'react'
import { ErrorState, Panel } from '../components/ui'
import { api, type ModelKind, type SentimentLang, type SentimentPrediction } from '../lib/api'
import { PageHeader } from './LabLayout'
import { ExplanationView, ModelControls, SentimentSummary, TextInput } from './shared'

export function Sentiment() {
  const [text, setText] = useState('Movie thumba chennagide, must watch')
  const [model, setModel] = useState<ModelKind>('logreg')
  const [lang, setLang] = useState<SentimentLang>('auto')
  const [res, setRes] = useState<SentimentPrediction | null>(null)
  const [err, setErr] = useState<unknown>(null)
  const [loading, setLoading] = useState(false)
  const run = async (l: SentimentLang = lang) => {
    setLoading(true); setErr(null)
    try { setRes(await api.nlp.sentiment(text, model, l)) } catch (e) { setErr(e) } finally { setLoading(false) }
  }
  return (
    <div>
      <PageHeader eyebrow="Pillar C" title="Sentiment Analysis">
        Three classes (positive, negative, neutral), with one model per language trained on DravidianCodeMix YouTube comments. In Auto mode the
        language-identification system routes the text to the Kannada, Tamil or Malayalam model. Logistic Regression reports softmax
        probabilities; Linear SVM reports raw decision scores, which are not probabilities.
      </PageHeader>
      <TextInput value={text} onChange={setText} onSubmit={() => void run()} loading={loading}><ModelControls model={model} setModel={setModel} lang={lang} setLang={setLang} /></TextInput>
      {err != null && <div className="mt-6"><ErrorState error={err} onRetry={() => void run()} /></div>}
      {res && (
        <div className="mt-6 grid gap-6 lg:grid-cols-2">
          <Panel title="Prediction">
            <SentimentSummary p={res} />
            {!res.supported && (
              <div className="mt-4 flex flex-wrap gap-2 text-xs">
                {(['kn', 'ta', 'ml'] as const).map((l) => (
                  <button key={l} onClick={() => { setLang(l); void run(l) }} className="focus-ring rounded-lg border border-line px-3 py-1.5 text-fog hover:text-white">Run {l} model anyway</button>
                ))}
              </div>
            )}
          </Panel>
          {res.explanation && <Panel title="Why this prediction?"><ExplanationView ex={res.explanation} /></Panel>}
        </div>
      )}
    </div>
  )
}
