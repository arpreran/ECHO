import { ArrowRight, Languages, Layers, Smile } from 'lucide-react'
import { Link } from 'react-router-dom'
import { ErrorState, Loading, Panel, pct, Stat, useAsync } from '../components/ui'
import { api } from '../lib/api'
import { PageHeader } from './LabLayout'

export function Overview() {
  const res = useAsync(() => api.nlp.results())
  const lid = res.data?.lid_results
  const sent = res.data?.sentiment_results
  const best = (lang: string) => {
    const runs = sent?.per_language[lang]?.runs
    if (!runs) return null
    return Object.values(runs).reduce((a, b) => (b.test.macro_f1 > a.test.macro_f1 ? b : a))
  }
  return (
    <div>
      <PageHeader eyebrow="ECHO NLP Lab" title="Script-independent language identification & code-mixing-aware sentiment">
        Research question: can script-independent language identification, language-aware preprocessing and character-level features improve sentiment
        classification for Romanized and code-mixed Dravidian languages? Everything here is computed by classical models trained in this project;
        no generative or pretrained neural model is used.
      </PageHeader>

      {res.loading ? <Loading /> : res.error ? <ErrorState error={res.error} onRetry={res.reload} /> : (
        <>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <Stat label="Romanized LID accuracy" value={pct(lid?.sentence_model.test_overall.accuracy)} hint={`${lid?.sentence_model.test_overall.n.toLocaleString()} held-out sentences, 5 classes`} />
            <Stat label="Off-the-shelf langid.py" value={pct(lid?.langid_baseline.test_overall.accuracy)} hint="Same test set; cannot output Romanized Indic" />
            <Stat label="Best Malayalam macro-F1" value={best('ml')?.test.macro_f1.toFixed(3) ?? '—'} hint={best('ml') ? `${best('ml')!.config} · ${best('ml')!.model}` : ''} />
            <Stat label="Best Kannada macro-F1" value={best('kn')?.test.macro_f1.toFixed(3) ?? '—'} hint={best('kn') ? `${best('kn')!.config} · ${best('kn')!.model}` : ''} />
          </div>
          <p className="mt-3 text-xs text-mist">“Best” is the highest test score among 12 configurations, shown for orientation only; deployed models were selected on the dev set. See Research Analytics for confidence intervals.</p>
        </>
      )}

      <div className="mt-8 grid gap-5 md:grid-cols-3">
        <Pillar to="/lab/language" icon={<Languages className="size-5" />} title="A · Romanized language identification"
          body="Unicode script rules for native scripts, character n-gram models for Latin script: English vs Romanized Kannada, Tamil, Malayalam (and Hindi), at sentence and word level." />
        <Pillar to="/lab/code-mixing" icon={<Layers className="size-5" />} title="B · Code-mixed text intelligence"
          body="Word-level language tags, switch points, Code-Mixing Index, M-index and I-index, script mixing, and language-aware spelling normalisation." />
        <Pillar to="/lab/sentiment" icon={<Smile className="size-5" />} title="C · Explainable sentiment"
          body="Word and character TF-IDF with Logistic Regression and Linear SVM, trained per language on DravidianCodeMix, with exact linear feature attributions." />
      </div>
      <Panel className="mt-8" title="Start exploring">
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          {[['/lab/workbench', 'Analyse a sentence end-to-end'], ['/lab/explainability', 'See why a prediction was made'], ['/lab/models', 'Compare LR vs SVM'],
            ['/lab/datasets', 'Browse the datasets'], ['/lab/analytics', 'Read the experiment results'], ['/lab/methodology', 'Methodology & limitations']].map(([to, l]) => (
            <Link key={to} to={to} className="focus-ring group flex items-center justify-between rounded-xl border border-line px-4 py-3 text-sm text-fog hover:border-violet-glow/40 hover:text-white">{l}<ArrowRight className="size-4 transition group-hover:translate-x-0.5" /></Link>
          ))}
        </div>
      </Panel>
    </div>
  )
}

function Pillar({ to, icon, title, body }: { to: string; icon: React.ReactNode; title: string; body: string }) {
  return (
    <Link to={to} className="focus-ring glass group rounded-2xl p-5 transition hover:border-violet-glow/40">
      <div className="mb-4 grid size-10 place-items-center rounded-xl bg-violet-glow/10 text-violet-300">{icon}</div>
      <h3 className="h-display text-sm font-semibold text-white">{title}</h3>
      <p className="mt-2 text-xs leading-relaxed text-mist">{body}</p>
    </Link>
  )
}
