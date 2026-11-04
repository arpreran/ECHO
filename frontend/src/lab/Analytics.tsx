import { useState } from 'react'
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { Badge, ErrorState, Loading, Panel, Segmented, useAsync } from '../components/ui'
import { api, type Boot, type SentimentResults } from '../lib/api'
import { chartAxis, chartGrid, chartTooltip, SERIES } from './charts'
import { PageHeader } from './LabLayout'
import { ConfusionMatrix } from './shared'

const LN: Record<string, string> = { kn: 'Kannada', ta: 'Tamil', ml: 'Malayalam' }
const CONFIGS = ['word', 'char', 'word+char', 'word+char+norm', 'word+char+lang', 'word+char+norm+lang']

export function Analytics() {
  const r = useAsync(() => api.nlp.results())
  const [kind, setKind] = useState<'logreg' | 'linear_svm'>('logreg')
  if (r.loading) return <Loading />
  if (r.error) return <ErrorState error={r.error} onRetry={r.reload} />
  const lid = r.data?.lid_results
  const sent = r.data?.sentiment_results
  if (!lid || !sent) return <ErrorState error="Experiment results not found. Run experiments/train_lid.py and experiments/run_sentiment.py." />

  const ablation = ['kn', 'ta', 'ml'].map((l) => {
    const row: Record<string, string | number> = { lang: LN[l], majority: sent.per_language[l].majority_baseline.macro_f1 }
    CONFIGS.forEach((c) => { row[c] = sent.per_language[l].runs[`${c}|${kind}`].test.macro_f1 })
    return row
  })
  const byType = ['kn', 'ta', 'ml'].map((l) => {
    const t = sent.per_language[l].runs[`word+char|${kind}`].test.by_text_type
    return { lang: LN[l], romanized: t.romanized?.macro_f1, native: t.native?.macro_f1, mixed_script: t.mixed_script?.macro_f1 }
  })
  const lidSources = Object.keys(lid.sentence_model.test_by_source).map((s) => ({
    source: s, 'ECHO LID': lid.sentence_model.test_by_source[s].accuracy, 'langid.py': lid.langid_baseline.test_by_source[s].accuracy, n: lid.sentence_model.test_by_source[s].n }))
  const lengths = Object.entries(lid.sentence_model.accuracy_by_length).map(([k, v]) => ({ bucket: k, accuracy: v, n: lid.sentence_model.n_by_length[k] }))

  return (
    <div className="space-y-6">
      <PageHeader eyebrow="Research Analytics" title="Experiment results">
        All numbers are read from <code className="font-mono">experiments/results/*.json</code>, produced by the training scripts with seed {sent.seed}. Test splits were used once,
        after hyper-parameters were selected on dev. Confidence intervals come from 1,000 paired bootstrap resamples of the test set.
      </PageHeader>

      <Panel title="Romanized language identification: ECHO LID vs off-the-shelf langid.py" eyebrow="sentence-level accuracy by test source">
        <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
          <div className="h-64"><ResponsiveContainer><BarChart data={lidSources} margin={{ left: -8 }}>
            <CartesianGrid {...chartGrid} /><XAxis dataKey="source" {...chartAxis} /><YAxis domain={[0, 1]} {...chartAxis} />
            <Tooltip {...chartTooltip} formatter={(v: number) => `${(v * 100).toFixed(1)}%`} /><Legend wrapperStyle={{ fontSize: 12 }} />
            <Bar dataKey="ECHO LID" fill="#22d3ee" radius={[4, 4, 0, 0]} /><Bar dataKey="langid.py" fill="#64748b" radius={[4, 4, 0, 0]} />
          </BarChart></ResponsiveContainer></div>
          <div className="space-y-2 text-sm">
            <p className="text-fog">Overall: ECHO LID <b className="text-white">{(lid.sentence_model.test_overall.accuracy * 100).toFixed(1)}%</b> accuracy (macro-F1 {lid.sentence_model.test_overall.macro_f1.toFixed(3)}) vs langid.py <b className="text-white">{(lid.langid_baseline.test_overall.accuracy * 100).toFixed(1)}%</b> on n={lid.sentence_model.test_overall.n.toLocaleString()}.</p>
            <p className="text-xs text-mist">langid.py has no Romanized Indic classes: it labels such text as English or another Latin-script language, so its 0% on Dakshina and DravidianCodeMix is structural, not a tuning problem.</p>
            <p className="text-xs text-mist">Word-level LID (held-out dictionary words): accuracy {(lid.word_model.test.accuracy * 100).toFixed(1)}%, macro-F1 {lid.word_model.test.macro_f1.toFixed(3)}; {lid.word_data.ambiguous_train_strings.toLocaleString()} training strings are attested in more than one language.</p>
          </div>
        </div>
      </Panel>

      <div className="grid gap-6 lg:grid-cols-2">
        <Panel title="LID accuracy by input length" eyebrow="sentence model">
          <div className="h-56"><ResponsiveContainer><BarChart data={lengths} margin={{ left: -8 }}>
            <CartesianGrid {...chartGrid} /><XAxis dataKey="bucket" {...chartAxis} /><YAxis domain={[0, 1]} {...chartAxis} />
            <Tooltip {...chartTooltip} formatter={(v: number, _n, p) => [`${(v * 100).toFixed(1)}% (n=${p.payload.n})`, 'accuracy']} />
            <Bar dataKey="accuracy" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
          </BarChart></ResponsiveContainer></div>
          <p className="text-xs text-mist">Short chat-like inputs are the hardest case and the main limitation for conversational use.</p>
        </Panel>
        <Panel title="Sentence LID confusion matrix (test)">
          <ConfusionMatrix labels={lid.sentence_model.test_overall.confusion_matrix.labels} matrix={lid.sentence_model.test_overall.confusion_matrix.matrix} />
        </Panel>
      </div>

      <Panel title="Sentiment ablation: test macro-F1 by feature configuration" actions={<Segmented value={kind} onChange={setKind} options={[{ value: 'logreg', label: 'Logistic Regression' }, { value: 'linear_svm', label: 'Linear SVM' }]} />}>
        <div className="h-80"><ResponsiveContainer><BarChart data={ablation} margin={{ left: -8 }}>
          <CartesianGrid {...chartGrid} /><XAxis dataKey="lang" {...chartAxis} /><YAxis domain={[0, 0.9]} {...chartAxis} />
          <Tooltip {...chartTooltip} formatter={(v: number) => v.toFixed(4)} /><Legend wrapperStyle={{ fontSize: 11 }} />
          <Bar dataKey="majority" fill="#334155" radius={[3, 3, 0, 0]} />
          {CONFIGS.map((c, i) => <Bar key={c} dataKey={c} fill={SERIES[i]} radius={[3, 3, 0, 0]} />)}
        </BarChart></ResponsiveContainer></div>
        <SignificanceTable sent={sent} kind={kind} />
      </Panel>

      <div className="grid gap-6 lg:grid-cols-2">
        <Panel title="Macro-F1 by script type" eyebrow={`word+char · ${kind}`}>
          <div className="h-60"><ResponsiveContainer><BarChart data={byType} margin={{ left: -8 }}>
            <CartesianGrid {...chartGrid} /><XAxis dataKey="lang" {...chartAxis} /><YAxis domain={[0, 1]} {...chartAxis} />
            <Tooltip {...chartTooltip} formatter={(v: number) => v?.toFixed(4)} /><Legend wrapperStyle={{ fontSize: 12 }} />
            <Bar dataKey="romanized" fill="#a5b4fc" /><Bar dataKey="native" fill="#22d3ee" /><Bar dataKey="mixed_script" fill="#8b5cf6" />
          </BarChart></ResponsiveContainer></div>
          <p className="text-xs text-mist">Native-script Kannada has few examples, which explains its low score; small subsets have wide uncertainty.</p>
        </Panel>
        <Panel title="Pooled model vs language routing" eyebrow="word+char, all three test sets combined">
          <div className="overflow-x-auto"><table className="w-full text-xs">
            <thead className="text-left text-mist"><tr><th className="py-1.5">Classifier</th><th className="text-right">Pooled</th><th className="text-right">Oracle routing</th><th className="text-right">LID routing</th><th className="text-right">LID − pooled (95% CI)</th></tr></thead>
            <tbody className="divide-y divide-line font-mono text-fog">
              {Object.entries(sent.routing).map(([k, v]) => (
                <tr key={k}><td className="py-2 font-sans">{k === 'logreg' ? 'Logistic Regression' : 'Linear SVM'}</td><td className="text-right">{v.pooled.macro_f1.toFixed(4)}</td><td className="text-right">{v.oracle_language_routing.macro_f1.toFixed(4)}</td><td className="text-right">{v.lid_routing.macro_f1.toFixed(4)}</td><td className="text-right"><BootCell b={v['lid_routing vs pooled']} /></td></tr>
              ))}
            </tbody>
          </table></div>
          <p className="mt-3 text-xs text-mist">LID chose the dataset’s language for {(Object.values(sent.routing)[0].lid_routing_language_accuracy * 100).toFixed(1)}% of test comments. Routing to language-specific models did not beat a single pooled model.</p>
        </Panel>
      </div>
    </div>
  )
}

function BootCell({ b }: { b: Boot }) {
  const sig = b.ci95[0] > 0 || b.ci95[1] < 0
  return <span className={'whitespace-nowrap ' + (sig ? (b.delta_macro_f1 > 0 ? 'text-pos' : 'text-neg') : 'text-mist')}>{b.delta_macro_f1 >= 0 ? '+' : ''}{b.delta_macro_f1.toFixed(4)} [{b.ci95[0].toFixed(3)}, {b.ci95[1].toFixed(3)}]</span>
}

function SignificanceTable({ sent, kind }: { sent: SentimentResults; kind: string }) {
  const comps = Object.keys(sent.significance.kn[kind] as Record<string, Boot>)
  return (
    <div className="mt-6 overflow-x-auto">
      <div className="eyebrow mb-2">Paired bootstrap: change in macro-F1 (95% CI). Coloured if the CI excludes 0.</div>
      <table className="w-full text-xs">
        <thead className="text-left text-mist"><tr><th className="py-1.5">Comparison</th>{['kn', 'ta', 'ml'].map((l) => <th key={l} className="text-right">{LN[l]}</th>)}</tr></thead>
        <tbody className="divide-y divide-line font-mono">
          {comps.map((c) => (
            <tr key={c}><td className="py-2 font-sans text-fog">{c}</td>{['kn', 'ta', 'ml'].map((l) => <td key={l} className="text-right"><BootCell b={(sent.significance[l][kind] as Record<string, Boot>)[c]} /></td>)}</tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-[11px] text-mist">These are 15 comparisons per classifier without multiple-comparison correction; isolated significant results should be read with caution. <Badge>test n: {['kn', 'ta', 'ml'].map((l) => `${l} ${sent.per_language[l].majority_baseline.n.toLocaleString()}`).join(' · ')}</Badge></p>
    </div>
  )
}
