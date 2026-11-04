import { useState } from 'react'
import { Badge, ErrorState, Panel, Stat } from '../components/ui'
import { api, type CodeMixResult } from '../lib/api'
import { PageHeader } from './LabLayout'
import { LANG_COLOR, LANG_LABEL, LangLegend, TextInput, TokenStrip } from './shared'

export function CodeMixing() {
  const [text, setText] = useState('ಇವತ್ತು class ತುಂಬಾ boring ide.')
  const [res, setRes] = useState<CodeMixResult | null>(null)
  const [err, setErr] = useState<unknown>(null)
  const [loading, setLoading] = useState(false)
  const run = async () => {
    setLoading(true); setErr(null)
    try { setRes(await api.nlp.codeMixing(text)) } catch (e) { setErr(e) } finally { setLoading(false) }
  }
  const s = res?.stats
  return (
    <div>
      <PageHeader eyebrow="Pillar B" title="Code-Mixing Intelligence">
        Word-level language tags (script rules + word LID) feed standard code-mixing measures: the Code-Mixing Index (Das & Gambäck, 2014),
        the M-index (Barnett et al., 2000) and the I-index (Guzmán et al., 2017). Script and language are separated: “ನಾನು” and “naanu”
        are the same language in different scripts. Word-level LID accuracy on held-out dictionary words is about 72%, so treat tags as estimates.
      </PageHeader>
      <TextInput value={text} onChange={setText} onSubmit={run} loading={loading} />
      {err != null && <div className="mt-6"><ErrorState error={err} onRetry={run} /></div>}
      {res && s && (
        <div className="mt-6 space-y-6">
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-5">
            <Stat label="Code-mixed" value={s.is_code_mixed ? 'Yes' : 'No'} hint={s.languages.map((l) => LANG_LABEL[l]).join(' + ')} />
            <Stat label="CMI" value={s.cmi} hint="0 = monolingual" />
            <Stat label="M-index" value={s.m_index} hint="0 = monolingual, 1 = balanced" />
            <Stat label="I-index" value={s.i_index} hint="switches / (n−1)" />
            <Stat label="Switch points" value={s.switch_points} hint={`${s.language_tokens} language tokens`} />
          </div>
          <Panel title="Word-level language map" actions={res.uncertain_tokens > 0 ? <Badge tone="warn">{res.uncertain_tokens} uncertain</Badge> : undefined}>
            <TokenStrip tokens={res.tokens} />
            <div className="mt-3"><LangLegend /></div>
          </Panel>
          <div className="grid gap-6 lg:grid-cols-2">
            <Panel title="Language distribution & spans">
              <div className="flex h-4 overflow-hidden rounded-full">
                {Object.entries(s.language_distribution).map(([l, v]) => <div key={l} style={{ width: `${v * 100}%`, background: LANG_COLOR[l] }} />)}
              </div>
              <div className="mt-2 flex flex-wrap gap-3 text-xs text-fog">{Object.entries(s.language_distribution).map(([l, v]) => <span key={l}>{LANG_LABEL[l]} {(v * 100).toFixed(0)}%</span>)}</div>
              <div className="eyebrow mb-2 mt-5">Monolingual spans in order</div>
              <div className="flex h-8 overflow-hidden rounded-lg">
                {s.spans.map((sp, i) => <div key={i} className="grid place-items-center border-r border-ink-950 text-[10px] font-medium text-ink-950" style={{ flex: sp.length, background: LANG_COLOR[sp.lang] }}>{sp.length}</div>)}
              </div>
              <div className="eyebrow mb-2 mt-5">Script of language tokens</div>
              <div className="flex flex-wrap gap-3 text-xs text-fog">{Object.entries(res.script_mixing).map(([k, v]) => <span key={k}>{k} {(v * 100).toFixed(0)}%</span>)}</div>
            </Panel>
            <Panel title="Language-aware normalisation">
              <dl className="space-y-3 text-sm">
                <div><dt className="eyebrow">Basic</dt><dd className="mt-1 font-mono text-fog">{res.normalized.basic}</dd></div>
                <div><dt className="eyebrow">Language-aware (Romanized Indic words folded)</dt><dd className="mt-1 font-mono text-cyan-soft">{res.normalized.language_aware}</dd></div>
              </dl>
              <div className="eyebrow mb-2 mt-5">Folding rules</div>
              <ol className="list-decimal space-y-0.5 pl-5 text-xs text-mist">{res.normalization_rules.map((r) => <li key={r}>{r}</li>)}</ol>
            </Panel>
          </div>
        </div>
      )}
    </div>
  )
}
