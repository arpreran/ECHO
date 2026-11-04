import { Panel } from '../components/ui'
import { PageHeader } from './LabLayout'

const STEPS = [
  ['Script identification', 'Each character is mapped to a Unicode block (Kannada U+0C80–0CFF, Tamil U+0B80–0BFF, Malayalam U+0D00–0D7F, Devanagari U+0900–097F, Latin). A text is native, romanized, mixed-script or other. Native-script words get their language directly from the script.'],
  ['Sentence-level Romanized LID', 'Latin-script words of a text → character 1–4-gram TF-IDF (word-bounded, sublinear tf) → multinomial Logistic Regression over {English, Romanized Kannada, Tamil, Malayalam, Hindi}. Training: Dakshina romanized sentences (dev split), DravidianCodeMix Latin-script comments (train split, “not-<language>” rows excluded), Tatoeba English. Groups were capped at 8,000 sentences; C was chosen on a 10% validation slice.'],
  ['Word-level LID', 'Character 1–5-gram TF-IDF + Logistic Regression trained on Dakshina’s attested romanization lexicons (train split) and English vocabulary from Tatoeba (words seen ≥ 3 times, hash-split). At inference, each Latin word chooses between English and the Indic language of its sentence (sentence-constrained decoding); confidence below 0.6 is flagged as uncertain.'],
  ['Code-mixing statistics', 'Over language-tagged tokens (numbers, emoji, punctuation and URLs are language-independent): CMI = 100·(1 − max_lang / (n − u)); M-index = (1 − Σp²)/((k − 1)Σp²); I-index = switch points/(n − 1). Script variants of one language count as one language.'],
  ['Normalisation', 'Basic: NFC, lowercase, URLs/mentions/numbers replaced with placeholders, character elongation inside words reduced to two. Language-aware: words tagged as Romanized Indic also get a spelling-variant key (collapse repeated letters; ph→f, w→v, q→k, ck→k, z→j); joiners are removed from native-script words.'],
  ['Sentiment features', 'Word uni+bigram TF-IDF (min_df 2) and/or character 2–5-gram TF-IDF (word-bounded, ≤300k features), on basic or language-aware text, optionally plus LID-derived features (share of English, Romanized and native-script tokens, CMI, I-index, log switch points, script type, code-mixed flag).'],
  ['Classifiers', 'Logistic Regression (lbfgs, class_weight=balanced) and Linear SVM (LinearSVC, class_weight=balanced). C was selected by dev macro-F1 from {0.3, 1, 3, 10} (LR) or {0.03, 0.1, 0.3, 1} (SVM). One model per language, plus a pooled model for the routing experiment.'],
  ['Evaluation', 'Official DravidianCodeMix train/dev/test splits after removing excluded labels, conflicting and exact duplicates, and test/dev items whose normalised text appears in train (or test items appearing in dev). Metrics: accuracy, macro precision/recall/F1, weighted F1, confusion matrices, per script type and per LID-estimated code-mixing status. Paired bootstrap (1,000 resamples) for differences. Seed 42 everywhere.'],
  ['Explainability', 'For a linear model, score_k(x) = Σ_j x_j·w_kj + b_k exactly. Feature contributions x_j·w_kj are reported for the predicted class; character n-gram contributions are shared among the tokens containing them, so token scores plus non-token features plus bias reproduce the model score (checked by an automated test).'],
]

const LIMITS = [
  'Sentiment data are YouTube movie-trailer comments; performance on other domains (including companion chat) is not measured.',
  'The 3-class task drops Mixed_feelings and maps unknown_state to neutral. That is a modelling choice, not the original annotation scheme.',
  'Kannada has only 561 test comments after cleaning, so its confidence intervals are wide. Native-script Kannada is particularly scarce.',
  'Word-level LID is evaluated on dictionary words out of context; no gold word-level annotation of code-mixed sentences was available. Code-mixing statistics therefore inherit word-LID errors (~72% accuracy on held-out words).',
  'Sentence LID accuracy falls to 72.8% for 1–3-word inputs. The app adds the word model for such inputs (weight chosen on validation data), which raises held-out short-input accuracy to 74.2%. Informal Kannada such as “Yeno hengidiyaa” is still misclassified because those forms are missing from the training data.',
  'Dataset-language labels for DravidianCodeMix comments are inherited from the file they came from; some comments are mostly English.',
  'Many configurations were compared without multiple-comparison correction.',
]

export function Methodology() {
  return (
    <div className="space-y-6">
      <PageHeader eyebrow="Methodology" title="How the NLP Lab works">
        Only Python, scikit-learn, NumPy, pandas, Unicode processing and regular expressions are used. No generative AI, external AI API, LLM, or
        pretrained transformer (MuRIL, XLM-R, IndicBERT, …) is involved in any prediction, experiment or metric. The off-the-shelf langid.py baseline is a
        Naive Bayes classifier.
      </PageHeader>
      <Panel title="Pipeline">
        <ol className="space-y-4">
          {STEPS.map(([t, d], i) => (
            <li key={t} className="grid grid-cols-[32px_1fr] gap-3">
              <span className="grid size-7 place-items-center rounded-lg bg-violet-glow/15 font-mono text-xs text-violet-300">{i + 1}</span>
              <div><h3 className="text-sm font-semibold text-white">{t}</h3><p className="mt-1 text-sm leading-relaxed text-mist">{d}</p></div>
            </li>
          ))}
        </ol>
      </Panel>
      <div className="grid gap-6 lg:grid-cols-2">
        <Panel title="Reproduce">
          <pre className="overflow-x-auto rounded-xl bg-ink-900 p-4 font-mono text-xs text-fog">{`.venv\\Scripts\\python experiments\\train_lid.py
.venv\\Scripts\\python experiments\\run_sentiment.py`}</pre>
          <p className="mt-3 text-xs text-mist">Outputs: models/lid/*.joblib, models/sentiment/*.joblib, experiments/results/lid_results.json, sentiment_results.json, dataset_audit.json.</p>
        </Panel>
        <Panel title="Known limitations">
          <ul className="list-disc space-y-1.5 pl-5 text-sm text-mist">{LIMITS.map((l) => <li key={l}>{l}</li>)}</ul>
        </Panel>
      </div>
    </div>
  )
}
