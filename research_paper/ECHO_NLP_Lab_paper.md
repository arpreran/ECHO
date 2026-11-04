# Script-Independent Language Identification and Code-Mixing-Aware Sentiment Analysis for Romanized and Code-Mixed Dravidian Languages

*ECHO NLP Lab — research paper*

---

## Abstract

Speakers of Kannada, Tamil and Malayalam often write their languages in the Latin alphabet and mix them with English inside one sentence. Off-the-shelf language identifiers treat such text as English or another European language, and sentiment classifiers trained without regard to script and language mixing may lose information. We study three questions with classical, fully reproducible methods: (i) whether a script-independent language identification (LID) pipeline can separate English from Romanized Dravidian languages; (ii) how much code-mixing real social-media data contain; and (iii) whether language-aware preprocessing, LID-derived features and character n-grams improve three-class sentiment classification.

Our LID system combines deterministic Unicode script rules with character n-gram TF-IDF + logistic-regression models. It is trained on the Dakshina romanization resources, DravidianCodeMix comments and Tatoeba English. On 27,501 held-out sentences it reaches 95.9% accuracy (macro-F1 0.961), whereas the off-the-shelf langid.py tool cannot label any Romanized Kannada, Tamil or Malayalam sentence correctly. Accuracy drops to 72.8% for inputs of one to three words.

In a separate follow-up experiment, we fuse the word-level model into the sentence-level decision for such short inputs, with the fusion weight tuned on validation data. This raises short-input test accuracy to 74.2% (+1.4 points, 95% CI [0.7, 2.0]). The fusion is used only in the deployed application; every other result in this paper uses the unfused model.

For sentiment, we train word, character and combined TF-IDF features with Logistic Regression and Linear SVM on DravidianCodeMix (Kannada, Tamil, Malayalam) after de-duplication and leakage removal. We evaluate on the official test splits with paired bootstrap confidence intervals.
- Character n-grams significantly improve macro-F1 over word features for Malayalam (+3.1 to +4.8 points). They do not do so significantly for Kannada or Tamil.
- Language-aware spelling normalisation and LID-derived code-mixing features give no consistent significant gain. One configuration (Kannada, Linear SVM, both components) improves by +3.3 points (95% CI [0.2, 6.4]).
- Routing each comment to a language-specific model chosen by our LID matches, but does not beat, a single pooled model.

We release the implementation, trained models and all result files as part of the ECHO NLP Lab.

**Keywords:** code-mixing, Romanized text, language identification, Dravidian languages, Kannada, Tamil, Malayalam, sentiment analysis, character n-grams, explainability, classical machine learning

---

## 1. Introduction

Social-media users in South India routinely write Kannada, Tamil and Malayalam in the Latin script ("Romanized" or transliterated text) and switch between these languages and English within a sentence. Examples are *"Naan office ge hogbeku today"* (Kannada–English) and *"Padam kollam, pakshe climax mosham aayi"* (Malayalam).

Such text breaks two common assumptions in NLP pipelines:
- **Script implies language.** Latin script is taken to mean English.
- **Each sentence is monolingual.**

A language detector that relies on script, or on character statistics learned only from standard orthographies, labels Romanized Dravidian text as English or as an unrelated Latin-script language. Downstream components then process it with the wrong assumptions.

Sentiment analysis of code-mixed Dravidian text has received attention through the DravidianCodeMix corpora and shared tasks [1, 2, 3, 4, 5]. Much of that work focuses on classification accuracy. This paper isolates the contribution of language-related processing to classical models, asking whether making a classical sentiment model "aware" of script and language — through script-independent LID, language-aware normalisation and code-mixing features — measurably helps. We deliberately restrict ourselves to transparent methods: TF-IDF features and linear classifiers. These are cheap, reproducible, and exactly explainable.

## 2. Problem statement

Given a short user-generated text that may be written in a native Dravidian script, in Latin script, or in both, and may mix a Dravidian language with English, the task has three parts:

1. Identify the script and the language of the text and of each word, distinguishing English from Romanized Kannada, Tamil and Malayalam.
2. Quantify the degree of code-mixing.
3. Classify the text's sentiment as positive, negative or neutral, with an explanation of the prediction.

## 3. Research gap

- **Off-the-shelf LID lacks Romanized classes.** Widely used identifiers such as langid.py [6] are trained on standard orthographies and offer no class for Romanized Dravidian text.
- **Sentiment work rarely isolates language-aware components.** Work on code-mixed Dravidian sentiment [1, 2, 3, 5] reports classifier comparisons. It seldom measures, with controlled ablations and significance testing, whether explicit language identification or spelling normalisation helps classical models.
- **Script and language are often conflated.** Corpora are described as "code-mixed" without measuring how much mixing they contain or how evaluation differs across native-script, Romanized and mixed-script subsets.

## 4. Research objectives

- **O1.** Build a script-independent LID system that separates English from Romanized Kannada, Tamil and Malayalam (and Romanized Hindi, for robustness) at sentence and word level, and compare it with an off-the-shelf identifier.
- **O2.** Measure code-mixing in DravidianCodeMix with standard indices computed from word-level tags.
- **O3.** Evaluate word, character and combined TF-IDF features, language-aware normalisation and LID-derived features for three-class sentiment, with Logistic Regression and Linear SVM, per language and per text type.
- **O4.** Provide exact, model-faithful explanations of each prediction.

## 5. Research questions

- **RQ1.** How accurately can classical character n-gram models identify Romanized Dravidian languages, compared with an off-the-shelf identifier, and how does accuracy depend on input length?
- **RQ2.** Do character-level features improve sentiment classification over word-level features?
- **RQ3.** Do language-aware preprocessing and LID-derived code-mixing features improve sentiment classification?
- **RQ4.** Does routing a text to a language-specific sentiment model by predicted language outperform a single pooled multilingual model?
- **RQ5.** How does performance differ across native-script, Romanized and mixed-script subsets?

## 6. Related work

**Code-mixed Dravidian sentiment resources.**
- Chakravarthi et al. built sentiment corpora of YouTube comments for Tamil–English [2] and Malayalam–English [3].
- Hande et al. released KanCMD for Kannada–English [4].
- These were consolidated in DravidianCodeMix [1], which adds offensive-language labels and baseline results.
- Shared tasks at FIRE 2020 [5] and DravidianLangTech 2021 [7] used these data.
- Code-mixed sentiment has also been studied for Hindi–English and Spanish–English, for example in SemEval-2020 Task 9 [8] and with sub-word models by Joshi et al. [9].

**Language identification of code-mixed and Romanized text.**
- The First Shared Task on Language Identification in Code-Switched Data [10] established word-level LID benchmarks.
- King and Abney [11] labelled word languages in mixed documents with weakly supervised methods.
- Das and Gambäck [12] identified word-level languages in code-mixed Indian social-media text and proposed the Code-Mixing Index (CMI).
- Bali et al. [13] analysed English–Hindi code-mixing on Facebook.
- For Romanized South Asian text, the Dakshina dataset [14] provides native-script Wikipedia text, attested romanization lexicons and human-romanized sentences for twelve languages.
- langid.py [6] is a widely used off-the-shelf identifier based on a Naive Bayes model over byte n-grams.

**Measuring code-mixing.**
- Beyond the CMI [12], the M-index derived from the LIDES coding conventions [15] measures the balance between languages.
- The I-index of Guzmán et al. [16] measures how often switches occur.

**Classical text classification and evaluation.**
- Linear models over word and character n-grams remain strong baselines for sentiment [17].
- We use scikit-learn [18] and LIBLINEAR [19].
- For significance testing we follow the paired bootstrap of Koehn [20] and the analysis of Berg-Kirkpatrick et al. [21].

## 7. Dataset description

All datasets were downloaded, their licences were checked, and their files were inspected for this study. All counts below are computed from the files.

### 7.1 DravidianCodeMix-2020 (sentiment)

DravidianCodeMix [1] (CC BY 4.0) contains YouTube comments on movie trailers annotated for sentiment, with official train/dev/test splits. The release uses two file layouts: `text;label;` for Kannada and Malayalam, and `text<TAB>label` for Tamil. Lines are therefore parsed from the right.

Raw label strings are *Positive, Negative, unknown_state, Mixed_feelings* and *not-<language>*. Tamil has one empty label. We define a three-class task:
- *Positive* → positive, *Negative* → negative.
- *unknown_state* (no sentiment expressed) → neutral.
- *Mixed_feelings* and *not-<language>* rows are excluded.

The mapping of unknown_state to neutral is our modelling choice.

**Cleaning and leakage control** (per language):
- Remove exact duplicates (after lower-casing and whitespace normalisation) within a split. If duplicates disagree in label, remove all copies.
- Remove dev and test items whose normalised text occurs in train, and test items that occur in dev.

Table 1 reports the effect.

**Table 1. DravidianCodeMix sentiment data before and after cleaning.**

| | Kannada | Tamil | Malayalam |
|---|---:|---:|---:|
| Raw rows (train+dev+test) | 7,671 | 44,020 | 19,616 |
| Rows with a 3-class label | 5,844 | 37,005 | 17,009 |
| Conflicting duplicates removed | 48 | 48 | 133 |
| Within-split duplicates removed | 184 | 226 | 441 |
| Cross-split leaks removed | 36 | 106 | 249 |
| **Final** (train / dev / test) | 4,462 / 553 / 561 | 29,364 / 3,642 / 3,619 | 13,054 / 1,577 / 1,555 |
| Test: positive / neutral / negative | 336 / 83 / 142 | 2,456 / 619 / 544 | 712 / 581 / 262 |
| Script type (all splits): Romanized / native / mixed | 3,800 / 976 / 800 | 29,389 / 5,502 / 1,730 | 8,808 / 4,347 / 3,029 |
| Code-mixed share (LID-estimated) | 66.2% | 78.1% | 65.6% |
| Mean CMI (LID-estimated) | 21.45 | 25.10 | 18.17 |

Script type is derived deterministically from Unicode code-point blocks. A comment is *native* if all its letters are in one Indic script, *Romanized* if all are Latin, and *mixed-script* if both occur. Code-mixed share and CMI are estimated with our word-level LID and inherit its errors (Section 15.1). The dataset contains no word-level language annotation.

### 7.2 Dakshina (romanization)

Dakshina v1.0 [14] (CC BY-SA 4.0) provides, for each language:
- a romanization lexicon of native-script words with attested Latin spellings, split into train/dev/test by native word;
- 10,000 Wikipedia sentences romanized by native speakers (5,000 dev, 5,000 test).

We use the Kannada, Tamil, Malayalam and Hindi portions. After lower-casing and keeping alphabetic strings, the lexicons yield the following distinct romanized training words:

| Language | Distinct romanized training words |
|---|---:|
| Kannada | 49,097 |
| Tamil | 66,286 |
| Malayalam | 55,948 |
| Hindi | 41,344 |

Of all training strings, 14,361 are attested in more than one language, which bounds word-level accuracy.

### 7.3 Tatoeba English

English sentences from the Tatoeba project (CC BY 2.0 FR) supply the English class. We use ASCII-only sentences.
- **Word level:** words occurring at least three times form the English vocabulary (32,878 training words), split by a hash of the word.
- **Sentence level:** 12,000 sentences are randomly sampled for training (capped at 8,000, as described below) and 3,000 for testing.

### 7.4 Annotation note

No resource we found provides gold word-level language labels for Romanized code-mixed *Kannada*. Rather than fabricate labels, we evaluate word-level LID on held-out dictionary words (Section 15.1) and treat in-context word tagging as an estimate.

A protocol for creating such data is given in Appendix A. Its outline:
- two annotators per sentence, using the tag set {en, kn, ta, ml, hi, univ, ne (named entity), mixed, other};
- adjudication of disagreements and reporting of Cohen's κ.

## 8. Methodology

The study has three stages, matching the three pillars of the ECHO NLP Lab:
- **Script-independent LID** (Sections 12, 13.1, 13.2 and 13.4): Unicode script identification, then sentence-level and word-level language models for Latin-script text.
- **Code-mixing analysis** (Sections 7.1 and 15.3): word tags are turned into code-mixing statistics.
- **Sentiment classification** (Sections 10–15): per-language TF-IDF + linear models in six feature configurations, evaluated with ablations and significance tests.

Hyper-parameters are always selected on development data, and each test set is used once per configuration.

## 9. System architecture

The NLP Lab is a Python package (`backend/nlp_lab`) exposed through a FastAPI service and a React interface.

```
text ─► Unicode script profiler ─► tokenizer
          │                          │
          │ native-script words ─────┤ (language = script)
          ▼                          ▼
   sentence LID (Latin part) ─► word LID + sentence-constrained decoding
          │                          │
          │                          ├─► code-mixing statistics (CMI, M, I, switches)
          │                          └─► language-aware normalisation
          ▼                                         │
   language routing (kn / ta / ml) ─► TF-IDF features (+ LID features) ─► LR / SVM
                                                    │
                                                    └─► exact linear explanation
```

Every component is classical. No generative model, external AI service or pretrained neural language model (e.g., MuRIL, XLM-R, IndicBERT) is used in any prediction, experiment or metric. The training scripts are `experiments/train_lid.py` and `experiments/run_sentiment.py`, and all outputs are written to `experiments/results/*.json`.

## 10. Preprocessing

**Tokenisation.** A regular expression extracts:
- URLs, @-mentions and hashtags;
- word tokens, including Indic combining marks and zero-width joiners, so that Indic words stay whole;
- single punctuation, emoji or symbol characters.

**Basic normalisation** (used by every configuration):
- Unicode NFC;
- lower-casing;
- URLs, mentions and digit strings replaced by `<url>`, `<user>` and `<num>`;
- runs of three or more identical characters inside a word reduced to two (*superrrr* → *superr*).

**Language-aware normalisation.** Tokens that the word-level LID tags as Romanized Kannada, Tamil, Malayalam or Hindi are replaced by a spelling-variant key. The rules are applied in order:
1. Collapse any repeated letter to a single letter.
2. ph→f, w→v, q→k, ck→k, z→j.

English tokens are left untouched. Zero-width joiners are removed from native-script tokens. The rules are deliberately small and deterministic, and they merge variants such as *hengidiyaa / hengidiyaaa / hengidiya*. Some merges are harmful, for example Tamil *zh* and *j*; Section 16 discusses this.

## 11. Feature engineering

| Configuration | Description |
|---|---|
| `word` | Word uni- and bigram TF-IDF on basic-normalised tokens (min_df = 2, sublinear tf) |
| `char` | Character 2–5-gram TF-IDF within word boundaries (`char_wb`) on basic-normalised text (min_df = 2, ≤ 300,000 features) |
| `word+char` | Concatenation of the two |
| `word+char+norm` | Both TF-IDF blocks computed on language-aware normalised text |
| `word+char+lang` | `word+char` plus eight LID-derived features: shares of English, Romanized-Indic and native-script tokens; CMI/100; I-index; log(1 + switch points); script type (one-hot); code-mixed flag |
| `word+char+norm+lang` | Both additions |

Vectorisers are fitted on training data only.

## 12. Algorithms

**Logistic Regression** (multinomial, L2, lbfgs solver, class_weight = balanced). C is chosen from {0.3, 1, 3, 10}. Probabilities are the softmax outputs.

**Linear SVM** (LinearSVC [19], squared hinge, one-vs-rest, class_weight = balanced). C is chosen from {0.03, 0.1, 0.3, 1}. The SVM yields decision scores, not probabilities, and the interface reports them as such.

**LID models.**
- The sentence model uses character 1–4-gram TF-IDF (`char_wb`, min_df = 3, ≤ 300,000 features) with multinomial logistic regression, C ∈ {3, 10, 30}.
- The word model uses character 1–5-gram TF-IDF with logistic regression, C ∈ {1, 3, 10}.

**Sentence-constrained word decoding.** Code-mixed comments usually mix English with one Indic language. Each Latin-script word therefore chooses between English and the Indic language *L\** implied by the sentence. *L\** is the native script present in the sentence, otherwise the most probable non-English class of the sentence model. The word's two probabilities are renormalised, and tags with confidence below 0.6 are flagged as uncertain. Native-script tokens receive the language of their script. Numbers, punctuation, emoji and URLs are language-independent.

**Code-mixing indices.** Let *n* be the number of tokens and *u* the number of language-independent tokens. Let *w_max* be the number of tokens in the most frequent language, and *p_j* the share of language *j* among the *k* languages present. Script variants of one language (e.g. `kn` and `kn_rom`) count as the same language.
- CMI = 100 · (1 − *w_max* / (*n* − *u*)), or 0 if *n* = *u* [12].
- M-index = (1 − Σ *p_j*²) / ((*k* − 1) Σ *p_j*²) [15].
- I-index = (number of language switches between consecutive language tokens) / (*n* − *u* − 1) [16].

A text is *code-mixed* if it contains tokens of at least two languages.

**Explanation.** For a linear model the score of class *k* is exactly *s_k(x) = Σ_j x_j w_kj + b_k*. We report the contribution *x_j w_kj* of every active feature to the predicted class:
- Word n-gram contributions are divided among the tokens they cover.
- Character n-gram contributions are divided among the tokens that contain them, in proportion to occurrence counts.
- LID features are reported separately.

Token contributions, non-token contributions and the bias sum exactly to the model score. An automated test checks this for both classifiers.

## 13. Experimental setup

### 13.1 Sentence-level LID

**Training data** (Latin-script part of each text):
- Dakshina romanized sentences, dev half;
- DravidianCodeMix Romanized comments from the train split, excluding *not-<language>* rows and labelled with the file's language;
- Tatoeba English.

Each (source, language) group is capped at 8,000 sentences.

**Test data:**
- Dakshina romanized test sentences;
- DravidianCodeMix Romanized test comments;
- 3,000 held-out Tatoeba sentences.

Test sentences identical to a training sentence are removed. C is chosen on a random 10% of the training data.

**Baseline.** langid.py 1.1.6 [6], used off the shelf. Its outputs `kn`, `ta`, `ml`, `hi` are mapped to the Romanized classes, and all other outputs to *other*.

### 13.2 Word-level LID

Training uses the Dakshina lexicon train splits and the English vocabulary train split. C is chosen on the dev splits, and the model is evaluated on the test splits. Dakshina splits by native word, so no native word appears in both training and test.

### 13.3 Sentiment

- One model per language and per (configuration, classifier) pair: 6 × 2 × 3 = 36 test runs.
- C is selected by dev macro-F1.
- A majority-class baseline is included.

**Routing experiment (RQ4).** On the union of the three test sets we compare, using `word+char` features:
- **Pooled:** one model trained on all three training sets.
- **Oracle routing:** each comment goes to the language-specific model of its source file.
- **LID routing:** each comment goes to the language-specific model chosen by our LID's primary-language prediction, with the pooled model used when the prediction is English or Hindi.

The random seed is 42 throughout.

### 13.4 Follow-up: short-input fusion (deployment only)

This experiment was designed *after* the main results showed weak short-input LID (Section 15.1). It is reported separately and is not used in any other experiment.

**Fused score.** For an input with at most three Latin-script words:

log p(c) = log p_sentence(c) + α · mean_t log p_word(c | t)

**Protocol.**
1. The 90/10 training/validation split of Section 13.1 is recreated exactly (same seed and random-number sequence).
2. A sentence model is fitted on the 90% part with the selected C.
3. α is chosen from {0, 0.25, 0.5, 1, 1.5, 2, 3} by macro-F1 on the 406 short validation inputs.
4. The final sentence and word models with that α are evaluated once on the 2,642 short test inputs.
5. The difference is tested with a paired bootstrap (1,000 resamples) against α = 0.

No test data influences any choice, no examples are added, and no individual input is special-cased (`experiments/short_text_lid.py`).

### 13.5 Software

Python 3.11, scikit-learn 1.9.1 [18], NumPy, pandas, FastAPI. The trained models and JSON result files are included in the repository.

## 14. Evaluation metrics

- **Metrics:** accuracy, macro-averaged precision, recall and F1 (the primary metric, because classes are imbalanced), weighted F1, per-class scores and confusion matrices.
- **Breakdowns:** sentiment results are also reported by script type (Romanized, native, mixed-script) and by LID-estimated code-mixing status (code-mixed vs monolingual).
- **Significance:** for pairs of systems we use a paired bootstrap over test items with 1,000 resamples [20, 21] and report the difference in macro-F1 with a 95% percentile interval. An interval excluding 0 is called significant.
- **Multiple comparisons:** we make 15 such comparisons per classifier without correction, so isolated significant results must be read with caution.

## 15. Results

### 15.1 Language identification (RQ1)

**Table 2. Sentence-level LID on held-out test data (accuracy).**

| Test source | n | ECHO LID | langid.py |
|---|---:|---:|---:|
| Dakshina romanized sentences (kn, ta, ml, hi) | 19,871 | **95.68%** | 0.00% |
| DravidianCodeMix Romanized comments (kn, ta, ml) | 4,630 | **95.01%** | 0.00% |
| Tatoeba English | 3,000 | **99.13%** | 96.83% |
| **Overall** | 27,501 | **95.94%** (macro-F1 0.961) | 10.56% (macro-F1 0.094) |

**Per-class results.** Per-class F1 of the sentence model:
- English 0.975
- Romanized Kannada 0.953
- Romanized Tamil 0.962
- Romanized Malayalam 0.951
- Romanized Hindi 0.963

The most frequent confusions are Tamil→Malayalam (184 sentences) and Tamil→Kannada (140).

**The langid.py baseline.** langid.py labelled 4,443 of the 24,501 Romanized-Indic test sentences as English and all others but one as non-English, non-Indic languages. It output an Indic language only once (one Kannada sentence labelled Malayalam). This failure is structural: the tool has no Romanized-Indic classes.

**Effect of input length.** Accuracy depends strongly on length.

| Input length | Accuracy | n |
|---|---:|---:|
| 1–3 words | 72.79% | 2,642 |
| 4–7 words | 96.60% | 8,775 |
| 8–15 words | 99.23% | 11,076 |
| 16+ words | 99.72% | 5,008 |

**Word-level LID** on held-out dictionary words (n = 25,956) reaches 72.33% accuracy (macro-F1 0.721), with per-class F1 as follows:

| Class | F1 |
|---|---:|
| English | 0.738 |
| Romanized Kannada | 0.762 |
| Romanized Tamil | 0.767 |
| Romanized Malayalam | 0.703 |
| Romanized Hindi | 0.634 |

This evaluates isolated words, the hardest setting. 14,361 training strings are attested in more than one language.

**Qualitative probes.** We report these unlabelled examples verbatim; they are not scored:
- *"Neenga epdi irukeenga"* → Romanized Tamil (0.999).
- *"Sukham ano"* → Romanized Malayalam (0.929).
- *"How are you?"* → English (0.998).
- *"Naan office ge hogbeku today."* → Romanized Kannada (0.824), with *office* and *today* tagged English.
- *"Yeno hengidiyaa"* → Romanized Tamil (0.648): a misclassification, since the colloquial Kannada form *yeno* is rare in Dakshina's Wikipedia-derived data.
- *"Bro, kal college jaana hai"* (Hindi–English) → Romanized Kannada (0.941): also wrong. The Hindi class is trained only on formal Dakshina sentences.

**Short-input fusion (follow-up, Section 13.4).** On validation, α = 0.25 performed best: macro-F1 rose from 0.783 (α = 0) to 0.803. With larger α it fell, for example to 0.723 at α = 3. On the held-out short test inputs (n = 2,642):

| Short test inputs (1–3 words) | Accuracy | Macro-F1 |
|---|---:|---:|
| Sentence model (all other reported results) | 72.79% | 0.736 |
| Sentence + word fusion, α = 0.25 (deployed app) | 74.19% | 0.746 |

The difference is +1.40 points (95% CI [0.72, 2.04]). By source, accuracy changed as follows:

| Source | n | Before | After |
|---|---:|---:|---:|
| Dakshina | 2,366 | 71.3% | 72.6% |
| DravidianCodeMix | 127 | 81.1% | 81.1% |
| Tatoeba | 149 | 89.3% | 93.3% |

The gain comes mainly from fewer Tamil→Malayalam and Tamil→Kannada errors. Romanized Kannada recall on short inputs fell slightly, from 427 to 422 of 516 correct. The improvement is real but small.

**Failure case: "Yeno hengidiyaa".** This informal Kannada greeting remains misclassified with or without fusion.
- **Why the models disagree:**
  - The sentence model gives Tamil 0.648 and Kannada 0.240.
  - The word model rates *hengidiyaa* as Kannada (0.64), but finds *yeno* ambiguous (English 0.37, Malayalam 0.24, Kannada 0.08).
  - At α = 0.25, the word evidence is too weak to reverse the sentence decision.
- **Why the training data cannot fix it:**
  - Neither *hengidiyaa* nor the standard spelling *hegiddiya* occurs in any training source.
  - *yeno* occurs twice in the Kannada comments and twice in the Tamil comments.
  - Kannada contributes fewer than half as many informal training comments as Tamil (3,575 vs 8,000).
  - The word ending *-iyaa* is 14 times more frequent in the Tamil comments than in the Kannada ones.
- **Contrast with longer input:** *"Yeno maga hegiddiya"* is correctly identified as Romanized Kannada (0.973).

We did not hard-code this example or add unverified training text. It remains a documented failure that more informal Romanized Kannada data would be needed to fix.

### 15.2 Sentiment: feature configurations and classifiers (RQ2, RQ3)

**Table 3. Test macro-F1 for each configuration** (LR = Logistic Regression, SVM = Linear SVM). C was selected on dev. The best value per language is in bold.

| Configuration | kn LR | kn SVM | ta LR | ta SVM | ml LR | ml SVM |
|---|---:|---:|---:|---:|---:|---:|
| Majority class | 0.250 | 0.250 | 0.270 | 0.270 | 0.209 | 0.209 |
| `word` | 0.659 | 0.667 | 0.609 | **0.624** | 0.726 | 0.734 |
| `char` | 0.680 | 0.693 | 0.602 | 0.622 | **0.774** | 0.765 |
| `word+char` | 0.690 | 0.678 | 0.609 | 0.622 | 0.761 | 0.767 |
| `word+char+norm` | 0.688 | 0.689 | 0.613 | 0.621 | 0.771 | 0.771 |
| `word+char+lang` | 0.685 | 0.694 | 0.611 | 0.623 | 0.758 | 0.762 |
| `word+char+norm+lang` | 0.708 | **0.711** | 0.619 | 0.617 | 0.771 | 0.769 |

All models are far above the majority baseline.

**Per-class F1** (`word+char`, LR):

| Language | Positive | Neutral | Negative |
|---|---:|---:|---:|
| Kannada | 0.817 | 0.580 | 0.673 |
| Tamil | 0.790 | 0.510 | 0.527 |
| Malayalam | 0.820 | 0.777 | 0.686 |

The neutral class (from *unknown_state*) is the hardest for Kannada and Tamil.

**Table 4. Paired bootstrap differences in test macro-F1, with 95% CIs.** † marks intervals that exclude 0. ‡ marks a borderline case: the upper bound rounds to zero and exactly 97.5% of resamples are ≤ 0.

| Comparison | Classifier | Kannada | Tamil | Malayalam |
|---|---|---|---|---|
| `char` − `word` | LR | +0.022 [−0.021, 0.062] | −0.007 [−0.022, 0.009] | **+0.048 [0.025, 0.070]†** |
| | SVM | +0.026 [−0.015, 0.061] | −0.002 [−0.016, 0.014] | **+0.031 [0.010, 0.053]†** |
| `word+char` − `word` | LR | +0.032 [−0.003, 0.066] | −0.000 [−0.013, 0.012] | **+0.035 [0.015, 0.057]†** |
| | SVM | +0.011 [−0.018, 0.042] | −0.002 [−0.014, 0.010] | **+0.034 [0.015, 0.052]†** |
| `+norm` − `word+char` | LR | −0.003 [−0.023, 0.018] | +0.004 [−0.005, 0.015] | +0.009 [−0.004, 0.022] |
| | SVM | +0.010 [−0.011, 0.031] | −0.001 [−0.007, 0.005] | +0.003 [−0.009, 0.016] |
| `+lang` − `word+char` | LR | −0.005 [−0.031, 0.022] | +0.002 [−0.006, 0.011] | −0.003 [−0.010, 0.004] |
| | SVM | +0.016 [−0.012, 0.044] | +0.002 [−0.005, 0.008] | −0.006 [−0.013, −0.000]‡ |
| `+norm+lang` − `word+char` | LR | +0.017 [−0.007, 0.042] | +0.010 [−0.001, 0.019] | +0.010 [−0.004, 0.024] |
| | SVM | **+0.033 [0.002, 0.064]†** | −0.004 [−0.012, 0.003] | +0.002 [−0.011, 0.015] |
| SVM − LR (`word+char`) | — | −0.012 [−0.039, 0.015] | **+0.013 [0.001, 0.025]†** | +0.006 [−0.009, 0.020] |

**RQ2 (character features).**
- **Malayalam:** character n-grams help significantly, with both classifiers and both as a replacement for and an addition to word features.
- **Kannada:** the point estimates favour character features (+1.1 to +3.2), but the 561-comment test set is too small for significance.
- **Tamil:** no effect.

**RQ3 (language-aware components).**
- Language-aware normalisation alone never produces a significant change.
- LID features alone never produce a significant gain. One case is a borderline *decrease* (Malayalam, Linear SVM, −0.6), whose interval reaches zero, so we do not count it as significant.
- The combination produces one significant gain: Kannada, Linear SVM, +3.3 (CI [0.2, 6.4]). Point estimates for the combination are positive in 5 of 6 language–classifier pairs, but the other four positive estimates are not significant.
- Given the number of comparisons, these isolated results are suggestive rather than conclusive. The answer to RQ3 is negative for consistent improvement.

**Classifiers.** Linear SVM is significantly better than LR only for Tamil (+1.3).

**Deployed models.** These were selected by dev macro-F1:
- Kannada: `char` (LR) and `word+char+lang` (SVM).
- Tamil: `word+char+norm+lang` for both classifiers.
- Malayalam: `word+char+norm+lang` for both.

Their test macro-F1 is 0.680 / 0.694 (kn, LR / SVM), 0.619 / 0.617 (ta) and 0.771 / 0.769 (ml). Dev-based selection does not always pick the configuration that is best on test (compare Table 3), as expected with small differences.

### 15.3 Script type and code-mixing (RQ5)

**Table 5. Test macro-F1 by script type** (`word+char`, LR). n in parentheses.

| | Romanized | Native | Mixed-script |
|---|---|---|---|
| Kannada | 0.694 (356) | 0.458 (116) | 0.657 (89) |
| Tamil | 0.612 (2,887) | 0.590 (537) | 0.571 (195) |
| Malayalam | 0.749 (835) | 0.752 (426) | 0.806 (294) |

**Script type.**
- **Kannada native-script comments are much harder** (0.458). They are only 17.5% of the cleaned Kannada data (976 of 5,576 comments), so the model sees few native-script examples.
- **Malayalam behaves differently:** it has substantial native and mixed-script data and performs as well or better on them.

**Code-mixing status.** By LID-estimated code-mixing status (`word+char`, LR):

| Language | Code-mixed comments | Monolingual comments |
|---|---:|---:|
| Kannada | 0.674 (n = 352) | 0.695 (n = 209) |
| Tamil | 0.613 (n = 2,836) | 0.589 (n = 783) |
| Malayalam | 0.770 (n = 1,016) | 0.746 (n = 539) |

There is no consistent penalty for code-mixed comments.

### 15.4 Pooled vs routed models (RQ4)

**Table 6. Macro-F1 on the combined test set** (n = 5,735, `word+char`).

| | Pooled | Oracle routing | LID routing |
|---|---:|---:|---:|
| Logistic Regression | 0.671 | 0.669 | 0.666 |
| Linear SVM | 0.682 | 0.684 | 0.681 |

- **LID routing accuracy:** our LID selected the source language for 96.3% of comments. 51 comments fell back to the pooled model.
- **LID routing vs pooled:** no significant difference (LR −0.005 [−0.014, 0.004]; SVM −0.002 [−0.010, 0.008]).
- **Oracle vs LID routing:** a small but significant cost of LID errors (LR +0.003 [0.000, 0.006]; SVM +0.003 [0.000, 0.006]).

## 16. Discussion

**Script-independent LID is necessary, and easy at sentence length.** The total failure of langid.py on Romanized Dravidian text confirms the motivating problem: a conventional identifier cannot be the first stage of a pipeline for this data. Character n-gram models trained on attested romanizations solve the problem for sentences of four or more words (≥ 96.6%). Short inputs remain hard (72.8% for 1–3 words). This matters for conversational use, where messages like *"Yeno hengidiyaa"* are typical. Our errors on that example and on a Hindi–English probe point to a register gap: Dakshina is formal Wikipedia text, and only Kannada, Tamil and Malayalam receive informal training data from DravidianCodeMix. Adding word-level evidence for short inputs helps a little (+1.4 points). It cannot recover informal forms that appear in neither training source, so the remaining problem is one of data rather than modelling.

**Character features help where spelling varies most.** Malayalam benefits clearly from character n-grams. A plausible explanation, which we have not tested directly, is that Malayalam has long agglutinative words and many Romanized spelling variants, which character n-grams share across variants while word features treat them as distinct.

**Explicit language awareness adds little on top of character n-grams.** Character n-grams already capture much of what our folding rules and LID features encode. The folding rules are also crude: collapsing doubled letters erases phonemic length distinctions (e.g. Tamil *pal* vs *paal*). Within a single-language model, LID features mostly describe *how* a comment is written rather than *what* it expresses. The one significant gain (Kannada, SVM) comes from the smallest test set and may not replicate. It also moved from +3.6 to +3.3 when the word-LID model was made deterministic (Appendix B), which shows how sensitive this result is.

**Language-specific models are not better than a pooled model here.** The three corpora share domain (movie-trailer comments) and much English vocabulary, so a pooled model can learn shared cues. Routing remains useful in a deployed system: it allows per-language models, explanations and calibration, and it costs little because LID errors are rare at comment length.

**Explainability.** Linear models allow faithful, exact decompositions. In the ECHO NLP Lab interface, users can see which words, character n-grams and code-mixing features drove a prediction. Character n-gram attributions are shared among the tokens containing them, which is exact in sum but a simplification per token.

## 17. Limitations

- **Domain.** All sentiment data are YouTube movie-trailer comments; results may not transfer to conversation or other domains.
- **Label mapping.** The three-class task drops *Mixed_feelings* and maps *unknown_state* to neutral, so our numbers are not directly comparable with five-class results on the same corpus [1, 5].
- **Small Kannada test set.** 561 comments give wide confidence intervals; native-script Kannada has 116 test comments.
- **Word-level evaluation.** Word-level LID is evaluated out of context on dictionary words. No gold word-level annotation of code-mixed Kannada sentences exists in our resources. Code-mixing statistics and the `+lang` features therefore inherit word-LID errors.
- **Noisy dataset-language labels.** Labels for DravidianCodeMix comments come from the file they appear in. Some comments are largely English, which adds noise to both LID training and evaluation.
- **Short-input weakness.** LID accuracy on 1–3-word inputs is 72.8% (74.2% with the deployed fusion). Colloquial Kannada and Hindi forms are under-represented; "Yeno hengidiyaa" remains misclassified as Tamil.
- **Statistical caveats.** We ran many comparisons without multiple-comparison correction and used a single random seed. Linear models with fixed data are deterministic, but the bootstrap intervals do not capture variation due to the choice of splits.
- **Baseline scope.** We compared against one off-the-shelf identifier (langid.py) only.

## 18. Conclusion

We built and evaluated a fully classical, reproducible pipeline for Romanized and code-mixed Dravidian text.
- **LID:** script-independent LID with character n-grams separates English from Romanized Kannada, Tamil and Malayalam with about 96% sentence accuracy, where an off-the-shelf identifier fails completely. Very short inputs remain an open problem. Validated word-level fusion improves them only modestly (72.8% → 74.2%).
- **Sentiment features:** character-level features significantly improve Malayalam sentiment classification. Explicit language-aware normalisation and code-mixing features did not yield consistent significant improvements over word-and-character TF-IDF.
- **Routing:** LID-based routing to language-specific models is accurate but not better than a pooled model.

These findings suggest that, for classical models, the most useful form of "language awareness" is sub-word representation rather than explicit language tags. They also show that gold word-level annotation of Romanized Kannada is the main missing resource.

**Future work:**
- annotating a word-level Romanized Kannada corpus following Appendix A;
- adding informal Romanized Hindi and Kannada data for short-text LID;
- testing data-driven rather than hand-written spelling normalisation.

## 19. References

1. B. R. Chakravarthi, R. Priyadharshini, V. Muralidaran, N. Jose, S. Suryawanshi, E. Sherly, and J. P. McCrae. 2022. DravidianCodeMix: sentiment analysis and offensive language identification dataset for Dravidian languages in code-mixed text. *Language Resources and Evaluation*, 56(3):765–806. https://doi.org/10.1007/s10579-022-09583-7
2. B. R. Chakravarthi, V. Muralidaran, R. Priyadharshini, and J. P. McCrae. 2020. Corpus Creation for Sentiment Analysis in Code-Mixed Tamil-English Text. In *Proceedings of the 1st Joint Workshop on Spoken Language Technologies for Under-resourced languages (SLTU) and Collaboration and Computing for Under-Resourced Languages (CCURL)*, pages 202–210, Marseille, France. https://aclanthology.org/2020.sltu-1.28/
3. B. R. Chakravarthi, N. Jose, S. Suryawanshi, E. Sherly, and J. P. McCrae. 2020. A Sentiment Analysis Dataset for Code-Mixed Malayalam-English. In *Proceedings of SLTU-CCURL 2020*, pages 177–184, Marseille, France. https://aclanthology.org/2020.sltu-1.25/
4. A. Hande, R. Priyadharshini, and B. R. Chakravarthi. 2020. KanCMD: Kannada CodeMixed Dataset for Sentiment Analysis and Offensive Language Detection. In *Proceedings of the Third Workshop on Computational Modeling of People's Opinions, Personality, and Emotion's in Social Media*, pages 54–63, Barcelona, Spain (Online). https://aclanthology.org/2020.peoples-1.6/
5. B. R. Chakravarthi, R. Priyadharshini, V. Muralidaran, S. Suryawanshi, N. Jose, E. Sherly, and J. P. McCrae. 2020. Overview of the track on Sentiment Analysis for Dravidian Languages in Code-Mixed Text. In *Forum for Information Retrieval Evaluation (FIRE 2020)*, pages 21–24. ACM. https://doi.org/10.1145/3441501.3441515
6. M. Lui and T. Baldwin. 2012. langid.py: An Off-the-shelf Language Identification Tool. In *Proceedings of the ACL 2012 System Demonstrations*, pages 25–30, Jeju Island, Korea. https://aclanthology.org/P12-3005/
7. B. R. Chakravarthi, R. Priyadharshini, N. Jose, A. Kumar M, T. Mandl, P. K. Kumaresan, R. Ponnusamy, H. R L, J. P. McCrae, and E. Sherly. 2021. Findings of the Shared Task on Offensive Language Identification in Tamil, Malayalam, and Kannada. In *Proceedings of the First Workshop on Speech and Language Technologies for Dravidian Languages*, pages 133–145, Kyiv. https://aclanthology.org/2021.dravidianlangtech-1.17/
8. P. Patwa, G. Aguilar, S. Kar, S. Pandey, S. PYKL, B. Gambäck, T. Chakraborty, T. Solorio, and A. Das. 2020. SemEval-2020 Task 9: Overview of Sentiment Analysis of Code-Mixed Tweets. In *Proceedings of the Fourteenth Workshop on Semantic Evaluation*, pages 774–790, Barcelona (online). https://doi.org/10.18653/v1/2020.semeval-1.100
9. A. Joshi, A. Prabhu, M. Shrivastava, and V. Varma. 2016. Towards Sub-Word Level Compositions for Sentiment Analysis of Hindi-English Code Mixed Text. In *Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: Technical Papers*, pages 2482–2491, Osaka, Japan. https://aclanthology.org/C16-1234/
10. T. Solorio, E. Blair, S. Maharjan, S. Bethard, M. Diab, M. Ghoneim, A. Hawwari, F. AlGhamdi, J. Hirschberg, A. Chang, and P. Fung. 2014. Overview for the First Shared Task on Language Identification in Code-Switched Data. In *Proceedings of the First Workshop on Computational Approaches to Code Switching*, pages 62–72, Doha, Qatar. https://doi.org/10.3115/v1/W14-3907
11. B. King and S. Abney. 2013. Labeling the Languages of Words in Mixed-Language Documents using Weakly Supervised Methods. In *Proceedings of NAACL-HLT 2013*, pages 1110–1119, Atlanta, Georgia. https://aclanthology.org/N13-1131/
12. A. Das and B. Gambäck. 2014. Identifying Languages at the Word Level in Code-Mixed Indian Social Media Text. In *Proceedings of the 11th International Conference on Natural Language Processing*, pages 378–387, Goa, India. https://aclanthology.org/W14-5152/
13. K. Bali, J. Sharma, M. Choudhury, and Y. Vyas. 2014. "I am borrowing ya mixing?" An Analysis of English-Hindi Code Mixing in Facebook. In *Proceedings of the First Workshop on Computational Approaches to Code Switching*, pages 116–126, Doha, Qatar. https://doi.org/10.3115/v1/W14-3914
14. B. Roark, L. Wolf-Sonkin, C. Kirov, S. J. Mielke, C. Johny, I. Demirsahin, and K. Hall. 2020. Processing South Asian Languages Written in the Latin Script: the Dakshina Dataset. In *Proceedings of the Twelfth Language Resources and Evaluation Conference*, pages 2413–2423, Marseille, France. https://aclanthology.org/2020.lrec-1.294/
15. R. Barnett, E. Codó, E. Eppler, M. Forcadell, P. Gardner-Chloros, R. van Hout, M. Moyer, M. C. Torras, M. T. Turell, M. Sebba, M. Starren, and S. Wensing. 2000. The LIDES Coding Manual. *International Journal of Bilingualism*, 4(2):131–132. https://doi.org/10.1177/13670069000040020101
16. G. Guzmán, J. Ricard, J. Serigos, B. E. Bullock, and A. J. Toribio. 2017. Metrics for Modeling Code-Switching Across Corpora. In *Proceedings of Interspeech 2017*, pages 67–71. https://doi.org/10.21437/Interspeech.2017-1429
17. S. Wang and C. Manning. 2012. Baselines and Bigrams: Simple, Good Sentiment and Topic Classification. In *Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)*, pages 90–94, Jeju Island, Korea. https://aclanthology.org/P12-2018/
18. F. Pedregosa, G. Varoquaux, A. Gramfort, V. Michel, B. Thirion, O. Grisel, M. Blondel, P. Prettenhofer, R. Weiss, V. Dubourg, J. Vanderplas, A. Passos, D. Cournapeau, M. Brucher, M. Perrot, and É. Duchesnay. 2011. Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12:2825–2830. https://www.jmlr.org/papers/v12/pedregosa11a.html
19. R.-E. Fan, K.-W. Chang, C.-J. Hsieh, X.-R. Wang, and C.-J. Lin. 2008. LIBLINEAR: A Library for Large Linear Classification. *Journal of Machine Learning Research*, 9:1871–1874. https://www.jmlr.org/papers/v9/fan08a.html
20. P. Koehn. 2004. Statistical Significance Tests for Machine Translation Evaluation. In *Proceedings of the 2004 Conference on Empirical Methods in Natural Language Processing*, pages 388–395, Barcelona, Spain. https://aclanthology.org/W04-3250/
21. T. Berg-Kirkpatrick, D. Burkett, and D. Klein. 2012. An Empirical Investigation of Statistical Significance in NLP. In *Proceedings of the 2012 Joint Conference on Empirical Methods in Natural Language Processing and Computational Natural Language Learning*, pages 995–1005, Jeju Island, Korea. https://aclanthology.org/D12-1091/

Each reference was checked against the ACL Anthology record, the Crossref DOI metadata or the JMLR article page. The BibTeX is in `research_paper/references.bib`. The Tatoeba corpus (https://tatoeba.org, CC BY 2.0 FR) is a data source rather than a publication.

---

## Appendix A. Proposed word-level annotation protocol for Romanized Kannada–English

This protocol is proposed; it has not been carried out.

1. **Sample.** Draw 1,000 Romanized or mixed-script comments, stratified by sentiment label, from the DravidianCodeMix Kannada *test* split, so that annotated data never overlaps training data.
2. **Tag set.**
   - `en`: English word, including English words in Indian spelling.
   - `kn`: Kannada word in any script, including Kannada inflection attached to an English stem (e.g. *office-ge* → `mixed`).
   - `ta`, `ml`, `hi`: other languages.
   - `ne`: named entity (people, films, places).
   - `univ`: numbers, emoji, punctuation, laughter (*hahaha*).
   - `mixed`: an intra-word switch.
   - `other`: any other language, or unintelligible.
3. **Guidelines.**
   - Tag by language of origin as used in the sentence.
   - Established English loanwords in Kannada morphology (*college-ge*) are `mixed`; bare loanwords (*college*) are `en`.
   - Each token is tagged independently of the sentence's majority language.
4. **Annotators.** Two native Kannada speakers fluent in English annotate independently.
   - Report Cohen's κ per tag.
   - A third annotator adjudicates disagreements.
5. **Use.** Evaluate in-context word-level LID and the code-mixing indices against the gold tags, and report CMI estimation error.

## Appendix B. Reproducibility

```
.venv\Scripts\python experiments\train_lid.py        # LID models + lid_results.json
.venv\Scripts\python experiments\short_text_lid.py   # follow-up fusion experiment (Section 13.4)
.venv\Scripts\python experiments\run_sentiment.py    # sentiment grid, bootstrap, routing
.venv\Scripts\python experiments\compare_runs.py     # diff against results/first_run/
```

- **Seed:** 42.
- **Selection:** hyper-parameter grids are as in Section 12; dev-set scores for every C value are stored in `sentiment_results.json` (`c_search`).

**Re-run check.** The full pipeline was run twice with scikit-learn 1.9.1.

The comparison (`experiments/results/reproducibility.json`) exposed a source of non-determinism in the first run:
- The word-LID training words were collected in Python sets, whose iteration order changes between processes.
- Training-row order slightly changes the fitted logistic regression.
- This was fixed by sorting.

After the fix, two consecutive `train_lid.py` runs produced identical results in every field except run time.

Against the first run:
- **Unchanged:** every result that does not depend on the word-LID model. This covers the dataset cleaning, the sentence-LID model and its langid.py comparison, the test metrics of all `word`, `char` and `word+char` runs, and the routing experiment.
- **Changed:** only results that use word-level tags.
  - The `+norm` and `+lang` runs moved by at most 0.0099 macro-F1 (largest: Kannada `word+char+norm+lang` LR, 0.698 → 0.708).
  - The code-mixed/monolingual breakdowns, whose subsets are defined by word tags, also changed.
  - Word-LID test accuracy moved from 72.30% to 72.33%.
  - The estimated code-mixed share moved by at most 0.05 percentage points, and mean CMI by at most 0.11 points.

This paper reports the final, deterministic run. Both runs are stored in `experiments/results/first_run/` and `experiments/results/final_run/`.
