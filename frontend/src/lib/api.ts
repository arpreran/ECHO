// Typed client for the ECHO backend. All requests go through the Vite proxy (/api).

export class ApiError extends Error {
  status: number
  retryable: boolean
  constructor(message: string, status: number, retryable = true) {
    super(message)
    this.status = status
    this.retryable = retryable
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(path, {
      ...init,
      headers: init?.body && !(init.body instanceof FormData) ? { 'Content-Type': 'application/json', ...init?.headers } : init?.headers,
    })
  } catch {
    throw new ApiError('Cannot reach the ECHO backend. Is it running on port 8000?', 0)
  }
  if (res.status === 204) return undefined as T
  const ct = res.headers.get('content-type') || ''
  const body = ct.includes('application/json') ? await res.json() : await res.text()
  if (!res.ok) {
    const d = (body as { detail?: unknown })?.detail
    let msg = typeof body === 'string' ? body : `Request failed (${res.status})`
    let retryable = res.status >= 500 || res.status === 429
    if (typeof d === 'string') msg = d
    else if (d && typeof d === 'object' && 'message' in d) {
      msg = String((d as { message: string }).message)
      retryable = Boolean((d as { retryable?: boolean }).retryable)
    } else if (Array.isArray(d)) msg = d.map((x: { msg?: string }) => x.msg).join('; ')
    throw new ApiError(msg, res.status, retryable)
  }
  return body as T
}

const post = <T>(path: string, data: unknown) => request<T>(path, { method: 'POST', body: JSON.stringify(data) })

// ------------------------------------------------------------------ types
export type Emotion = 'neutral' | 'happy' | 'supportive' | 'curious'

export interface DetectedLanguage {
  primary: string
  primary_name: string
  confidence: number | null
  uncertain: boolean
  languages: string[]
  code_mixed: boolean
  text_type: string
  summary: string | null
  model: string
}

export interface Message {
  id: string
  conversation_id: string
  sender: 'user' | 'assistant'
  content: string
  created_at: string
  input_mode: 'text' | 'voice'
  language: DetectedLanguage | null
  emotion: Emotion | null
  metadata: Record<string, unknown>
}

export interface Conversation {
  id: string
  title: string
  created_at: string
  updated_at: string
  message_count: number
  last_message?: string | null
}

export interface ChatResponse {
  conversation_id: string | null
  conversation_title: string | null
  persisted: boolean
  user_message: Message | null
  assistant_message: Message | null
  reply: string
  emotion: Emotion
  detected_language: DetectedLanguage | null
  memory_suggestions: string[]
  context: { recent_turns: number; memories: number; past_snippets: { conversation_id: string; title: string; created_at: string }[] }
  meta: { provider: string; model: string; user_mood?: string | null; reply_language?: string | null }
}

export interface Preferences {
  display_name: string
  reply_language: 'auto' | 'en' | 'kn' | 'hi' | 'ta' | 'ml'
  reply_script: 'match' | 'romanized' | 'native'
  style: 'friendly' | 'concise' | 'detailed' | 'playful'
  voice_input_lang: string
  voice_name: string
  speech_rate: number
  auto_speak: boolean
  save_history: boolean
  use_memories: boolean
  use_past_conversations: boolean
  suggest_memories: boolean
}

export interface Memory {
  id: string
  content: string
  category: string
  source_message_id: string | null
  created_at: string
  updated_at: string
}

export interface ProviderStatus {
  chat: { provider: string; model: string; configured: boolean; reachable?: boolean; model_available?: boolean; label?: string }
  stt: { provider: string; configured: boolean }
  tts: { provider: string; configured: boolean; model: string | null }
  lid_available?: boolean
  data_sent_to_provider?: string[]
}

export interface Health {
  status: string
  database_schema_version: number
  nlp_lab: { lid_ready: boolean; sentiment_models: string[]; errors: string[] }
  companion: ProviderStatus
}

// NLP Lab
export interface TokenTag {
  token: string
  script: string
  lang: string
  confidence: number | null
  uncertain: boolean
  probabilities: Record<string, number> | null
}

export interface LIDResult {
  script_profile: { distribution: Record<string, number>; dominant_script: string | null; text_type: string; letter_count: number; emoji_count: number }
  primary_language: string
  primary_language_name: string
  confidence: number | null
  decision_source: string
  sentence_probabilities: Record<string, number> | null
  uncertain: boolean
  unsupported: boolean
  tokens: TokenTag[]
  word_language_counts: Record<string, number>
}

export interface CodeMixResult {
  tokens: TokenTag[]
  stats: {
    tokens: number; language_tokens: number; language_independent_tokens: number
    language_distribution: Record<string, number>; languages: string[]; is_code_mixed: boolean
    cmi: number; m_index: number; i_index: number; switch_points: number; spans: { lang: string; length: number }[]
  }
  script_mixing: Record<string, number>
  normalized: { basic: string; language_aware: string }
  normalization_rules: string[]
  uncertain_tokens: number
  language_names: Record<string, string>
}

export interface Explanation {
  target_class: string
  score: number
  bias: number
  non_token_contribution: number
  tokens: { token: string; normalized: string; lang: string; contribution: number }[]
  top_features: { feature: string; block: string; contribution: number }[]
  method: string
}

export interface SentimentPrediction {
  supported: boolean
  message?: string
  routing: { language: string | null; source: string; lid: { primary_language: string; primary_language_name: string; confidence: number | null; uncertain: boolean } | null }
  label?: 'positive' | 'negative' | 'neutral'
  model?: string
  model_kind?: string
  feature_config?: string
  feature_config_description?: string
  trained_on?: string
  language_name?: string
  decision_scores?: Record<string, number>
  probabilities?: Record<string, number> | null
  confidence?: number | null
  confidence_type?: string
  margin?: number
  test_macro_f1?: number
  test_accuracy?: number
  explanation?: Explanation
}

export interface Metrics {
  n: number; accuracy: number; macro_precision: number; macro_recall: number; macro_f1: number; weighted_f1?: number
  per_class: Record<string, { precision: number; recall: number; f1: number; support: number }>
  confusion_matrix: number[][]
}

export interface ModelInfo {
  id: string; language: string; language_name: string; model: string; model_kind: string
  feature_config: string; feature_config_description: string; C: number; train_size: number
  dev_macro_f1: number; test_metrics: Metrics; n_features: number
}

export type SentimentLang = 'auto' | 'kn' | 'ta' | 'ml'
export type ModelKind = 'logreg' | 'linear_svm'

// ------------------------------------------------------------- endpoints
export const api = {
  health: () => request<Health>('/api/health'),
  companion: {
    status: () => request<ProviderStatus>('/api/companion/status'),
    chat: (body: { message: string; conversation_id?: string | null; input_mode?: 'text' | 'voice'; incognito?: boolean; history?: { role: string; content: string }[] }) =>
      post<ChatResponse>('/api/companion/chat', body),
    transcribe: (audio: Blob, languageHint?: string) => {
      const fd = new FormData()
      fd.append('audio', audio, 'speech.webm')
      if (languageHint) fd.append('language_hint', languageHint)
      return request<{ text: string; provider: string; model: string; detected_language: DetectedLanguage | null }>('/api/companion/voice/transcribe', { method: 'POST', body: fd })
    },
    speak: async (text: string, voice?: string): Promise<ArrayBuffer> => {
      const res = await fetch('/api/companion/voice/speak', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text, voice: voice || null }) })
      if (!res.ok) {
        const b = await res.json().catch(() => ({}))
        throw new ApiError(b?.detail?.message || b?.detail || 'Speech synthesis failed', res.status)
      }
      return res.arrayBuffer()
    },
    conversations: () => request<Conversation[]>('/api/companion/conversations'),
    conversation: (id: string) => request<Conversation & { messages: Message[] }>(`/api/companion/conversations/${id}`),
    createConversation: (title?: string) => post<Conversation>('/api/companion/conversations', { title: title || 'New conversation' }),
    rename: (id: string, title: string) => request<Conversation>(`/api/companion/conversations/${id}`, { method: 'PATCH', body: JSON.stringify({ title }) }),
    remove: (id: string) => request<void>(`/api/companion/conversations/${id}`, { method: 'DELETE' }),
    removeAll: () => request<{ deleted: number }>('/api/companion/conversations?confirm=true', { method: 'DELETE' }),
    search: (q: string) => request<{ id: string; conversation_id: string; sender: string; content: string; created_at: string; title: string }[]>(`/api/companion/conversations/search?q=${encodeURIComponent(q)}`),
    exportUrl: (id: string, format: 'json' | 'markdown') => `/api/companion/conversations/${id}/export?format=${format}`,
    preferences: () => request<Preferences>('/api/companion/preferences'),
    savePreferences: (p: Partial<Preferences>) => post<Preferences>('/api/companion/preferences', p),
    memories: () => request<Memory[]>('/api/companion/memories'),
    addMemory: (content: string, category = 'general', source_message_id?: string | null) => post<Memory>('/api/companion/memories', { content, category, source_message_id }),
    updateMemory: (id: string, content: string, category?: string) => request<Memory>(`/api/companion/memories/${id}`, { method: 'PATCH', body: JSON.stringify({ content, category: category || 'general' }) }),
    deleteMemory: (id: string) => request<void>(`/api/companion/memories/${id}`, { method: 'DELETE' }),
    deleteAllMemories: () => request<{ deleted: number }>('/api/companion/memories?confirm=true', { method: 'DELETE' }),
  },
  nlp: {
    status: () => request<{ lid_ready: boolean; sentiment_models: string[]; errors: string[] }>('/api/nlp/status'),
    lid: (text: string) => post<LIDResult>('/api/nlp/language-identification', { text }),
    codeMixing: (text: string) => post<CodeMixResult>('/api/nlp/code-mixing', { text }),
    sentiment: (text: string, model: ModelKind, language: SentimentLang) => post<SentimentPrediction>('/api/nlp/sentiment', { text, model, language }),
    compare: (text: string, language: SentimentLang) => post<{ supported: boolean; routing: SentimentPrediction['routing']; language_name?: string; results: SentimentPrediction[]; agree?: boolean }>('/api/nlp/compare', { text, language }),
    analyze: (text: string, model: ModelKind, language: SentimentLang) => post<{ language_identification: LIDResult; code_mixing: CodeMixResult; sentiment: SentimentPrediction }>('/api/nlp/analyze', { text, model, language }),
    models: () => request<ModelInfo[]>('/api/nlp/models'),
    features: (id: string, n = 20) => request<{ id: string; classes: string[]; features: Record<string, { positive: { feature: string; block: string; weight: number }[]; negative: { feature: string; block: string; weight: number }[] }> }>(`/api/nlp/models/${id}/features?n=${n}`),
    results: () => request<{ lid_results: LidResults | null; sentiment_results: SentimentResults | null; dataset_audit: DatasetAudit | null }>('/api/nlp/results'),
    datasets: () => request<{ sources: { name: string; use: string; licence: string; url: string; citation: string }[]; audit: DatasetAudit | null; lid_data: { word_data: unknown; sentence_data: Record<string, number> | null } }>('/api/nlp/datasets'),
    samples: (params: Record<string, string | number | undefined>) => {
      const q = new URLSearchParams()
      Object.entries(params).forEach(([k, v]) => { if (v !== undefined && v !== '') q.set(k, String(v)) })
      return request<{ total: number; offset: number; limit: number; rows: { text: string; raw_label: string; label: string; lang: string; split: string; text_type: string }[] }>(`/api/nlp/datasets/samples?${q}`)
    },
  },
}

// Shapes of the experiment result files (subset used by the UI).
export interface LidEval { n: number; accuracy: number; macro_f1: number; per_class: Record<string, { precision: number; recall: number; 'f1-score': number; support: number }>; confusion_matrix: { labels: string[]; matrix: number[][] } }
export interface LidResults {
  seed: number; classes: string[]
  word_data: { by_split_lang: Record<string, number>; ambiguous_train_strings: number }
  word_model: { selected_C: number; dev_macro_f1: number; test: LidEval }
  sentence_data: Record<string, number>
  sentence_model: { selected_C: number; validation_macro_f1: number; test_overall: LidEval; test_by_source: Record<string, LidEval>; accuracy_by_length: Record<string, number>; n_by_length: Record<string, number> }
  langid_baseline: { description: string; test_overall: LidEval; test_by_source: Record<string, LidEval> }
  qualitative_probes: { text: string; primary: string; confidence: number | null; tokens: [string, string, number | null][] }[]
}
export interface Run { config: string; model: string; selected_C: number; dev_macro_f1: number; c_search: { C: number; dev_macro_f1: number }[]; n_features: number; test: Metrics & { by_text_type: Record<string, Metrics>; by_code_mixing: Record<string, Metrics> }; fit_seconds: number }
export interface Boot { delta_macro_f1: number; ci95: [number, number]; p_le_0: number; resamples: number }
export interface SentimentResults {
  seed: number; c_grid: Record<string, number[]>; configs: Record<string, string>
  per_language: Record<string, { majority_baseline: Metrics; runs: Record<string, Run> }>
  significance: Record<string, Record<string, Record<string, Boot> | Boot>>
  routing: Record<string, { config: string; pooled: Metrics; oracle_language_routing: Metrics; lid_routing: Metrics; lid_routing_language_accuracy: number; lid_routing_fallback_to_pooled: number; 'lid_routing vs pooled': Boot; 'oracle vs lid_routing': Boot }>
  deployed: Record<string, { config: string; C: number; dev_macro_f1: number }>
}
export interface DatasetAudit {
  source: string; label_mapping: Record<string, string | null>; excluded: string[]
  raw_audit: Record<string, { rows: number; by_split: Record<string, number>; raw_labels: Record<string, number>; text_type: Record<string, number> }>
  cleaning: Record<string, Record<string, number | Record<string, number>>>
  clean_label_distribution: Record<string, Record<string, number>>
  clean_text_type_distribution: Record<string, Record<string, number>>
  clean_code_mixed_share: Record<string, number>
  code_mixing_mean_cmi: Record<string, number>
  note: string
}
