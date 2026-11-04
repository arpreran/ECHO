import { Navigate, Route, Routes } from 'react-router-dom'
import { Landing } from './pages/Landing'
import { CompanionLayout } from './companion/CompanionLayout'
import { TalkPage } from './companion/TalkPage'
import { HistoryPage } from './companion/HistoryPage'
import { MemoryPage } from './companion/MemoryPage'
import { SettingsPage } from './companion/SettingsPage'
import { LabLayout } from './lab/LabLayout'
import { Overview } from './lab/Overview'
import { Workbench } from './lab/Workbench'
import { LanguageDetector } from './lab/LanguageDetector'
import { CodeMixing } from './lab/CodeMixing'
import { Sentiment } from './lab/Sentiment'
import { Explainability } from './lab/Explainability'
import { ModelComparison } from './lab/ModelComparison'
import { DatasetExplorer } from './lab/DatasetExplorer'
import { Analytics } from './lab/Analytics'
import { Methodology } from './lab/Methodology'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Landing />} />
      <Route path="/companion" element={<CompanionLayout />}>
        <Route index element={<TalkPage />} />
        <Route path="history" element={<HistoryPage />} />
        <Route path="memory" element={<MemoryPage />} />
        <Route path="settings" element={<SettingsPage />} />
      </Route>
      <Route path="/lab" element={<LabLayout />}>
        <Route index element={<Overview />} />
        <Route path="workbench" element={<Workbench />} />
        <Route path="language" element={<LanguageDetector />} />
        <Route path="code-mixing" element={<CodeMixing />} />
        <Route path="sentiment" element={<Sentiment />} />
        <Route path="explainability" element={<Explainability />} />
        <Route path="models" element={<ModelComparison />} />
        <Route path="datasets" element={<DatasetExplorer />} />
        <Route path="analytics" element={<Analytics />} />
        <Route path="methodology" element={<Methodology />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
