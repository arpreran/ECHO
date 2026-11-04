import { createContext, useCallback, useContext, useEffect, useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import { Brain, History, MessageCircle, Settings2 } from 'lucide-react'
import { Logo, StatusDot, useHealth } from '../components/brand'
import { cx } from '../components/ui'
import { api, type Preferences, type ProviderStatus } from '../lib/api'

interface Ctx {
  prefs: Preferences | null
  savePrefs: (p: Partial<Preferences>) => Promise<void>
  status: ProviderStatus | null
  online: boolean | null
  conversationId: string | null
  setConversationId: (id: string | null) => void
}
const CompanionCtx = createContext<Ctx | null>(null)
export const useCompanion = () => useContext(CompanionCtx)!

const CONV_KEY = 'echo.conversationId'

export function CompanionLayout() {
  const { online } = useHealth()
  const [prefs, setPrefs] = useState<Preferences | null>(null)
  const [status, setStatus] = useState<ProviderStatus | null>(null)
  const [conversationId, setConv] = useState<string | null>(() => { try { return localStorage.getItem(CONV_KEY) } catch { return null } })

  useEffect(() => {
    if (!online) return
    api.companion.preferences().then(setPrefs).catch(() => {})
    api.companion.status().then(setStatus).catch(() => {})
  }, [online])

  const savePrefs = useCallback(async (p: Partial<Preferences>) => { setPrefs(await api.companion.savePreferences(p)) }, [])
  const setConversationId = useCallback((id: string | null) => {
    setConv(id)
    try { if (id) localStorage.setItem(CONV_KEY, id); else localStorage.removeItem(CONV_KEY) } catch { /* storage unavailable */ }
  }, [])

  const nav = [
    { to: '/companion', end: true, label: 'Talk', icon: MessageCircle },
    { to: '/companion/history', label: 'History', icon: History },
    { to: '/companion/memory', label: 'Memory', icon: Brain },
    { to: '/companion/settings', label: 'Settings', icon: Settings2 },
  ]
  const local = status?.chat.provider === 'ollama'
  const localIssue = !local ? null : !status?.chat.reachable ? 'Ollama not running' : !status?.chat.model_available ? `${status?.chat.model} not installed` : null
  const apiLabel = online === false ? 'Backend offline' : status
    ? (localIssue ?? (status.chat.configured ? (local ? `Ollama · ${status.chat.model}` : `${status.chat.label ?? status.chat.provider} · ${status.chat.model}`) : 'Chat API not configured'))
    : 'Connecting…'
  const apiOk = online === false ? false : status ? status.chat.configured && !localIssue : null

  return (
    <CompanionCtx.Provider value={{ prefs, savePrefs, status, online, conversationId, setConversationId }}>
      <div className="flex min-h-screen flex-col">
        <header className="sticky top-0 z-20 border-b border-line bg-ink-950/80 backdrop-blur-xl">
          <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-4 py-3 sm:px-6">
            <Logo sub="Companion" />
            <nav className="order-3 flex w-full gap-1 overflow-x-auto sm:order-none sm:w-auto">
              {nav.map(({ to, end, label, icon: Icon }) => (
                <NavLink key={to} to={to} end={end}
                  className={({ isActive }) => cx('focus-ring flex items-center gap-2 rounded-xl px-3 py-2 text-sm transition-colors', isActive ? 'bg-cyan-glow/10 text-cyan-soft' : 'text-mist hover:text-white')}>
                  <Icon className="size-4" />{label}
                </NavLink>
              ))}
            </nav>
            <StatusDot ok={apiOk} label={apiLabel} />
          </div>
        </header>
        <main className="mx-auto w-full max-w-6xl flex-1 px-4 py-6 sm:px-6"><Outlet /></main>
      </div>
    </CompanionCtx.Provider>
  )
}
