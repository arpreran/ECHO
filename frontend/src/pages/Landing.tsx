import { motion } from 'framer-motion'
import { ArrowRight, AudioLines, FlaskConical } from 'lucide-react'
import { useRef } from 'react'
import { Link } from 'react-router-dom'
import { Logo, StatusDot, useHealth } from '../components/brand'
import { Orb } from '../components/Orb'

export function Landing() {
  const { health, online } = useHealth()
  const level = useRef(0)
  return (
    <div className="mx-auto flex min-h-screen max-w-6xl flex-col px-4 sm:px-8">
      <header className="flex items-center justify-between py-6">
        <Logo />
        <StatusDot ok={online} label={online === null ? 'Connecting…' : online ? 'Backend online' : 'Backend offline'} />
      </header>

      <main className="flex flex-1 flex-col items-center justify-center py-10">
        <motion.div initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ duration: 0.8 }}>
          <Orb state="idle" levelRef={level} size={200} />
        </motion.div>
        <motion.h1 initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.15 }}
          className="h-display mt-2 text-center text-5xl font-semibold tracking-[0.3em] text-white sm:text-6xl">ECHO</motion.h1>
        <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}
          className="mt-3 text-center text-lg text-mist">Intelligence, in Two Modes.</motion.p>

        <div className="mt-12 grid w-full max-w-4xl gap-5 md:grid-cols-2">
          <ModeCard to="/companion" delay={0.4} icon={<AudioLines className="size-6" />} accent="cyan"
            title="ECHO Companion" tagline="Talk to your AI friend."
            body="A voice-first multilingual companion for Kannada, English, Hindi, Tamil and Malayalam, including code-mixed and Romanized speech, with memory that you control."
            status={health ? (health.companion.chat.configured ? `${health.companion.chat.provider} · ${health.companion.chat.model}` : 'API key not configured') : undefined}
            statusOk={health?.companion.chat.configured} />
          <ModeCard to="/lab" delay={0.5} icon={<FlaskConical className="size-6" />} accent="violet"
            title="ECHO NLP Lab" tagline="Explore multilingual language intelligence."
            body="Romanized Dravidian language identification, code-mixing analysis and explainable sentiment classification, built with classical machine learning only."
            status={health ? (health.nlp_lab.lid_ready ? `${health.nlp_lab.sentiment_models.length} trained models loaded` : 'Models not trained') : undefined}
            statusOk={health?.nlp_lab.lid_ready} />
        </div>
      </main>
      <footer className="py-6 text-center text-xs text-mist">The Companion uses an external conversational AI API. The NLP Lab runs entirely on local classical models.</footer>
    </div>
  )
}

function ModeCard({ to, title, tagline, body, icon, accent, delay, status, statusOk }: {
  to: string; title: string; tagline: string; body: string; icon: React.ReactNode; accent: 'cyan' | 'violet'; delay: number; status?: string; statusOk?: boolean
}) {
  const ring = accent === 'cyan' ? 'hover:border-cyan-glow/50 hover:shadow-[0_0_60px_-20px_rgba(34,211,238,0.6)]' : 'hover:border-violet-glow/50 hover:shadow-[0_0_60px_-20px_rgba(139,92,246,0.6)]'
  const iconBg = accent === 'cyan' ? 'bg-cyan-glow/10 text-cyan-soft' : 'bg-violet-glow/10 text-violet-300'
  return (
    <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay }}>
      <Link to={to} className={`focus-ring glass group block h-full rounded-3xl p-7 transition-all duration-300 ${ring}`}>
        <div className={`mb-6 grid size-12 place-items-center rounded-2xl ${iconBg}`}>{icon}</div>
        <div className="eyebrow">{tagline}</div>
        <h2 className="h-display mt-1 text-2xl font-semibold text-white">{title}</h2>
        <p className="mt-3 text-sm leading-relaxed text-mist">{body}</p>
        <div className="mt-6 flex items-center justify-between">
          {status ? <StatusDot ok={statusOk ?? null} label={status} /> : <span />}
          <span className="inline-flex items-center gap-1 text-sm font-medium text-white">Enter <ArrowRight className="size-4 transition group-hover:translate-x-1" /></span>
        </div>
      </Link>
    </motion.div>
  )
}
