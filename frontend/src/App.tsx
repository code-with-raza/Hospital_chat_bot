
import { FormEvent, useRef, useState } from 'react'
import ChatWindow from './components/ChatWindow'
import Header from './components/Header'
import ServiceMenu from './components/ServiceMenu'
import { ChatResponse, Message } from './types'
import './index.css'

const welcomeMessage: Message = {
  role: 'assistant',
  content:
    'Welcome to Northstar Health. I can help you find a doctor, check appointment availability, or look up your patient information.',
}

function createSessionId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return crypto.randomUUID()
  }

  return `${Date.now()}-${Math.random().toString(16).slice(2)}`
}

function getSessionId(): string {
  const savedSessionId = sessionStorage.getItem('hospital-chat-session')
  if (savedSessionId) return savedSessionId

  const newSessionId = createSessionId()
  sessionStorage.setItem('hospital-chat-session', newSessionId)
  return newSessionId
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([welcomeMessage])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [sessionId, setSessionId] = useState(getSessionId)
  const [menuOpen, setMenuOpen] = useState(false)
  const activeSessionId = useRef(sessionId)

  async function submitMessage(message: string) {
    if (!message.trim() || loading) return

    const requestSessionId = activeSessionId.current
    setMessages(current => [...current, { role: 'user', content: message }])
    setInput('')
    setLoading(true)

    try {
      const response = await fetch('http://localhost:8000/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: requestSessionId, message }),
      })

      if (!response.ok) {
        throw new Error('The hospital assistant is temporarily unavailable.')
      }

      const result = (await response.json()) as ChatResponse
      if (requestSessionId !== activeSessionId.current) return

      setMessages(current => [
        ...current,
        {
          role: 'assistant',
          content: result.response,
          type: result.type,
          data: result.data,
        },
      ])
    } catch (error) {
      if (requestSessionId !== activeSessionId.current) return

      const message = error instanceof Error
        ? error.message
        : 'Something went wrong. Please try again.'
      setMessages(current => [
        ...current,
        { role: 'assistant', content: message, type: 'error' },
      ])
    } finally {
      if (requestSessionId === activeSessionId.current) setLoading(false)
    }
  }

  function send(event: FormEvent) {
    event.preventDefault()
    void submitMessage(input)
  }

  function startNewChat() {
    const nextSessionId = createSessionId()
    activeSessionId.current = nextSessionId
    sessionStorage.setItem('hospital-chat-session', nextSessionId)
    setSessionId(nextSessionId)
    setMessages([welcomeMessage])
    setInput('')
    setLoading(false)
    setMenuOpen(false)
  }

  function chooseMenuAction(message: string | null) {
    setMenuOpen(false)
    if (message === null) {
      startNewChat()
      return
    }
    void submitMessage(message)
  }

  return (
    <div className="app">
      <Header menuOpen={menuOpen} onToggleMenu={() => setMenuOpen(open => !open)} />
      <div className={`chat-layout ${menuOpen ? 'menu-open' : ''}`}>
        {menuOpen && <ServiceMenu onSelect={chooseMenuAction} />}
        <ChatWindow
          messages={messages}
          input={input}
          setInput={setInput}
          loading={loading}
          send={send}
          selectDoctor={name => void submitMessage(name)}
          selectSuggestion={message => void submitMessage(message)}
        />
      </div>
    </div>
  )
}
