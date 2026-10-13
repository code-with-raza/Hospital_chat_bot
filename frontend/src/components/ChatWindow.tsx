import { FormEvent, useEffect, useRef } from 'react'
import { HeartPulse, LockKeyhole, Sparkles } from 'lucide-react'
import { Message } from '../types'
import ChatInput from './ChatInput'
import MessageBubble from './MessageBubble'

type ChatWindowProps = {
  messages: Message[]
  input: string
  setInput: (value: string) => void
  loading: boolean
  send: (event: FormEvent) => void
  selectDoctor: (name: string) => void
  selectSuggestion: (message: string) => void
}

function WelcomePanel({
  disabled,
  onSelect,
}: {
  disabled: boolean
  onSelect: (message: string) => void
}) {
  const suggestions = [
    { label: 'Find a cardiologist', message: 'I need a cardiologist' },
    { label: 'Book an appointment', message: 'I want to book an appointment' },
    { label: 'Check my appointment', message: 'What is my appointment?' },
  ]

  return (
    <div className="welcome">
      <div className="welcome-mark"><Sparkles size={22} /></div>
      <span className="eyebrow">YOUR HOSPITAL, MADE EASIER</span>
      <h1>How can we help<br />you today?</h1>
      <p>
        Find the right specialist, explore appointment times, or get help with
        your hospital visit.
      </p>
      <div className="suggestions">
        {suggestions.map(suggestion => (
          <button
            key={suggestion.label}
            type="button"
            disabled={disabled}
            onClick={() => onSelect(suggestion.message)}
          >
            {suggestion.label}
          </button>
        ))}
      </div>
    </div>
  )
}

function TypingIndicator() {
  return (
    <div className="typing" aria-label="Assistant is responding">
      <span /><span /><span />
      <small>Finding the right information…</small>
    </div>
  )
}

export default function ChatWindow({
  messages,
  input,
  setInput,
  loading,
  send,
  selectDoctor,
  selectSuggestion,
}: ChatWindowProps) {
  const endOfConversation = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endOfConversation.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  return (
    <main className="chat-shell">
      <div className="chat-toolbar">
        <div className="assistant-presence">
          <div className="assistant-mark"><HeartPulse size={19} /></div>
          <div>
            <strong>Care assistant</strong>
            <span><i /> Here to help, any time</span>
          </div>
        </div>
        <div className="toolbar-note">
          <LockKeyhole size={15} /> Your conversation is private
        </div>
      </div>

      <section className="conversation" aria-live="polite">
        {messages.length === 1 && (
          <WelcomePanel disabled={loading} onSelect={selectSuggestion} />
        )}
        {messages.map((message, index) => (
          <MessageBubble
            key={index}
            message={message}
            onSelectDoctor={selectDoctor}
          />
        ))}
        {loading && <TypingIndicator />}
        <div ref={endOfConversation} />
      </section>

      <div className="composer-wrap">
        <ChatInput
          value={input}
          onChange={setInput}
          onSubmit={send}
          disabled={loading}
        />
        <p>
          For medical advice, please contact a healthcare professional. This
          assistant handles hospital services only.
        </p>
      </div>
    </main>
  )
}
