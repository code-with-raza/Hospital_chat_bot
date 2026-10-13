import { FormEvent, useEffect, useRef } from 'react'
import { ArrowUp, Paperclip } from 'lucide-react'

type ChatInputProps = {
  value: string
  onChange: (value: string) => void
  onSubmit: (event: FormEvent) => void
  disabled: boolean
}

export default function ChatInput({
  value,
  onChange,
  onSubmit,
  disabled,
}: ChatInputProps) {
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (!disabled) inputRef.current?.focus()
  }, [disabled])

  return (
    <form className="composer" onSubmit={onSubmit}>
      <button
        type="button"
        className="attach-button"
        aria-label="Attachments unavailable"
      >
        <Paperclip size={19} />
      </button>
      <input
        ref={inputRef}
        value={value}
        onChange={event => onChange(event.target.value)}
        placeholder="Ask about doctors or appointments..."
        disabled={disabled}
        aria-label="Message"
      />
      <button
        className="send-button"
        type="submit"
        disabled={disabled || !value.trim()}
        aria-label="Send message"
      >
        <ArrowUp size={19} />
      </button>
    </form>
  )
}
