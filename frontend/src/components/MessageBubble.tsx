import { Bot, UserRound } from 'lucide-react'
import { Doctor, Message } from '../types'
import AppointmentCard from './AppointmentCard'
import DoctorCard from './DoctorCard'
import MarkdownContent from './MarkdownContent'

type MessageBubbleProps = {
  message: Message
  onSelectDoctor: (name: string) => void
}

const cardTypes = new Set([
  'appointment_slots',
  'appointment_confirmation',
  'appointment_details',
  'patient_details',
])

export default function MessageBubble({ message, onSelectDoctor }: MessageBubbleProps) {
  const isUser = message.role === 'user'
  const doctors = message.type === 'doctor_list' && Array.isArray(message.data)
    ? message.data as Doctor[]
    : []

  return (
    <div className={`message-row ${isUser ? 'user-row' : ''}`}>
      <div className={`message-avatar ${isUser ? 'user-avatar' : ''}`}>
        {isUser ? <UserRound size={17} /> : <Bot size={18} />}
      </div>

      <div className="message-content">
        <span className="sender-label">{isUser ? 'You' : 'Northstar assistant'}</span>
        <div className={`bubble ${isUser ? 'user-bubble' : ''}`}>
          {isUser ? message.content : <MarkdownContent content={message.content} />}
        </div>

        {doctors.length > 0 && (
          <div className="doctor-list">
            {doctors.map(doctor => (
              <DoctorCard
                key={doctor.id}
                doctor={doctor}
                onSelect={onSelectDoctor}
              />
            ))}
          </div>
        )}

        {message.type && cardTypes.has(message.type) && (
          <AppointmentCard data={message.data} type={message.type} />
        )}
      </div>
    </div>
  )
}
