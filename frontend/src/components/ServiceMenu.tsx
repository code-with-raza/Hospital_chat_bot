import {
  CalendarDays,
  CalendarPlus,
  MessageCircle,
  Stethoscope,
  UserRound,
} from 'lucide-react'

type ServiceMenuProps = {
  onSelect: (message: string | null) => void
}

const services = [
  {
    label: 'Chat with us',
    detail: 'Start a fresh conversation',
    icon: MessageCircle,
    message: null,
  },
  {
    label: 'Find a doctor',
    detail: 'Browse hospital specialists',
    icon: Stethoscope,
    message: 'Show me all doctors',
  },
  {
    label: 'Book an appointment',
    detail: 'Choose a doctor and time',
    icon: CalendarPlus,
    message: 'I want to book an appointment',
  },
  {
    label: 'Check an appointment',
    detail: 'Verify with your patient ID and registered phone',
    icon: CalendarDays,
    message: 'What is my appointment?',
  },
  {
    label: 'Patient records',
    detail: 'Verify with your patient ID and registered phone',
    icon: UserRound,
    message: 'I want to check my patient details',
  },
]

export default function ServiceMenu({ onSelect }: ServiceMenuProps) {
  return (
    <aside className="service-menu" aria-label="Patient services">
      <p className="menu-eyebrow">Your care</p>
      <nav aria-label="Hospital services">
        {services.map(({ label, detail, icon: Icon, message }, index) => (
          <button
            className={`service-link ${index === 0 ? 'active' : ''}`}
            key={label}
            type="button"
            onClick={() => onSelect(message)}
          >
            <Icon size={18} aria-hidden="true" />
            <span>
              <strong>{label}</strong>
              <small>{detail}</small>
            </span>
          </button>
        ))}
      </nav>

      <div className="menu-help">
        <strong>Here for your visit</strong>
        <span>Choose a service to get started. Your conversation stays private.</span>
      </div>
    </aside>
  )
}
