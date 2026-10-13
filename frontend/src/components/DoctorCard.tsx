import { CalendarDays, Clock3, UserRound } from 'lucide-react'
import { Doctor } from '../types'

type DoctorCardProps = {
  doctor: Doctor
  onSelect: (name: string) => void
}

export default function DoctorCard({ doctor, onSelect }: DoctorCardProps) {
  return (
    <article className="doctor-card">
      <div className="doctor-avatar"><UserRound size={21} /></div>
      <div className="doctor-info">
        <h3>{doctor.name}</h3>
        <span className="specialty">{doctor.specialization}</span>
        <div className="doctor-meta">
          <span><Clock3 size={14} />{doctor.experience_years} years experience</span>
          <span><CalendarDays size={14} />{doctor.working_days.join(' · ')}</span>
        </div>
      </div>
      <button
        className="select-button"
        onClick={() => onSelect(doctor.name)}
        type="button"
      >
        Choose doctor
      </button>
    </article>
  )
}
