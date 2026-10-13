import { CalendarCheck, Clock3, Stethoscope } from 'lucide-react'
import { Appointment, PatientRecord } from '../types'

type SlotData = {
  slots?: string[]
  date?: string
  working_days?: string[]
  doctor?: { name: string }
}

type ConfirmationData = {
  appointment?: Appointment
  patient?: PatientRecord
}

type AppointmentCardProps = {
  data: unknown
  type: string
}

function formatTime(value: string): string {
  const match = value.match(/^(\d{1,2}):(\d{2})$/)
  if (!match) return value

  const hour = Number(match[1])
  const minute = match[2]
  const period = hour < 12 ? 'AM' : 'PM'
  const displayHour = hour % 12 || 12
  return `${displayHour}:${minute} ${period}`
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function AppointmentDetails({
  appointment,
  patientId,
}: {
  appointment: Appointment
  patientId?: string
}) {
  return (
    <div className="result-card appointment-result">
      <div className="result-title">
        <CalendarCheck size={17} /> Appointment details
      </div>
      <strong>{appointment.doctor_name}</strong>
      <span>{appointment.specialization}</span>
      <div className="appointment-meta">
        <span><CalendarCheck size={15} />{appointment.date}</span>
        <span><Clock3 size={15} />{formatTime(appointment.time)}</span>
        <span><Stethoscope size={15} />{appointment.status}</span>
      </div>
      <small>Appointment ID: {appointment.appointment_id}</small>
      {appointment.doctor_id && <small>Doctor ID: {appointment.doctor_id}</small>}
      {(patientId ?? appointment.patient_id) && (
        <small>Patient ID: {patientId ?? appointment.patient_id}</small>
      )}
    </div>
  )
}

function AppointmentSlots({ data }: { data: SlotData }) {
  const slots = data.slots ?? []
  const title = data.doctor?.name
    ? `${data.doctor.name} · ${data.date ? `Times for ${data.date}` : 'Available times'}`
    : data.date
      ? `Times for ${data.date}`
      : 'Available appointment times'

  return (
    <div className="result-card">
      <div className="result-title"><Clock3 size={17} />{title}</div>
      {data.working_days && <p>Working days: {data.working_days.join(', ')}</p>}
      <div className="slot-list">
        {slots.map(slot => <span key={slot}>{formatTime(slot)}</span>)}
      </div>
      <p className="card-hint">Type your preferred date or time to continue.</p>
    </div>
  )
}

function PatientDetails({ patient }: { patient: PatientRecord }) {
  return (
    <div className="result-card">
      <div className="result-title"><Stethoscope size={17} />Patient record</div>
      <strong>{patient.name}</strong>
      <span>{patient.age} years · {patient.weight} kg · {patient.phone}</span>
      <small>Patient ID: {patient.patient_id}</small>
    </div>
  )
}

export default function AppointmentCard({ data, type }: AppointmentCardProps) {
  if (!isRecord(data)) return null

  if (type === 'appointment_slots') {
    return <AppointmentSlots data={data as SlotData} />
  }

  if (type === 'appointment_details' && Array.isArray(data)) {
    return (
      <div className="appointment-list">
        {data.map((item, index) => (
          <AppointmentDetails key={index} appointment={item as Appointment} />
        ))}
      </div>
    )
  }

  if (type === 'appointment_confirmation' && isRecord(data.appointment)) {
    const confirmation = data as ConfirmationData
    return (
      <AppointmentDetails
        appointment={confirmation.appointment!}
        patientId={confirmation.patient?.patient_id}
      />
    )
  }

  if (type === 'patient_details' && 'patient_id' in data) {
    return <PatientDetails patient={data as unknown as PatientRecord} />
  }

  return null
}
