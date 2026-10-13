export type Doctor = {
  id: string
  name: string
  specialization: string
  experience_years: number
  working_days: string[]
  appointment_slots: string[]
}

export type Appointment = {
  appointment_id: string
  patient_id?: string
  doctor_id?: string
  doctor_name: string
  specialization: string
  date: string
  time: string
  status: string
}

export type PatientRecord = {
  patient_id: string
  name: string
  phone: string
  age: number
  weight: number
}

export type ChatResponse = {
  response: string
  type: string
  data?: unknown
}

export type Message = {
  role: 'assistant' | 'user'
  content: string
  type?: string
  data?: unknown
}
