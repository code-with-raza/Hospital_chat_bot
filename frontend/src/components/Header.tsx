import { Activity, Menu, ShieldCheck, X } from 'lucide-react'

type HeaderProps = {
  menuOpen: boolean
  onToggleMenu: () => void
}

export default function Header({ menuOpen, onToggleMenu }: HeaderProps) {
  return (
    <header className="topbar">
      <div className="brand">
        <button
          className="menu-toggle"
          type="button"
          aria-label={menuOpen ? 'Close patient services menu' : 'Open patient services menu'}
          aria-expanded={menuOpen}
          onClick={onToggleMenu}
        >
          {menuOpen ? <X size={19} /> : <Menu size={19} />}
        </button>
        <div className="brand-icon"><Activity size={23} /></div>
        <div>
          <strong>Northstar Health</strong>
          <span>Patient support</span>
        </div>
      </div>

      <div className="online">
        <span className="online-dot" />Online
        <span className="secure"><ShieldCheck size={15} />Private &amp; secure</span>
      </div>
    </header>
  )
}
