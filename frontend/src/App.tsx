import { Outlet } from 'react-router-dom'
import { Header } from './components/Header'

// Phase 2.3 (card t_ffe6d6af): the app's outermost layout. Renders the
// shared Header (login/signup links or email + logout) above a router
// outlet. Per the card's hard rule, this stays minimal — no deep refactor
// of the Phase 1 search/filter UI. The Home page mounts at `/` and brings
// its own internal layout.

export default function App() {
  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100">
      <Header />
      <Outlet />
    </div>
  )
}