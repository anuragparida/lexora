import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import './index.css'
import App from './App.tsx'
import { Home } from './pages/Home'
import { WeaknessProfilePage } from './pages/WeaknessProfilePage'
import { StudyStub } from './pages/StudyStub'
import { AuthForm } from './components/AuthForm'
import { ProtectedRoute } from './components/ProtectedRoute'

// Phase 2.3 (card t_ffe6d6af): top-level router.
//
// Route map:
//   /                 public   Phase 1 search/filter UI
//   /login            public   auth form (login)
//   /signup           public   auth form (signup)
//   /weakness-profile protected placeholder; Phase 2.4 fills it in
//   /study            protected placeholder; Phase 5+ fills it in
//
// Only /weakness-profile and /study are gated. The existing Anki-deck
// flow on / stays open (per the card's hard rule).

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<App />}>
          <Route path="/" element={<Home />} />
          <Route path="/login" element={<AuthForm mode="login" />} />
          <Route path="/signup" element={<AuthForm mode="signup" />} />
          <Route
            path="/weakness-profile"
            element={
              <ProtectedRoute>
                {(user) => <WeaknessProfilePage user={user} />}
              </ProtectedRoute>
            }
          />
          <Route
            path="/study"
            element={
              <ProtectedRoute>
                {() => <StudyStub />}
              </ProtectedRoute>
            }
          />
        </Route>
      </Routes>
    </BrowserRouter>
  </StrictMode>,
)