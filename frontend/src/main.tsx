import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Toaster } from 'sonner'
import './index.css'
import App from './App.tsx'
import { Home } from './pages/Home'
import { WeaknessProfilePage } from './pages/WeaknessProfilePage'
import { StudyStub } from './pages/StudyStub'
import { DiagnosticPage } from './pages/DiagnosticPage'
import { ClozePage } from './pages/ClozePage'
import { MatchingPage } from './pages/MatchingPage'
import { ComprehensionPage } from './pages/ComprehensionPage'
import { IdiomPage } from './pages/IdiomPage'
import { PhraseMatchPage } from './pages/PhraseMatchPage'
import { SessionPage } from './pages/SessionPage'
import { AuthForm } from './components/AuthForm'
import { ProtectedRoute } from './components/ProtectedRoute'

// Phase 2.3 (card t_ffe6d6af) + Phase 3.2 (card t_64055c49) +
// Phase 4.5 (card t_4a9f172e) + Phase 5.6 (card t_f9375354):
// top-level router.
//
// Route map:
//   /                 public   Phase 1 search/filter UI
//   /login            public   auth form (login)
//   /signup           public   auth form (signup)
//   /weakness-profile protected Phase 2.4 10-axis slider form
//   /diagnostic       protected Phase 3.2 multi-step probe
//   /study            protected placeholder; Phase 5+ fills it in
//   /exercises/cloze  protected Phase 4.5 minimal cloze surface
//   /exercises/session protected Phase 9.6 session mixer; lands users
//                              whose due_by_type union has any
//                              nonzero bucket (gate widens Phase 5.6)
//   /exercises/due    protected Phase 5.6 gate-target; mounts
//                              ClozePage so users with due cards land
//                              in the actual study flow. Phase 5.5
//                              (card t_f253456b) may swap the mount
//                              point for a dedicated component.
//
// /weakness-profile, /diagnostic, /study, /exercises/cloze, and
// /exercises/due are gated. The existing Anki-deck flow on /
// stays open (per the Phase 2.3 hard rule).
//
// Phase 4.5 also adds the global sonner <Toaster /> so the
// cloze-page submit toast (and any future 5xx toasts) has a
// mount point. Dark theme to match the slate background.

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <Toaster
        theme="dark"
        position="bottom-right"
        toastOptions={{
          style: {
            background: 'rgb(15 23 42)', // slate-900
            color: 'rgb(226 232 240)', // slate-200
            border: '1px solid rgb(30 41 59)', // slate-800
          },
        }}
      />
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
            path="/diagnostic"
            element={
              <ProtectedRoute>
                {() => <DiagnosticPage />}
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
          <Route
            path="/exercises/cloze"
            element={
              <ProtectedRoute>
                {(user) => <ClozePage user={user} />}
              </ProtectedRoute>
            }
          />
          <Route
            path="/exercises/match"
            element={
              <ProtectedRoute>
                {() => <MatchingPage />}
              </ProtectedRoute>
            }
          />
          <Route
            path="/exercises/comprehension"
            element={
              <ProtectedRoute>
                {() => <ComprehensionPage />}
              </ProtectedRoute>
            }
          />
          <Route
            path="/exercises/idiom"
            element={
              <ProtectedRoute>
                {() => <IdiomPage />}
              </ProtectedRoute>
            }
          />
          <Route
            path="/exercises/phrase_match"
            element={
              <ProtectedRoute>
                {() => <PhraseMatchPage />}
              </ProtectedRoute>
            }
          />
          <Route
            path="/exercises/session"
            element={
              <ProtectedRoute>
                {() => <SessionPage />}
              </ProtectedRoute>
            }
          />
          <Route
            path="/exercises/due"
            element={
              <ProtectedRoute>
                {(user) => <ClozePage user={user} />}
              </ProtectedRoute>
            }
          />
        </Route>
      </Routes>
    </BrowserRouter>
  </StrictMode>,
)