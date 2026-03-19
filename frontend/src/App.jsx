import { lazy, Suspense } from 'react'
import LandingPage from './pages/LandingPage'
import MapPage from './pages/MapPage'
import AoiDashboard from './pages/AoiDashboard'
import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './layout/Layout'

const ResearchPage = lazy(() => import('./pages/ResearchPage'))

function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />

      {/* Legacy demo routes → research */}
      <Route path="/demo" element={<Navigate to="/research" replace />} />
      <Route path="/demo/map" element={<Navigate to="/research" replace />} />
      <Route path="/demo/aoi" element={<Navigate to="/research" replace />} />

      <Route
        path="/research"
        element={
          <Suspense fallback={<div style={{ color: '#a8d8a8', padding: 24 }}>Loading…</div>}>
            <ResearchPage />
          </Suspense>
        }
      />
    </Routes>
  )
}

export default App