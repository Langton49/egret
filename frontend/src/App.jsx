import LandingPage from './pages/LandingPage'
import Dashboard from './pages/Dashboard'
import MapPage from './pages/MapPage'
import Reports from './pages/Reports'
import Settings from './pages/Settings'
import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './layout/Layout'

function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />

      {/* Redirect /demo → / */}
      <Route path="/demo" element={<Navigate to="/" replace />} />

      {/* Keep nested routes only for subpages */}
      <Route path="/demo" element={<Layout />}>
        <Route path="map" element={<MapPage />} />
      </Route>
    </Routes>
  )
}

export default App