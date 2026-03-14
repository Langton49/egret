import LandingPage from './pages/LandingPage'
import MapPage from './pages/MapPage'
import AoiDashboard from './pages/AoiDashboard'
import { Routes, Route, Navigate } from 'react-router-dom'
import Layout from './layout/Layout'

function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />

      {/* Redirect /demo → / */}
      <Route path="/demo" element={<Navigate to="/" replace />} />

      <Route path="/demo" element={<Layout />}>
        <Route path="map" element={<MapPage />} />
        <Route path="aoi" element={<AoiDashboard />} />
      </Route>
    </Routes>
  )
}

export default App