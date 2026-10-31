import LandingPage from './pages/LandingPage'
import MapPage from './pages/MapPage'
import AoiDashboard from './pages/AoiDashboard'
import { Routes, Route, Navigate } from 'react-router-dom'

function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/map" element={<MapPage />} />
      <Route path="/map/score" element={<AoiDashboard />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

export default App
