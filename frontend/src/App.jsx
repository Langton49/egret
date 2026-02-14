import LandingPage from './pages/LandingPage'
import Dashboard from './pages/Dashboard'
import { Routes, Route } from 'react-router-dom'
import Layout from './layout/Layout'

function App() {

  return (
    <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/dashboard" element={<Layout />}>
        <Route index element={<Dashboard />} />
      </Route>
    </Routes>
  )
}

export default App
