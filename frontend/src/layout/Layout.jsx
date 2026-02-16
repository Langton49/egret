import { Outlet, useLocation } from "react-router-dom";
import SideMenu from "../components/SideMenu";
import { useState, useRef } from "react";
import './Layout.css'

function Layout() {
  const [menuOpen, setMenuOpen] = useState(false);
  const location = useLocation();
  const isMapMode = location.pathname === '/demo/map';
  const prevPathRef = useRef(location.pathname);

  if (location.pathname !== prevPathRef.current) {
    prevPathRef.current = location.pathname;
    if (isMapMode && menuOpen) {
      setMenuOpen(false);
    }
  }

  return (
    <div className="layout">
      <SideMenu
        isOpen={menuOpen}
        onToggle={() => setMenuOpen(!menuOpen)}
        isMapMode={isMapMode}
      />
      <main className={`main-area ${isMapMode ? 'map-mode' : ''}`}>
        {isMapMode && (
          <div
            className={`blur-overlay ${menuOpen ? 'visible' : ''}`}
            onClick={() => setMenuOpen(false)}
          />
        )}
        <Outlet />
      </main>
    </div>
  )
}

export default Layout