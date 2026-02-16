import { NavLink } from "react-router-dom";
import './SideMenu.css'

function SideMenu({ isOpen, onToggle, isMapMode }) {
  return (
    <div className={`sidemenu ${!isOpen ? 'collapsed' : ''} ${isMapMode ? 'overlay-mode' : ''}`}>
    <div className="sidemenu_header">
        <div className="brand">Egret</div>
        <button
          className={`sidemenu_collapse ${isOpen ? 'collapsed' : ''}`}
          onClick={onToggle}
        >
          <span className="bar bar-top"></span>
          <span className="bar bar-mid"></span>
          <span className="bar bar-bot"></span>
        </button>
      </div>
      <div className="announcement">
              <div className="announcement_header">FYI</div>
              <p className="announcement-hint">
                This is a <strong>demo</strong> version of Egret showing the capabilities of our habitat scoring model.
              </p>
              <p className="announcement-hint">
                 Feel free to look around while we work on getting you more features.
              </p>
        </div>
      {/* <nav className="nav_panel">
        <NavLink to="/dashboard" end className="nav-btn"><div>Dashboard</div></NavLink>
        <NavLink to="/dashboard/map" className="nav-btn"><div>Map</div></NavLink>
        <NavLink to="/dashboard/reports" className="nav-btn"><div>Reports</div></NavLink>
        <NavLink to="/dashboard/settings" className="nav-btn"><div>Settings</div></NavLink>
      </nav> */}
    </div>
  )
}

export default SideMenu