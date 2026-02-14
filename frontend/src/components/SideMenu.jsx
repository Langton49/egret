import { useState } from "react";
import { NavLink } from "react-router-dom";
import './SideMenu.css'

function SideMenu (){
    const [collapsed, setCollapsed] = useState(false);

    return (
        <div className={`sidemenu ${collapsed ? 'collapsed' : ''}`}>
            <div className="sidemenu_header">
                <div className="brand">Egret</div>
                <button 
                    className={`sidemenu_collapse ${collapsed ? '' : 'collapsed'}`}
                    onClick={() => setCollapsed(!collapsed)}
                >
                    <span className="bar bar-top"></span>
                    <span className="bar bar-mid"></span>
                    <span className="bar bar-bot"></span>
                </button>
            </div>
            <nav className="nav_panel">
            <NavLink to="/dashboard" className="nav-btn"><div>Dashboard</div></NavLink>
            <NavLink to="/dashboard/reports" className='nav-btn'><div>Reports</div></NavLink>
            <NavLink to="/dashboard/settings" className="nav-btn"><div>Settings</div></NavLink>
            </nav>
        </div>
        
    )
}

export default SideMenu;