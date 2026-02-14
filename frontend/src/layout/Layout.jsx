import { Outlet } from "react-router-dom";
import SideMenu from "../components/SideMenu";
import MainContent from "../components/MainContent";
import './Layout.css'

function Layout() {
  return (
    <div className='layout' style={{ display: "flex" }}>
      <SideMenu />
      <main style={{ flex: 1 }}>
        <MainContent />
      </main>
    </div>
  );
}

export default Layout;