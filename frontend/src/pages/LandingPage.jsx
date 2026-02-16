import HeroSection from "../components/HeroSection";
import MainSection from "../components/MainSection";
import Footer from "../components/Footer";
import "./LandingPage.css"

function LandingPage () {

    return (
        <>
        <div className="landing_page">
            <HeroSection/>
            <MainSection/>
            <Footer/>
        </div>
        
        </>
    );

}

export default LandingPage;