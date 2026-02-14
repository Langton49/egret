import HeroSection from "../components/HeroSection";
import MainSection from "../components/MainSection";
import "./LandingPage.css"

function LandingPage () {

    return (
        <>
        <div className="landing_page">
            <HeroSection/>
            <MainSection/>
        </div>
        
        </>
    );

}

export default LandingPage;