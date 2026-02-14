import "./HeroSection.css";
import northernHarrier from "../assets/northern_harrier.png"
import woodStork from '../assets/wood_stork.png'
import leastTern from '../assets/least_tern.png'
import triHeron from '../assets/tri_heron.png'
import osprey from '../assets/osprey.png'
function HeroSection () {

    const birds = [
    {
        name: "Great Egret", latin: "Ardea alba",
        image: northernHarrier,
        x: 0.85, y: 0.12, size: 34, type: "heron"
    },
    {
        name: "Roseate Spoonbill", latin: "Platalea ajaja",
        image: northernHarrier,
        x: 0.20, y: 0.65, size: 30, type: "wader"
    },
    {
        name: "Brown Pelican", latin: "Pelecanus occidentalis",
        image: northernHarrier,
        x: 0.90, y: 0.64, size: 36, type: "seabird"
    },
    {
        name: "Prothonotary Warbler", latin: "Protonotaria citrea",
        image: northernHarrier,
        x: 0.12, y: 0.32, size: 22, type: "songbird"
    },
    {
        name: "Clapper Rail", latin: "Rallus crepitans",
        image: northernHarrier,
        x: 0.10, y: 0.90, size: 24, type: "rail"
    },
    {
        name: "Black Skimmer", latin: "Rynchops niger",
        image: northernHarrier,
        x: 0.72, y: 0.72, size: 28, type: "seabird"
    },
    {
        name: "Seaside Sparrow", latin: "Ammospiza maritima",
        image: northernHarrier,
        x: 0.28, y: 0.1, size: 20, type: "songbird"
    },
    {
        name: "Osprey", latin: "Pandion haliaetus",
        image: osprey,
        x: 0.62, y: 0.12, size: 32, type: "raptor"
    },
    {
        name: "Tricolored Heron", latin: "Egretta tricolor",
        image: triHeron,
        x: 0.38, y: 0.80, size: 30, type: "heron"
    },
    {
        name: "Least Tern", latin: "Sternula antillarum",
        image: leastTern,
        x: 0.85, y: 0.9, size: 22, type: "seabird"
    },
    {
        name: "Wood Stork", latin: "Mycteria americana",
        image: woodStork,
        x: 0.82, y: 0.38, size: 32, type: "wader"
    },
    {
        name: "Northern Harrier", latin: "Circus hudsonius",
        image: northernHarrier,
        x: 0.42, y: 0.15, size: 28, type: "raptor"
    }
];
    return (
        <>
        <div className="hero">
            <div className="ind_bird">
            {birds.map((bird, i) => (
                <img 
                    src={bird.image}
                    key={i}
                    alt={bird.name}
                    style={{
                        position: 'absolute',
                        left: `${bird.x * 100}%`,
                        top: `${bird.y * 100}%`,
                        transform: 'translate(-50%, -50%)',
                        width: bird.size * 2,
                        height: "auto",
                        cursor: 'pointer',
                    }}
                />
            ))}
            </div>
            <div className="cta">
                <p className="main_phrase"><span>Prioritization</span> made easier</p>
                <p className="subphrase">Protect Louisiana's avian species with this GIS centric tool for analysis and insights deriviation.</p>
                <div>
                    <button>How It Works</button>
                    <button>View Demo</button>
                </div>
                
            </div>
        </div>
        </>
    )
}

export default HeroSection;