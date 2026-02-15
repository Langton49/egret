import { useRef, useState } from 'react'
import "./HeroSection.css";
import northernHarrier from "../assets/northern_harrier.png"
import woodStork from '../assets/wood_stork.png'
import leastTern from '../assets/least_tern.png'
import triHeron from '../assets/tri_heron.png'
import osprey from '../assets/osprey.png'
import seasideSparrow from '../assets/seaside_sparrow.png'
import blackSkimmer from '../assets/black_skimmer.png'
import clapperRail from '../assets/clapper_rail.png'
import pWarbler from '../assets/p_warbler.png'
import brownPelican from '../assets/brown_pelican.png'
import rSpoonbill from '../assets/r_spoonbill.png'
import greatEgret from '../assets/great_egret.png'
import greatEgretCall from '../assets/great_egret.mp3'
import spoonbillCall from '../assets/roseatte_spoonbill.mp3'
import brownPelicanCall from '../assets/brown_pelican.mp3'
import warblerCall from '../assets/warbler.mp3'
import railCall from '../assets/clapper_rail.mp3'
import skimmerCall from '../assets/black_skimmer.mp3'
import sparrowCall from '../assets/seaside_sparrow.mp3'
import ospreyCall from '../assets/osprey.mp3'
import heronCall from '../assets/tricolored_heron.mp3'
import ternCall from '../assets/least_tern.mp3'
import storkCall from '../assets/wood_stork.mp3'
import harrierCall from '../assets/northern_harrier.mp3'

function HeroSection() {
    const [audioEnabled, setAudioEnabled] = useState(false)
    const audioEnabledRef = useRef(false)
    const currentCall = useRef(null)

    const enableAudio = () => {
        setAudioEnabled(true)
        audioEnabledRef.current = true
    }

    const disableAudio = () => {
        stopCall()
        setAudioEnabled(false)
        audioEnabledRef.current = false
    }

    const playCall = (sound) => {
        if (!audioEnabledRef.current) return
        if (currentCall.current) {
            currentCall.current.pause()
            currentCall.current.currentTime = 0
        }
        currentCall.current = new Audio(sound)
        currentCall.current.volume = 0.4
        currentCall.current.play().catch(() => {})
    }

    const stopCall = () => {
        if (currentCall.current) {
            currentCall.current.pause()
            currentCall.current.currentTime = 0
            currentCall.current = null
        }
    }

    const birds = [
        {
            name: "Great Egret", latin: "Ardea alba",
            image: greatEgret, sound: greatEgretCall,
            x: 0.62, y: 0.12, size: 34, type: "heron"
        },
        {
            name: "Roseate Spoonbill", latin: "Platalea ajaja",
            image: rSpoonbill, sound: spoonbillCall,
            x: 0.20, y: 0.65, size: 30, type: "wader"
        },
        {
            name: "Brown Pelican", latin: "Pelecanus occidentalis",
            image: brownPelican, sound: brownPelicanCall,
            x: 0.42, y: 0.15, size: 36, type: "seabird"
        },
        {
            name: "Prothonotary Warbler", latin: "Protonotaria citrea",
            image: pWarbler, sound: warblerCall,
            x: 0.12, y: 0.32, size: 22, type: "songbird"
        },
        {
            name: "Clapper Rail", latin: "Rallus crepitans",
            image: clapperRail, sound: railCall,
            x: 0.10, y: 0.90, size: 24, type: "rail"
        },
        {
            name: "Black Skimmer", latin: "Rynchops niger",
            image: blackSkimmer, sound: skimmerCall,
            x: 0.72, y: 0.72, size: 28, type: "seabird"
        },
        {
            name: "Seaside Sparrow", latin: "Ammospiza maritima",
            image: seasideSparrow, sound: sparrowCall,
            x: 0.85, y: 0.9, size: 22, type: "songbird"
        },
        {
            name: "Osprey", latin: "Pandion haliaetus",
            image: osprey, sound: ospreyCall,
            x: 0.85, y: 0.12, size: 32, type: "raptor"
        },
        {
            name: "Tricolored Heron", latin: "Egretta tricolor",
            image: triHeron, sound: heronCall,
            x: 0.38, y: 0.80, size: 30, type: "heron"
        },
        {
            name: "Least Tern", latin: "Sternula antillarum",
            image: leastTern, sound: ternCall,
            x: 0.28, y: 0.1, size: 20, type: "seabird"
        },
        {
            name: "Wood Stork", latin: "Mycteria americana",
            image: woodStork, sound: storkCall,
            x: 0.82, y: 0.38, size: 32, type: "wader"
        },
        {
            name: "Northern Harrier", latin: "Circus hudsonius",
            image: northernHarrier, sound: harrierCall,
            x: 0.90, y: 0.64, size: 28, type: "raptor"
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
                            onMouseEnter={() => playCall(bird.sound)}
                            onMouseLeave={stopCall}
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
                    <button
                        className="sound_toggle"
                        onClick={() => {
                            if (audioEnabled) {
                                disableAudio()
                            } else {
                                enableAudio()
                            }
                        }}
                    >
                        {audioEnabled ? '🔊 Bird Calls On' : '🔇 Turn On Bird Calls'}
                    </button>
                </div>
            </div>
        </>
    )
}

export default HeroSection;