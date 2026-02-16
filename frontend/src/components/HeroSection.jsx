import { useRef, useState, useEffect } from 'react'
import { NavLink } from "react-router-dom";
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
import egretVid from '../assets/egret.gif'
import spoonBillVid from '../assets/roseatte_spoonbill.gif'
import pelicanVid from '../assets/brown_pelican.gif'
import warblerVid from '../assets/warbler.gif'
import railVid from '../assets/rail.gif'
import skimmerVid from '../assets/black_skimmer.gif'
import sparrowVid from '../assets/sparrow.gif'
import ospreyVid from '../assets/osprey.gif'
import heronVid from '../assets/heron.gif'
import ternVid from '../assets/least_tern.gif'
import storkVid from '../assets/wood_stork.gif'
import harrierVid from '../assets/northern_harrier.gif'

function HeroSection() {
    const [audioEnabled, setAudioEnabled] = useState(false)
    const [hoveredBird, setHoveredBird] = useState(null)
    const [displayedVid, setDisplayedVid] = useState(null)
    const audioEnabledRef = useRef(false)
    const currentCall = useRef(null)
    const fadeTimeout = useRef(null)

    useEffect(() => {
        if (hoveredBird) {
            if (fadeTimeout.current) {
                clearTimeout(fadeTimeout.current)
                fadeTimeout.current = null
            }
            setDisplayedVid(hoveredBird.vid)
        } else {
            fadeTimeout.current = setTimeout(() => {
                setDisplayedVid(null)
                fadeTimeout.current = null
            }, 600)
        }

        return () => {
            if (fadeTimeout.current) {
                clearTimeout(fadeTimeout.current)
            }
        }
    }, [hoveredBird])

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

    /*
     * Bird positions arranged in a loose elliptical constellation
     * around the center CTA, with larger species pushed outward
     * and smaller songbirds tucked closer in.
     *
     * The layout reads roughly as:
     *
     *        tern       pelican    egret      osprey
     *     warbler                                stork
     *        spoonbill                        harrier
     *     rail          heron      skimmer    sparrow
     */
    const birds = [
        {
            name: "Great Egret", latin: "Ardea alba",
            image: greatEgret, sound: greatEgretCall,
            vid: egretVid,
            x: 0.62, y: 0.10, size: 34, type: "heron"
        },
        {
            name: "Roseate Spoonbill", latin: "Platalea ajaja",
            image: rSpoonbill, sound: spoonbillCall,
            vid: spoonBillVid,
            x: 0.14, y: 0.62, size: 30, type: "wader"
        },
        {
            name: "Brown Pelican", latin: "Pelecanus occidentalis",
            image: brownPelican, sound: brownPelicanCall,
            vid: pelicanVid,
            x: 0.42, y: 0.08, size: 36, type: "seabird"
        },
        {
            name: "Prothonotary Warbler", latin: "Protonotaria citrea",
            image: pWarbler, sound: warblerCall,
            vid: warblerVid,
            x: 0.08, y: 0.34, size: 22, type: "songbird"
        },
        {
            name: "Clapper Rail", latin: "Rallus crepitans",
            image: clapperRail, sound: railCall,
            vid: railVid,
            x: 0.10, y: 0.88, size: 24, type: "rail"
        },
        {
            name: "Black Skimmer", latin: "Rynchops niger",
            image: blackSkimmer, sound: skimmerCall,
            vid: skimmerVid,
            x: 0.68, y: 0.82, size: 28, type: "seabird"
        },
        {
            name: "Seaside Sparrow", latin: "Ammospiza maritima",
            image: seasideSparrow, sound: sparrowCall,
            vid: sparrowVid,
            x: 0.88, y: 0.86, size: 22, type: "songbird"
        },
        {
            name: "Osprey", latin: "Pandion haliaetus",
            image: osprey, sound: ospreyCall,
            vid: ospreyVid,
            x: 0.88, y: 0.10, size: 32, type: "raptor"
        },
        {
            name: "Tricolored Heron", latin: "Egretta tricolor",
            image: triHeron, sound: heronCall,
            vid: heronVid,
            x: 0.40, y: 0.85, size: 30, type: "heron"
        },
        {
            name: "Least Tern", latin: "Sternula antillarum",
            image: leastTern, sound: ternCall,
            vid: ternVid,
            x: 0.22, y: 0.08, size: 20, type: "seabird"
        },
        {
            name: "Wood Stork", latin: "Mycteria americana",
            image: woodStork, sound: storkCall,
            vid: storkVid,
            x: 0.90, y: 0.38, size: 32, type: "wader"
        },
        {
            name: "Northern Harrier", latin: "Circus hudsonius",
            image: northernHarrier, sound: harrierCall,
            vid: harrierVid,
            x: 0.88, y: 0.62, size: 28, type: "raptor"
        }
    ];

    return (
        <>
            <div className="hero">
                <div
                    className={`hero-bg-vid ${hoveredBird ? 'visible' : ''}`}
                    style={{
                        backgroundImage: displayedVid ? `url(${displayedVid})` : 'none',
                    }}
                />

                {/* Bird name tooltip */}
                {hoveredBird && (
                    <div
                        className="bird-tooltip"
                        style={{
                            left: `${hoveredBird.x * 100}%`,
                            top: `${hoveredBird.y * 100}%`,
                        }}
                    >
                        <span className="bird-tooltip__name">{hoveredBird.name}</span>
                        <span className="bird-tooltip__latin">{hoveredBird.latin}</span>
                    </div>
                )}

                <div className="ind_bird">
                    {birds.map((bird, i) => (
                        <img
                            src={bird.image}
                            key={i}
                            alt={bird.name}
                            className="bird-sprite"
                            onMouseEnter={() => {
                                setHoveredBird(bird)
                                playCall(bird.sound)
                            }}
                            onMouseLeave={() => {
                                setHoveredBird(null)
                                stopCall()
                            }}
                            style={{
                                position: 'absolute',
                                left: `${bird.x * 100}%`,
                                top: `${bird.y * 100}%`,
                                transform: 'translate(-50%, -50%)',
                                width: bird.size * 2,
                                height: "auto",
                                cursor: "pointer",
                                animationDelay: `${i * 0.12}s`,
                            }}
                        />
                    ))}
                </div>

                <div className={`cta ${hoveredBird ? 'dimmed' : ''}`}>
                    <p className="main_phrase">
                        Know the land<br />
                        <span>before the loss.</span>
                    </p>
                    <p className="subphrase">
                        Predict habitat suitability for nesting bird species — so conservation acts on evidence, not intuition.
                    </p>
                    <div className="cta-btns">
                        <NavLink to="/" className="nav_btn nav_btn--secondary">
                            How It Works
                        </NavLink>
                        <NavLink to="/demo/map" className="nav_btn nav_btn--primary">
                            View Demo
                        </NavLink>
                    </div>
                    <div
                        className={`audio-indicator ${audioEnabled ? "active" : ""}`}
                        onClick={() => (audioEnabled ? disableAudio() : enableAudio())}
                    >
                        <div className="audio-bars">
                            <div className="audio-bar"></div>
                            <div className="audio-bar"></div>
                            <div className="audio-bar"></div>
                            <div className="audio-bar"></div>
                            <div className="audio-bar"></div>
                        </div>
                        <span className="audio-label">
                            {audioEnabled ? "Soundscape On" : "Enable Soundscape"}
                        </span>
                    </div>
                </div>
            </div>
        </>
    )
}

export default HeroSection;