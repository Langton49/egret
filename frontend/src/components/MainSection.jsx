import { useEffect } from 'react'
import './MainSection.css'
import introVid from '../assets/IntroGif.gif'
import hiwVid from '../assets/hiwVid.gif'
import binoculars from '../assets/binoculars.png'
import brain from '../assets/brain.png'
import satelite from '../assets/satelite.png'
import copernicusLogo from '../assets/cop_logo.png'
import ebirdLogo from '../assets/ebird_logo.png'
import inatLogo from '../assets/inat_logo.png'

function MainSection() {

    useEffect(() => {
        const sections = document.querySelectorAll(
            '.intro, .approach, .pillars, .built-with'
        )

        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('show')
                        observer.unobserve(entry.target)
                    }
                })
            },
            { threshold: 0.2 }
        )

        sections.forEach(section => observer.observe(section))
        return () => observer.disconnect()
    }, [])

    return (
        <div className="main-section">

            {/* ── The Problem ── */}
            <section className="intro">
                <div className="intro__text">
                    <span className="section-label">The Problem</span>
                    <h2>
                        Conservation moves slowly.<br />
                        Habitat loss does not.
                    </h2>
                    <p>
                        Identifying where protection or restoration will yield the greatest
                        ecological return typically requires months of field surveys,
                        fragmented datasets, and costly analysis. Meanwhile, critical
                        landscapes quietly degrade. Egret closes that gap — giving
                        researchers and decision-makers a faster, evidence-based lens on
                        habitat potential before the window closes.
                    </p>
                </div>

                <div className="intro__media">
                    <img src={introVid} alt="Wetland landscape" />
                </div>
            </section>

            {/* ── The Approach ── */}
            <section className="approach">
                <div className="approach__media">
                    <img src={hiwVid} alt="Snowy egret in habitat" />
                </div>

                <div className="approach__text">
                    <span className="section-label">The Approach</span>
                    <h2>
                        Satellite signal meets<br />
                        ecological ground truth.
                    </h2>
                    <p>
                        Egret models habitat suitability by fusing spectral indices derived
                        from satellite imagery — vegetation health, surface moisture,
                        canopy density — with millions of verified bird observations. A
                        gradient-boosting model learns which environmental signatures
                        consistently predict nesting activity, then scores unsurveyed
                        landscapes accordingly.
                    </p>
                </div>
            </section>

            {/* ── Three Pillars ── */}
            <section className="pillars">
                <div className="pillar">
                    <div className="pillar__icon">
                        <img src={binoculars} alt="" aria-hidden="true" />
                    </div>
                    <h3>Observation Data</h3>
                    <p>
                        Millions of georeferenced bird sightings sourced from eBird and iNaturalist,
                        spanning species, date, and location.
                    </p>
                </div>

                <div className="pillar__divider" />

                <div className="pillar">
                    <div className="pillar__icon">
                        <img src={satelite} alt="" aria-hidden="true" />
                    </div>
                    <h3>Spectral Indices</h3>
                    <p>
                        Raw multispectral bands from Copernicus Sentinel-2, processed
                        locally into indices like NDVI and NDWI to characterize vegetation
                        vigor and hydrological conditions.
                    </p>
                </div>

                <div className="pillar__divider" />

                <div className="pillar">
                    <div className="pillar__icon">
                        <img src={brain} alt="" aria-hidden="true" />
                    </div>
                    <h3>Predictive Modeling</h3>
                    <p>
                        A gradient-boosting classifier trained on paired observation and
                        spectral data, producing continuous suitability scores across any
                        target landscape.
                    </p>
                </div>
            </section>

            {/* ── Built With ── */}
            <section className="built-with">
                <span className="section-label">Built With</span>
                <div className="built-with__logos">
                    <img src={ebirdLogo} alt="eBird" />
                    <img src={inatLogo} alt="iNaturalist" />
                    <img src={copernicusLogo} alt="Copernicus" />
                </div>
            </section>

        </div>
    )
}

export default MainSection