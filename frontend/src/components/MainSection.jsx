import { useEffect } from 'react'
import './MainSection.css'
import introVid from '../assets/IntroGif.gif'
import hiwVid from '../assets/hiwVid.gif'
import binoculars from '../assets/binoculars.png'
import brain from '../assets/brain.png'
import satelite from '../assets/satelite.png'
import  copernicusLogo from '../assets/cop_logo.png'
import ebirdLogo from '../assets/ebird_logo.png'
import inatLogo from '../assets/inat_logo.png'

function MainSection () {

    useEffect(() => {
        const sections = document.querySelectorAll('.intro, .hiw')

        const observer = new IntersectionObserver(
            (entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('show')
                        observer.unobserve(entry.target) // animate once
                    }
                })
            },
            { threshold: 0.25 }
        )

        sections.forEach(section => observer.observe(section))

        return () => observer.disconnect()
    }, [])

    return (
        <div className="main_section">
            <div className="intro">
                <div className="intro_text">
                    <h1>What Is It?</h1>
                    <p>
                        Conservation efforts are costly and can take weeks to months of research to project scope.
                        The problem is where would conservation have the highest impact especially for wildlife.
                        Egret is a new approach to conservation research and prioritization.
                    </p>
                </div>

                <div className="intro_vid">
                    <img src={introVid} alt="" />
                </div>
            </div>

            <div className="hiw">
                <div className="hiw_vid">
                    <img src={hiwVid} alt="" />
                </div>

                <div className="hiw_text">
                    <h1>How It Works?</h1>
                    <p>
                        Pairing high-value satellite imagery data with public bird observations from citizen scientists
                        and official datasets, Egret makes it easy to evaluate an area for its potential to be a valuable nesting ground.
                        Using machine learning, the Egret dashboard allows you to get an overview of how an area is behaving and
                        whether it could be harboring avian species.
                    </p>
                </div>
            </div>
           <div className='hiw_panels'>
                <div className='desc'>
                    <img src={binoculars} alt="Binoculars" />
                    <p>Citizen science bird observation data</p>
                </div>
                <div className='desc'>
                    <img src={satelite} alt="Satellite" />
                    <p>High-resolution satellite imagery</p>
                </div>
                <div className='desc'>
                    <img src={brain} alt="Brain" />
                    <p>Machine learning habitat analysis</p>
                </div>
            </div>
            <div className='built_with'>
                <h1>Built With</h1>
                <div>
                    <img src={ebirdLogo} alt="" />
                    <img src={inatLogo} alt="" />
                    <img src={copernicusLogo} alt="" />
                </div>

            </div>
            <footer>
                HEllo
            </footer>
        </div>
    )
}

export default MainSection
