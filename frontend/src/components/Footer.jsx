import './Footer.css'

function Footer() {
    return (
        <footer className="footer">
            <div className="footer__inner">

                <div className="footer__top">
                    <div className="footer__brand">
                        <h2>Egret</h2>
                        <p>Habitat suitability modeling for conservation research.</p>
                    </div>

                    <nav className="footer__links">
                        <div className="footer__col">
                            <span className="footer__heading">Product</span>
                            <a href="/dashboard">Dashboard</a>
                            <a href="/docs">Documentation</a>
                            <a href="/api">API</a>
                        </div>

                        <div className="footer__col">
                            <span className="footer__heading">Company</span>
                            <a href="/contact">Contact</a>
                            <a href="/privacy">Privacy Policy</a>
                            <a href="/terms">Terms of Service</a>
                        </div>
                    </nav>
                </div>

                <div className="footer__divider" />

                <div className="footer__bottom">
                    <p>&copy; {new Date().getFullYear()} Egret. All rights reserved.</p>

                    <ul className="footer__socials">
                        <li>
                            <a href="#" aria-label="LinkedIn">
                                <i className="fa-brands fa-linkedin"></i>
                            </a>
                        </li>
                        <li>
                            <a href="#" aria-label="GitHub">
                                <i className="fa-brands fa-github"></i>
                            </a>
                        </li>
                        <li>
                            <a href="#" aria-label="Twitter/X">
                                <i className="fa-brands fa-x-twitter"></i>
                            </a>
                        </li>
                    </ul>
                </div>

            </div>
        </footer>
    )
}

export default Footer