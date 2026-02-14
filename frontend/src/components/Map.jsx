import { useEffect, useRef, useState } from 'react';
import mapboxgl from 'mapbox-gl';
import MapboxDraw from '@mapbox/mapbox-gl-draw';
import 'mapbox-gl/dist/mapbox-gl.css';
import '@mapbox/mapbox-gl-draw/dist/mapbox-gl-draw.css';
import './Map.css';

mapboxgl.accessToken = import.meta.env.VITE_MAPBOX;
const backendUrl = import.meta.env.VITE_BACKEND;

function Map() {
    const mapRef = useRef(null);
    const mapContainerRef = useRef(null);
    const drawRef = useRef(null);
    const [aoi, setAoi] = useState(null);
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState(null);

    useEffect(() => {
        mapRef.current = new mapboxgl.Map({
            container: mapContainerRef.current,
            style: "mapbox://styles/mapbox/standard",
            center: [-89.85, 29.5],
            zoom: 8
        });

        const draw = new MapboxDraw({
            displayControlsDefault: false,
            controls: {
                polygon: true,
                rectangle: true,
                trash: true
            },
            defaultMode: 'simple_select'
        });

        drawRef.current = draw;
        mapRef.current.addControl(draw, 'top-left');

        const updateAOI = () => {
            const data = draw.getAll();
            if (data.features.length > 0) {
                setAoi(data);
            } else {
                setAoi(null);
                setResults(null);
            }
        };

        mapRef.current.on('draw.create', updateAOI);
        mapRef.current.on('draw.update', updateAOI);
        mapRef.current.on('draw.delete', updateAOI);

        return () => {
            if (mapRef.current) {
                mapRef.current.remove();
            }
        };
    }, []);

    useEffect(() => {
        const sidebar = document.querySelector('.sidemenu');
        if (!sidebar || !mapRef.current) return;

        const handleTransitionEnd = (e) => {
            if (e.propertyName === 'width') {
                mapRef.current.resize();
            }
        };

        sidebar.addEventListener('transitionend', handleTransitionEnd);
        return () => sidebar.removeEventListener('transitionend', handleTransitionEnd);
    }, []);

    const handleAnalyze = async () => {
        if (!aoi) return;

        setLoading(true);
        try {
            const response = await fetch(`${backendUrl}/aoidata/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ aoi: aoi }),
            });
            const data = await response.json();
            console.log('Analysis result:', data);
            setResults(data);
        } catch (err) {
            console.error('Analysis failed:', err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div id="map-container" ref={mapContainerRef}>
            {aoi && (
                <button
                    className="analyze-btn"
                    onClick={handleAnalyze}
                    disabled={loading}
                >
                    {loading ? 'Analyzing...' : 'Analyze Area'}
                </button>
            )}

            {results && (
                <div className="results-panel">
                    <div className="results-header">
                        <h3>Area Overview</h3>
                        <button onClick={() => setResults(null)}>✕</button>
                    </div>

                    {results.area_km2 && (
                        <p>{results.area_km2} km²</p>
                    )}

                    <div className="result-card">
                        <h4>Condition</h4>
                        <p>{results.condition}</p>
                        <p>{results.condition_detail}</p>
                    </div>

                    <div className="result-card">
                        <h4>Sightings</h4>
                        <p>{results.total_sightings.toLocaleString()} total observations</p>
                        <p>{results.species_count} species recorded</p>
                    </div>

                    <div className="result-card">
                        <h4>Diversity</h4>
                        <p>{results.diversity_level}</p>
                    </div>

                    <div className="result-card">
                        <h4>Trends</h4>
                        <p>{results.vegetation_trend}</p>
                        <p>{results.water_trend}</p>
                    </div>

                    <div className="result-card">
                        <h4>Notable Change</h4>
                        <p>{results.notable_change}</p>
                    </div>

                    {results.top_species.length > 0 && (
                        <div className="result-card">
                            <h4>Top Orders</h4>
                            {results.top_species.map((s, i) => (
                                <p key={i}>{s.order}: {s.count.toLocaleString()}</p>
                            ))}
                        </div>
                    )}

                    <div className="result-card">
                        <h4>Data Coverage</h4>
                        <p>{results.data_coverage}</p>
                    </div>
                </div>
            )}
        </div>
    );
}

export default Map;