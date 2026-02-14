import { useEffect, useRef, useState } from 'react';
import mapboxgl from 'mapbox-gl';
import MapboxDraw from '@mapbox/mapbox-gl-draw';
import 'mapbox-gl/dist/mapbox-gl.css';
import '@mapbox/mapbox-gl-draw/dist/mapbox-gl-draw.css';
import './Map.css';

mapboxgl.accessToken = import.meta.env.VITE_MAPBOX;

function Map() {
    const mapRef = useRef(null);
    const mapContainerRef = useRef(null);
    const drawRef = useRef(null);
    const [aoi, setAoi] = useState(null);
    const [loading, setLoading] = useState(false);

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
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ aoi: aoi }),
            });
            const data = await response.json();
            console.log('Analysis result:', data);
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
        </div>
    );
}

export default Map;