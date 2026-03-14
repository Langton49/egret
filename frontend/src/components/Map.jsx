import { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import mapboxgl from 'mapbox-gl';
import MapboxDraw from '@mapbox/mapbox-gl-draw';
import 'mapbox-gl/dist/mapbox-gl.css';
import '@mapbox/mapbox-gl-draw/dist/mapbox-gl-draw.css';
import './Map.css';

mapboxgl.accessToken = import.meta.env.VITE_MAPBOX;
const backendUrl = import.meta.env.VITE_BACKEND;

const AOI_BOUNDS = [-90.628342, 28.927421, -89.067224, 30.106372];
const [SW_LNG, SW_LAT, NE_LNG, NE_LAT] = AOI_BOUNDS;

const AOI_BOUNDARY_GEOJSON = {
    type: 'Feature',
    geometry: {
        type: 'Polygon',
        coordinates: [[
            [SW_LNG, SW_LAT],
            [NE_LNG, SW_LAT],
            [NE_LNG, NE_LAT],
            [SW_LNG, NE_LAT],
            [SW_LNG, SW_LAT],
        ]],
    },
};

const isWithinBounds = (feature) => {
    const coords = feature.geometry.coordinates.flat(Infinity);
    for (let i = 0; i < coords.length; i += 2) {
        const lng = coords[i];
        const lat = coords[i + 1];
        if (lng < SW_LNG || lng > NE_LNG || lat < SW_LAT || lat > NE_LAT) {
            return false;
        }
    }
    return true;
};

function Map({ onDrawReady }) {
    const navigate = useNavigate();
    const mapRef = useRef(null);
    const mapContainerRef = useRef(null);
    const drawRef = useRef(null);
    const [mapLoaded, setMapLoaded] = useState(false);
    const [aoi, setAoi] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    useEffect(() => {
        mapRef.current = new mapboxgl.Map({
            container: mapContainerRef.current,
            style: 'mapbox://styles/mapbox/standard',
            center: [(SW_LNG + NE_LNG) / 2, (SW_LAT + NE_LAT) / 2],
            zoom: 8,
            minZoom: 7.5,
            maxBounds: [
                [SW_LNG - 0.15, SW_LAT - 0.15],
                [NE_LNG + 0.15, NE_LAT + 0.15],
            ],
        });

        mapRef.current.on('load', () => {
            setMapLoaded(true);
            mapRef.current.fitBounds(
                [[SW_LNG, SW_LAT], [NE_LNG, NE_LAT]],
                { padding: 40, duration: 0 }
            );

            if (!mapRef.current.getSource('aoi-boundary')) {
                mapRef.current.addSource('aoi-boundary', {
                    type: 'geojson',
                    data: AOI_BOUNDARY_GEOJSON,
                });

                mapRef.current.addLayer({
                    id: 'aoi-boundary-fill',
                    type: 'fill',
                    source: 'aoi-boundary',
                    paint: {
                        'fill-color': '#ff6600',
                        'fill-opacity': 0.05,
                    },
                });

                mapRef.current.addLayer({
                    id: 'aoi-boundary-line',
                    type: 'line',
                    source: 'aoi-boundary',
                    paint: {
                        'line-color': '#ff6600',
                        'line-width': 2,
                        'line-dasharray': [3, 2],
                    },
                });
            }
        });

        const draw = new MapboxDraw({
            displayControlsDefault: false,
            controls: {},
            defaultMode: 'simple_select',
        });
        mapRef.current.addControl(draw);
        drawRef.current = draw;
        if (onDrawReady) onDrawReady(draw);

        const canvas = mapRef.current.getCanvas();
        let rectStart = null;
        let rectBox = null;

        const onMouseDown = (e) => {
            if (!e.shiftKey) return;
            e.preventDefault();
            mapRef.current.dragPan.disable();
            rectStart = mapRef.current.unproject([e.offsetX, e.offsetY]);

            rectBox = document.createElement('div');
            rectBox.className = 'rect-draw-box';
            rectBox.style.left = `${e.offsetX}px`;
            rectBox.style.top = `${e.offsetY}px`;
            mapContainerRef.current.appendChild(rectBox);
        };

        const onMouseMove = (e) => {
            if (!rectStart || !rectBox) return;
            const startPoint = mapRef.current.project(rectStart);
            const x = Math.min(startPoint.x, e.offsetX);
            const y = Math.min(startPoint.y, e.offsetY);
            const w = Math.abs(e.offsetX - startPoint.x);
            const h = Math.abs(e.offsetY - startPoint.y);
            rectBox.style.left = `${x}px`;
            rectBox.style.top = `${y}px`;
            rectBox.style.width = `${w}px`;
            rectBox.style.height = `${h}px`;
        };

        const onMouseUp = (e) => {
            if (!rectStart) return;
            mapRef.current.dragPan.enable();

            if (rectBox) {
                rectBox.remove();
                rectBox = null;
            }

            const rectEnd = mapRef.current.unproject([e.offsetX, e.offsetY]);
            const sw = [Math.min(rectStart.lng, rectEnd.lng), Math.min(rectStart.lat, rectEnd.lat)];
            const ne = [Math.max(rectStart.lng, rectEnd.lng), Math.max(rectStart.lat, rectEnd.lat)];
            rectStart = null;

            if (Math.abs(ne[0] - sw[0]) < 0.001 || Math.abs(ne[1] - sw[1]) < 0.001) return;

            const rectFeature = {
                type: 'Feature',
                properties: {},
                geometry: {
                    type: 'Polygon',
                    coordinates: [[
                        [sw[0], sw[1]],
                        [ne[0], sw[1]],
                        [ne[0], ne[1]],
                        [sw[0], ne[1]],
                        [sw[0], sw[1]],
                    ]],
                },
            };

            if (!isWithinBounds(rectFeature)) {
                setError('Please draw within the study area boundary.');
                return;
            }

            draw.deleteAll();
            draw.add(rectFeature);
            setError(null);
            setAoi(draw.getAll());
        };

        canvas.addEventListener('mousedown', onMouseDown);
        canvas.addEventListener('mousemove', onMouseMove);
        canvas.addEventListener('mouseup', onMouseUp);

        const updateAOI = () => {
            const data = draw.getAll();
            if (data.features.length > 0) {
                const lastFeature = data.features[data.features.length - 1];
                if (!isWithinBounds(lastFeature)) {
                    draw.delete(lastFeature.id);
                    setError('Please draw within the study area boundary.');
                    setAoi(null);
                    return;
                }
                setError(null);
                setAoi(data);
            } else {
                setAoi(null);
                setError(null);
            }
        };

        mapRef.current.on('draw.create', updateAOI);
        mapRef.current.on('draw.update', updateAOI);
        mapRef.current.on('draw.delete', updateAOI);

        return () => {
            canvas.removeEventListener('mousedown', onMouseDown);
            canvas.removeEventListener('mousemove', onMouseMove);
            canvas.removeEventListener('mouseup', onMouseUp);
            if (mapRef.current) {
                if (drawRef.current) {
                    mapRef.current.removeControl(drawRef.current);
                    drawRef.current = null;
                }
                mapRef.current.remove();
                mapRef.current = null;
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

    const handleSelectAoi = async () => {
        if (!aoi) return;
        setLoading(true);
        setError(null);
        try {
            const response = await fetch(`${backendUrl}/aoidata/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ aoi }),
            });
            const results = await response.json();
            navigate('/demo/aoi', { state: { aoi, results } });
        } catch (err) {
            console.error('Analysis failed:', err);
            setError('Analysis failed. Try again.');
            setLoading(false);
        }
    };

    return (
        <div id="map-container" ref={mapContainerRef}>
            {!mapLoaded && (
                <div className="map-skeleton">
                    <div className="map-skeleton-shimmer"></div>
                    <div className="map-skeleton-ui">
                        <div className="skeleton-block skeleton-toolbar"></div>
                        <div className="skeleton-block skeleton-hint"></div>
                    </div>
                </div>
            )}

            {aoi && (
                <div className="action-bar">
                    <button
                        className="select-aoi-btn"
                        onClick={handleSelectAoi}
                        disabled={loading}
                    >
                        {loading ? (
                            <><span className="action-spinner" /> Analyzing...</>
                        ) : (
                            'Select AOI'
                        )}
                    </button>
                </div>
            )}

            {error && (
                <div className="error-bar">
                    <span>{error}</span>
                    <button onClick={() => setError(null)}>✕</button>
                </div>
            )}
        </div>
    );
}

export default Map;
