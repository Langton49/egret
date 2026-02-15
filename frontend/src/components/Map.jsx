import { useCallback, useEffect, useRef, useState } from 'react';
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

function useDraggable() {
    const elRef = useRef(null);
    const isDragging = useRef(false);
    const offset = useRef({ x: 0, y: 0 });
    const cleanupRef = useRef(null);

    const callbackRef = useCallback((node) => {
        // Clean up previous listeners
        if (cleanupRef.current) {
            cleanupRef.current();
            cleanupRef.current = null;
        }

        elRef.current = node;
        if (!node) return;

        const onMouseDown = (e) => {
            if (e.target.closest('button, a, input, select, textarea')) return;
            const header = node.querySelector('.results-header');
            if (header && !header.contains(e.target)) return;

            isDragging.current = true;
            const rect = node.getBoundingClientRect();
            // Lock in current position as left/top before dragging
            node.style.left = `${rect.left}px`;
            node.style.top = `${rect.top}px`;
            node.style.right = 'auto';
            node.style.bottom = 'auto';
            offset.current = { x: e.clientX - rect.left, y: e.clientY - rect.top };
            node.style.cursor = 'grabbing';
            node.style.userSelect = 'none';
            e.preventDefault();
        };

        const onMouseMove = (e) => {
            if (!isDragging.current) return;
            const x = e.clientX - offset.current.x;
            const y = e.clientY - offset.current.y;
            node.style.left = `${x}px`;
            node.style.top = `${y}px`;
            node.style.right = 'auto';
            node.style.bottom = 'auto';
        };

        const onMouseUp = () => {
            if (!isDragging.current) return;
            isDragging.current = false;
            node.style.cursor = '';
            node.style.userSelect = '';
        };

        node.addEventListener('mousedown', onMouseDown);
        window.addEventListener('mousemove', onMouseMove);
        window.addEventListener('mouseup', onMouseUp);

        cleanupRef.current = () => {
            node.removeEventListener('mousedown', onMouseDown);
            window.removeEventListener('mousemove', onMouseMove);
            window.removeEventListener('mouseup', onMouseUp);
        };
    }, []);

    return callbackRef;
}

function Map() {
    const mapRef = useRef(null);
    const mapContainerRef = useRef(null);
    const drawRef = useRef(null);
    const pollRef = useRef(null);
    const [aoi, setAoi] = useState(null);
    const [loading, setLoading] = useState(false);
    const [results, setResults] = useState(null);
    const [scoring, setScoring] = useState(false);
    const [progress, setProgress] = useState(null);
    const [scoreResults, setScoreResults] = useState(null);
    const [error, setError] = useState(null);
    const [drawHint, setDrawHint] = useState(true);

    const resultsPanelRef = useDraggable();
    const scorePanelRef = useDraggable();

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
            mapRef.current.fitBounds(
                [[SW_LNG, SW_LAT], [NE_LNG, NE_LAT]],
                { padding: 40, duration: 0 }
            );

            // Show study area boundary (guard against double-mount)
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
            controls: {
                polygon: true,
                trash: true,
            },
            defaultMode: 'simple_select',
        });

        drawRef.current = draw;
        mapRef.current.addControl(draw, 'top-left');

        // Shift+drag rectangle drawing
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

            // Ignore tiny accidental drags
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

            // Clear previous drawings and add the rectangle
            draw.deleteAll();
            const ids = draw.add(rectFeature);
            setDrawHint(false);
            setError(null);
            setAoi(draw.getAll());
        };

        canvas.addEventListener('mousedown', onMouseDown);
        canvas.addEventListener('mousemove', onMouseMove);
        canvas.addEventListener('mouseup', onMouseUp);

        const updateAOI = () => {
            const data = draw.getAll();
            if (data.features.length > 0) {
                setDrawHint(false);
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
                setResults(null);
                setScoreResults(null);
                setError(null);
            }
        };

        mapRef.current.on('draw.create', updateAOI);
        mapRef.current.on('draw.update', updateAOI);
        mapRef.current.on('draw.delete', updateAOI);

        return () => {
            if (pollRef.current) clearInterval(pollRef.current);
            canvas.removeEventListener('mousedown', onMouseDown);
            canvas.removeEventListener('mousemove', onMouseMove);
            canvas.removeEventListener('mouseup', onMouseUp);
            if (mapRef.current) mapRef.current.remove();
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
        setError(null);
        try {
            const response = await fetch(`${backendUrl}/aoidata/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ aoi }),
            });
            const data = await response.json();
            setResults(data);
        } catch (err) {
            console.error('Analysis failed:', err);
            setError('Quick analysis failed. Try again.');
        } finally {
            setLoading(false);
        }
    };

    const handleScore = async () => {
        if (!aoi) return;
        setScoring(true);
        setProgress('Starting...');
        setScoreResults(null);
        setError(null);

        try {
            const res = await fetch(`${backendUrl}/habitat/score`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ aoi }),
            });
            const { job_id } = await res.json();

            pollRef.current = setInterval(async () => {
                try {
                    const status = await fetch(`${backendUrl}/habitat/score/${job_id}`);
                    const data = await status.json();
                    setProgress(data.progress);

                    if (data.status === 'complete') {
                        clearInterval(pollRef.current);
                        pollRef.current = null;
                        setScoreResults(data.result);
                        setScoring(false);

                        if (data.result.cell_geojson && mapRef.current) {
                            if (mapRef.current.getSource('scored-cells')) {
                                mapRef.current.getSource('scored-cells').setData(data.result.cell_geojson);
                            } else {
                                mapRef.current.addSource('scored-cells', {
                                    type: 'geojson',
                                    data: data.result.cell_geojson,
                                });
                                mapRef.current.addLayer({
                                    id: 'scored-cells-layer',
                                    type: 'circle',
                                    source: 'scored-cells',
                                    paint: {
                                        'circle-radius': 6,
                                        'circle-color': [
                                            'interpolate', ['linear'],
                                            ['get', 'suitability'],
                                            0, '#d73027',
                                            40, '#fee08b',
                                            70, '#66bd63',
                                            100, '#1a9850',
                                        ],
                                        'circle-opacity': 0.7,
                                        'circle-stroke-width': 1,
                                        'circle-stroke-color': '#fff',
                                        'circle-stroke-opacity': 0.4,
                                    },
                                });
                            }
                        }
                    } else if (data.status === 'failed') {
                        clearInterval(pollRef.current);
                        pollRef.current = null;
                        setError(data.error);
                        setScoring(false);
                    }
                } catch {
                    clearInterval(pollRef.current);
                    pollRef.current = null;
                    setError('Lost connection to server.');
                    setScoring(false);
                }
            }, 3000);
        } catch (err) {
            console.error('Scoring failed:', err);
            setError('Failed to start scoring job.');
            setScoring(false);
        }
    };

    return (
        <div id="map-container" ref={mapContainerRef}>
            {drawHint && (
                <div className="draw-hint">
                    <div className="draw-hint-content">
                        <span className="draw-hint-icon">▧</span>
                        <div>
                            <strong>Shift + drag</strong> to draw a rectangle over your area of interest.
                            <br />
                            <span className="draw-hint-sub">Or use the polygon tool on the left to draw a custom shape (click points, double-click to finish).</span>
                        </div>
                        <button className="draw-hint-close" onClick={() => setDrawHint(false)}>✕</button>
                    </div>
                </div>
            )}
            {aoi && (
                <div className="action-bar">
                    <button
                        className="analyze-btn"
                        onClick={handleAnalyze}
                        disabled={loading || scoring}
                    >
                        {loading ? 'Analyzing...' : 'Quick Analysis'}
                    </button>
                    <button
                        className="score-btn"
                        onClick={handleScore}
                        disabled={scoring || loading}
                    >
                        {scoring ? 'Scoring...' : 'Score Habitat'}
                    </button>
                </div>
            )}

            {scoring && (
                <div className="progress-bar">
                    <div className="progress-spinner"></div>
                    <span>{progress}</span>
                </div>
            )}

            {error && (
                <div className="error-bar">
                    <span>{error}</span>
                    <button onClick={() => setError(null)}>✕</button>
                </div>
            )}

            {results && (
                <div className="results-panel" ref={resultsPanelRef}>
                    <div className="results-header" style={{ cursor: 'grab' }}>
                        <h3>Area Overview</h3>
                        <button onClick={() => setResults(null)}>✕</button>
                    </div>

                    {results.area_km2 && <p>{results.area_km2} km²</p>}

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

            {scoreResults && (
    <div className="results-panel score-panel" ref={scorePanelRef}>
        <div className="results-header" style={{ cursor: 'grab' }}>
            <h3>Habitat Score</h3>
            <button onClick={() => {
                setScoreResults(null);
                if (mapRef.current?.getLayer('scored-cells-layer')) {
                    mapRef.current.removeLayer('scored-cells-layer');
                    mapRef.current.removeSource('scored-cells');
                }
            }}>✕</button>
        </div>

        <p className="score-summary">{scoreResults.summary}</p>
        <p>{scoreResults.n_scenes} scenes • {scoreResults.acquisition_date}</p>

        <div className="result-card">
            <h4>Suitability</h4>
            <p>Mean: {scoreResults.mean_suitability}%</p>
            <p>Max: {scoreResults.max_suitability}%</p>
        </div>

        <div className="result-card">
            <h4>Habitat Breakdown</h4>
            {Object.entries(scoreResults.archetype_breakdown).map(([name, info]) => (
                <p key={name}>{name}: {info.count} cells ({info.pct}%)</p>
            ))}
        </div>

        <div className="result-card">
            <h4>Top Cells</h4>
            {scoreResults.top_cells.map((c, i) => (
                <p key={i}>
                    ({c.lon.toFixed(3)}, {c.lat.toFixed(3)}) — {(c.probability * 100).toFixed(1)}% • {c.archetype}
                </p>
            ))}
        </div>
    </div>
)}
        </div>
    );
}

export default Map;