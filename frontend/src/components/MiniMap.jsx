import { useEffect, useRef } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';

mapboxgl.accessToken = import.meta.env.VITE_MAPBOX;

const ARCHETYPE_FILL = [
    'match', ['get', 'archetype'],
    'Highly suitable',    '#1a9850',
    'Moderately suitable','#91cf60',
    'Marginal',           '#fee08b',
    'Unsuitable',         '#d73027',
    'Open water',         '#2166ac',
    '#999999',
];

function MiniMap({ aoiGeojson, scoredCells }) {
    const containerRef = useRef(null);
    const mapRef = useRef(null);

    useEffect(() => {
        if (!containerRef.current || !aoiGeojson) return;

        const coords = aoiGeojson.features[0].geometry.coordinates[0];
        const lngs = coords.map(c => c[0]);
        const lats = coords.map(c => c[1]);
        const bounds = [
            [Math.min(...lngs), Math.min(...lats)],
            [Math.max(...lngs), Math.max(...lats)],
        ];

        const map = new mapboxgl.Map({
            container: containerRef.current,
            style: 'mapbox://styles/mapbox/standard',
            interactive: false,
            attributionControl: false,
        });
        mapRef.current = map;

        map.on('load', () => {
            map.fitBounds(bounds, { padding: 32, duration: 0 });

            map.addSource('mini-aoi', { type: 'geojson', data: aoiGeojson });
            map.addLayer({
                id: 'mini-aoi-fill',
                type: 'fill',
                source: 'mini-aoi',
                paint: { 'fill-color': '#b8960c', 'fill-opacity': 0.12 },
            });
            map.addLayer({
                id: 'mini-aoi-line',
                type: 'line',
                source: 'mini-aoi',
                paint: { 'line-color': '#b8960c', 'line-width': 2, 'line-dasharray': [3, 2] },
            });
        });

        return () => {
            map.remove();
            mapRef.current = null;
        };
    }, []);

    // Update scored cells layer when data arrives
    useEffect(() => {
        const map = mapRef.current;
        if (!map || !scoredCells) return;

        const addCells = () => {
            if (map.getSource('mini-scored')) {
                map.getSource('mini-scored').setData(scoredCells);
            } else {
                map.addSource('mini-scored', { type: 'geojson', data: scoredCells });
                map.addLayer({
                    id: 'mini-scored-layer',
                    type: 'fill',
                    source: 'mini-scored',
                    paint: {
                        'fill-color': ARCHETYPE_FILL,
                        'fill-opacity': 0.72,
                        'fill-outline-color': 'rgba(255,255,255,0.2)',
                    },
                });
            }
        };

        if (map.loaded()) {
            addCells();
        } else {
            map.once('load', addCells);
        }
    }, [scoredCells]);

    return (
        <div
            ref={containerRef}
            style={{ width: '100%', height: '100%', borderRadius: '12px', overflow: 'hidden' }}
        />
    );
}

export default MiniMap;
