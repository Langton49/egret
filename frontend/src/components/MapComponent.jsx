import { useState } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import './MapComponent.css';
import Map from './Map';
import MapToolPanel from './MapToolPanel';

mapboxgl.accessToken = import.meta.env.VITE_MAPBOX;

function MapComponent() {
  const [draw, setDraw] = useState(null);
  const [drawHint, setDrawHint] = useState(true);

  return (
    <div className="map-wrapper">
      <div className="map">
        <Map onDrawReady={setDraw} drawHint={drawHint} setDrawHint={setDrawHint} />
      </div>
      <MapToolPanel draw={draw} drawHint={drawHint} />
    </div>
  );
}

export default MapComponent;