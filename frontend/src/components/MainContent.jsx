import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';
import './MainContent.css';
import Map from './Map';

mapboxgl.accessToken = import.meta.env.VITE_MAPBOX;

function MainContent() {
    return (
         <div className='main_content'>
            <div className='map'>
                <Map />
            </div>
        </div>
    );
}

export default MainContent;