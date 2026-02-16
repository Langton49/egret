import { useState } from 'react';
import './MapToolPanel.css';

function MapToolPanel({ draw, drawHint }) {
  const [open, setOpen] = useState(true);

  const handlePolygon = () => {
    if (draw) draw.changeMode('draw_polygon');
  };

  const handleTrash = () => {
    if (draw) draw.trash();
  };

  return (
    <div className={`map-tool-panel ${!open ? 'collapsed' : ''}`}>
      <div className="tool-panel-header">
        {open && <h4 className="tool-panel-heading">Map Tools</h4>}
        <button
          className={`tool-panel-toggle ${open ? 'collapsed' : ''}`}
          onClick={() => setOpen(!open)}
        >
          <span className="bar bar-top" />
          <span className="bar bar-mid" />
          <span className="bar bar-bot" />
        </button>
      </div>
      {open && (
        <div className='drawing_tools'>
          {drawHint && (
            <div className="tool-panel-content">
              <div className="tool_header">Getting Started</div>
              <p className="tool-hint">
                <strong>Shift + drag</strong> on the map to draw a rectangle over your area of interest.
              </p>
              <p className="tool-hint">
                Or use the <strong>Polygon</strong> button below to draw a custom shape (click points, double-click to finish).
              </p>
              <p className="tool-hint">
                Select your AOI and use the <strong>Delete</strong> button to remove it from the map.
              </p>
            </div>
          )}
          <div className="tool-panel-content">
            <div className='tool_header'>Draw</div>
            <div className="tool-panel-buttons">
              <button onClick={handlePolygon} className="tool-btn">
                Polygon
              </button>
              <button onClick={handleTrash} className="tool-btn">
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default MapToolPanel;