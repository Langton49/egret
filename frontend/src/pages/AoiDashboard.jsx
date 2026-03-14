import { useLocation, useNavigate } from 'react-router-dom';
import { useEffect, useRef, useState } from 'react';
import {
    BarChart, Bar, XAxis, YAxis, Tooltip, Cell,
    PieChart, Pie, Legend, ResponsiveContainer,
} from 'recharts';
import MiniMap from '../components/MiniMap';
import './AoiDashboard.css';

const backendUrl = import.meta.env.VITE_BACKEND;

const ARCHETYPE_COLORS = {
    'Highly suitable':    '#1a9850',
    'Moderately suitable':'#91cf60',
    'Marginal':           '#fee08b',
    'Unsuitable':         '#d73027',
    'Open water':         '#2166ac',
};

const CONDITION_COLORS = {
    'Excellent': '#006837',
    'Good':      '#1a9850',
    'Fair':      '#d4a017',
    'Poor':      '#d73027',
};

// Semi-circle gauge
function SuitabilityGauge({ value }) {
    const r = 70;
    const stroke = 10;
    const arc = Math.PI * r;
    const filled = (value / 100) * arc;
    const color = value >= 65 ? '#1a9850' : value >= 35 ? '#d4a017' : '#d73027';
    const d = `M ${stroke} 80 A ${r} ${r} 0 0 1 ${160 - stroke} 80`;

    return (
        <div className="gauge-wrapper">
            <svg viewBox="0 0 160 90" className="gauge-svg">
                <path d={d} fill="none" stroke="rgba(0,0,0,0.08)" strokeWidth={stroke} strokeLinecap="round" />
                <path
                    d={d}
                    fill="none"
                    stroke={color}
                    strokeWidth={stroke}
                    strokeLinecap="round"
                    strokeDasharray={`${filled} ${arc}`}
                />
            </svg>
            <div className="gauge-value" style={{ color }}>{value}%</div>
            <div className="gauge-label">Mean Suitability</div>
        </div>
    );
}

function AoiDashboard() {
    const location = useLocation();
    const navigate = useNavigate();
    const state = location.state;

    const [scoring, setScoring] = useState(false);
    const [progress, setProgress] = useState(null);
    const [scoreResults, setScoreResults] = useState(null);
    const [error, setError] = useState(null);
    const pollRef = useRef(null);

    useEffect(() => {
        if (!state?.aoi || !state?.results) {
            navigate('/demo/map', { replace: true });
        }
        return () => {
            if (pollRef.current) clearInterval(pollRef.current);
        };
    }, []);

    if (!state?.aoi || !state?.results) return null;

    const { aoi, results } = state;

    const handleScore = async () => {
        setScoring(true);
        setProgress('Starting...');
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
                    } else if (data.status === 'failed') {
                        clearInterval(pollRef.current);
                        pollRef.current = null;
                        setError(data.error || 'Scoring failed.');
                        setScoring(false);
                    }
                } catch {
                    clearInterval(pollRef.current);
                    pollRef.current = null;
                    setError('Lost connection to server.');
                    setScoring(false);
                }
            }, 3000);
        } catch {
            setError('Failed to start scoring job.');
            setScoring(false);
        }
    };

    // Chart data
    const speciesData = results.top_species.map(s => ({ name: s.order, count: s.count }));

    const archetypeData = scoreResults
        ? Object.entries(scoreResults.archetype_breakdown).map(([name, info]) => ({
            name, value: info.count, pct: info.pct,
        }))
        : [];

    const conditionColor = CONDITION_COLORS[results.condition] || '#b8960c';

    return (
        <div className="aoi-dashboard">

            {/* ── Header ── */}
            <header className="aoi-dash-header">
                <button className="dash-back-btn" onClick={() => navigate('/demo/map')}>
                    ← Back to Map
                </button>
                <div className="aoi-dash-title">
                    <span className="aoi-dash-area">{results.area_km2} km²</span>
                    <span className="aoi-dash-subtitle">Area of Interest</span>
                </div>
                <button
                    className="dash-score-btn"
                    onClick={handleScore}
                    disabled={scoring || !!scoreResults}
                >
                    {scoring ? (
                        <><span className="dash-spinner" />{progress || 'Scoring...'}</>
                    ) : scoreResults ? (
                        '✓ Scored'
                    ) : (
                        'Score Habitat'
                    )}
                </button>
            </header>

            {error && (
                <div className="dash-error-bar">
                    {error}
                    <button onClick={() => setError(null)}>✕</button>
                </div>
            )}

            {/* ── Body ── */}
            <div className="aoi-dash-body">

                {/* Left: mini map */}
                <div className="aoi-dash-map-col">
                    <div className="mini-map-container">
                        <MiniMap
                            aoiGeojson={aoi}
                            scoredCells={scoreResults?.cell_geojson || null}
                        />
                    </div>

                    {scoreResults && (
                        <div className="map-legend">
                            {Object.entries(ARCHETYPE_COLORS).map(([name, color]) => (
                                <div key={name} className="legend-item">
                                    <span className="legend-dot" style={{ background: color }} />
                                    <span>{name}</span>
                                </div>
                            ))}
                        </div>
                    )}
                </div>

                {/* Right: summary data */}
                <div className="aoi-dash-data-col">

                    {/* Stat cards */}
                    <div className="stat-cards-row">
                        <div className="stat-card">
                            <div className="stat-label">Condition</div>
                            <div className="stat-value" style={{ color: conditionColor }}>
                                {results.condition}
                            </div>
                            <div className="stat-sub">{results.condition_detail}</div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-label">Sightings</div>
                            <div className="stat-value">{results.total_sightings.toLocaleString()}</div>
                            <div className="stat-sub">{results.species_count} species recorded</div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-label">Diversity</div>
                            <div className="stat-value">{results.diversity_level}</div>
                        </div>
                    </div>

                    {/* Species bar chart */}
                    {speciesData.length > 0 && (
                        <div className="chart-card">
                            <div className="chart-title">Top Species Orders</div>
                            <ResponsiveContainer width="100%" height={180}>
                                <BarChart
                                    data={speciesData}
                                    layout="vertical"
                                    margin={{ left: 8, right: 20, top: 4, bottom: 4 }}
                                >
                                    <XAxis type="number" hide />
                                    <YAxis
                                        type="category"
                                        dataKey="name"
                                        width={120}
                                        tick={{ fontSize: 11, fontFamily: 'DM Sans', fill: '#2c2c2c' }}
                                        axisLine={false}
                                        tickLine={false}
                                    />
                                    <Tooltip
                                        formatter={(v) => [v.toLocaleString(), 'Observations']}
                                        contentStyle={{
                                            fontFamily: 'DM Sans', fontSize: 12,
                                            borderRadius: 8, border: '1px solid rgba(0,0,0,0.08)',
                                            background: 'rgba(247,246,243,0.96)',
                                        }}
                                    />
                                    <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                                        {speciesData.map((_, i) => (
                                            <Cell key={i} fill={`rgba(184,150,12,${1 - i * 0.1})`} />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    )}

                    {/* Info cards */}
                    <div className="info-cards-row">
                        <div className="info-card">
                            <div className="info-label">Trends</div>
                            <p>{results.vegetation_trend}</p>
                            <p>{results.water_trend}</p>
                        </div>
                        <div className="info-card">
                            <div className="info-label">Notable Change</div>
                            <p>{results.notable_change}</p>
                        </div>
                        <div className="info-card">
                            <div className="info-label">Data Coverage</div>
                            <p>{results.data_coverage}</p>
                        </div>
                    </div>
                </div>
            </div>

            {/* ── Score section ── */}
            {(scoring || scoreResults) && (
                <div className="score-section">
                    <div className="score-section-header">
                        <span className="score-section-title">Habitat Score</span>
                        {scoreResults && (
                            <span className="score-meta-strip">
                                {scoreResults.n_scenes} scenes · {scoreResults.acquisition_date}
                            </span>
                        )}
                    </div>

                    {scoring && (
                        <div className="score-loading">
                            <span className="dash-spinner large" />
                            <span>{progress}</span>
                        </div>
                    )}

                    {scoreResults && (
                        <div className="score-results-grid">

                            {/* Gauge */}
                            <div className="score-card gauge-card">
                                <SuitabilityGauge value={scoreResults.mean_suitability} />
                                <div className="gauge-max">Max: {scoreResults.max_suitability}%</div>
                            </div>

                            {/* Donut chart */}
                            <div className="score-card">
                                <div className="chart-title">Habitat Breakdown</div>
                                <ResponsiveContainer width="100%" height={220}>
                                    <PieChart>
                                        <Pie
                                            data={archetypeData}
                                            cx="50%"
                                            cy="45%"
                                            innerRadius={52}
                                            outerRadius={80}
                                            dataKey="value"
                                            labelLine={false}
                                        >
                                            {archetypeData.map((entry, i) => (
                                                <Cell key={i} fill={ARCHETYPE_COLORS[entry.name] || '#999'} />
                                            ))}
                                        </Pie>
                                        <Legend
                                            iconType="circle"
                                            iconSize={8}
                                            formatter={(value) => (
                                                <span style={{ fontSize: 11, fontFamily: 'DM Sans', color: '#2c2c2c' }}>
                                                    {value}
                                                </span>
                                            )}
                                        />
                                        <Tooltip
                                            formatter={(v, name, { payload }) => [
                                                `${payload.pct}% · ${v} cells`, name
                                            ]}
                                            contentStyle={{
                                                fontFamily: 'DM Sans', fontSize: 12,
                                                borderRadius: 8, border: '1px solid rgba(0,0,0,0.08)',
                                                background: 'rgba(247,246,243,0.96)',
                                            }}
                                        />
                                    </PieChart>
                                </ResponsiveContainer>
                            </div>

                            {/* Summary text */}
                            <div className="score-card summary-card">
                                <div className="chart-title">Summary</div>
                                <p className="score-summary-text">{scoreResults.summary}</p>
                            </div>

                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

export default AoiDashboard;
