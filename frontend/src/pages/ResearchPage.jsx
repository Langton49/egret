import { useState, useEffect, useRef, useCallback, useMemo } from 'react'
import ReactMarkdown from 'react-markdown'
import mapboxgl from 'mapbox-gl'
import Map, { Source, Layer, useMap, NavigationControl } from 'react-map-gl/mapbox'
import MapboxDraw from '@mapbox/mapbox-gl-draw'
import '@mapbox/mapbox-gl-draw/dist/mapbox-gl-draw.css'
import 'mapbox-gl/dist/mapbox-gl.css'
import './ResearchPage.css'

mapboxgl.accessToken = import.meta.env.VITE_MAPBOX
const API = `${import.meta.env.VITE_BACKEND || 'http://localhost:8000'}/api/research`

const INDICES = ['NDVI', 'MNDWI', 'NDWI', 'NDMI', 'water_mask']
const PANEL_TABS = ['timeline', 'data']
const SNAP_HEIGHTS = [220, 340, 500]
const MAX_HEIGHT_RATIO = 0.72

// ─── Map layer styles ────────────────────────────────────────────────────────

const clusterLayer = {
  id: 'clusters',
  type: 'circle',
  filter: ['has', 'point_count'],
  paint: {
    'circle-color': ['step', ['get', 'point_count'], '#4a8a6a', 20, '#3a7a5a', 50, '#2a6a4a'],
    'circle-radius': ['step', ['get', 'point_count'], 18, 20, 24, 50, 30],
    'circle-opacity': 0.85,
    'circle-stroke-width': 2,
    'circle-stroke-color': '#a8d8a8',
  },
}

const clusterCountLayer = {
  id: 'cluster-count',
  type: 'symbol',
  filter: ['has', 'point_count'],
  layout: {
    'text-field': '{point_count_abbreviated}',
    'text-font': ['DIN Offc Pro Medium', 'Arial Unicode MS Bold'],
    'text-size': 13,
  },
  paint: { 'text-color': '#ffffff' },
}

const markerLayer = {
  id: 'colony-markers',
  type: 'circle',
  filter: ['!', ['has', 'point_count']],
  paint: {
    'circle-radius': ['case', ['get', 'has_surveys'], 7, 5],
    'circle-color': ['case', ['get', 'has_surveys'], '#a8d8a8', '#5a8a6a'],
    'circle-opacity': 0.9,
    'circle-stroke-width': 1.5,
    'circle-stroke-color': '#ffffff',
  },
}

const spectralOverlayLayer = {
  id: 'colony-spectral-overlay',
  type: 'circle',
  paint: {
    'circle-radius': 7,
    'circle-color': ['case',
      ['==', ['get', 'value'], null], 'rgba(128,128,128,0.3)',
      ['interpolate', ['linear'], ['get', 'value'],
        0.0, '#ff2200',
        0.3, '#ffee00',
        0.8, '#00cc44',
      ],
    ],
    'circle-opacity': 0.75,
    'circle-blur': 0.3,
  },
}

// ─── Draw styles ─────────────────────────────────────────────────────────────

const DRAW_STYLES = [
  { id: 'gl-draw-polygon-fill', type: 'fill', filter: ['all', ['==', '$type', 'Polygon'], ['!=', 'mode', 'static']],
    paint: { 'fill-color': '#b8960c', 'fill-opacity': 0.12 } },
  { id: 'gl-draw-polygon-stroke', type: 'line', filter: ['all', ['==', '$type', 'Polygon'], ['!=', 'mode', 'static']],
    layout: { 'line-cap': 'round', 'line-join': 'round' },
    paint: { 'line-color': '#b8960c', 'line-width': 2 } },
  { id: 'gl-draw-polygon-midpoint', type: 'circle', filter: ['all', ['==', '$type', 'Point'], ['==', 'meta', 'midpoint']],
    paint: { 'circle-radius': 4, 'circle-color': '#b8960c' } },
  { id: 'gl-draw-vertex', type: 'circle', filter: ['all', ['==', '$type', 'Point'], ['==', 'meta', 'vertex']],
    paint: { 'circle-radius': 5, 'circle-color': '#fff', 'circle-stroke-color': '#b8960c', 'circle-stroke-width': 2 } },
  { id: 'gl-draw-polygon-fill-static', type: 'fill', filter: ['all', ['==', '$type', 'Polygon'], ['==', 'mode', 'static']],
    paint: { 'fill-color': '#b8960c', 'fill-opacity': 0.08 } },
  { id: 'gl-draw-polygon-stroke-static', type: 'line', filter: ['all', ['==', '$type', 'Polygon'], ['==', 'mode', 'static']],
    paint: { 'line-color': '#b8960c', 'line-width': 2 } },
]

// ─── Sub-components ───────────────────────────────────────────────────────────

function DrawControl({ onDrawCreate }) {
  const { current: mapRef } = useMap()
  const drawRef = useRef(null)
  const drawElRef = useRef(null)

  useEffect(() => {
    const map = mapRef?.getMap()
    if (!map || drawRef.current) return

    const draw = new MapboxDraw({
      displayControlsDefault: false,
      controls: { polygon: true, trash: true },
      styles: DRAW_STYLES,
      touchEnabled: false,
    })

    // Use draw.onAdd() directly instead of map.addControl().
    // map.addControl() puts the instance into map._controls, which means
    // Map.remove() will call draw.onRemove() during its own teardown — after
    // the terrain has already been cleaned up — causing an uncatchable crash.
    // By bypassing addControl we own the lifecycle and handle cleanup safely.
    const el = draw.onAdd(map)
    drawElRef.current = el
    const slot = map.getContainer().querySelector('.mapboxgl-ctrl-top-right')
    if (slot) slot.appendChild(el)

    drawRef.current = draw

    const handleCreate = (e) => onDrawCreate?.(e.features[0])
    map.on('draw.create', handleCreate)

    return () => {
      map.off('draw.create', handleCreate)
      try { draw.onRemove(map) } catch (_) {}
      drawElRef.current?.parentNode?.removeChild(drawElRef.current)
      drawElRef.current = null
      drawRef.current = null
    }
  }, [mapRef, onDrawCreate])

  return null
}

// ─── Left panel detail cards ──────────────────────────────────────────────────

function SurveyCard({ data, year }) {
  const d = data?.[String(year)]
  if (!d) return <p className="empty-hint">No survey data for {year}.</p>
  return (
    <div className="survey-card">
      <h4>Survey {year}</h4>
      <div className="survey-grid">
        <span>Total birds</span><span>{d.total_birds ?? '–'}</span>
        <span>Active nests</span><span>{d.active_nests ?? '–'}</span>
        <span>Species</span><span>{d.n_species ?? '–'}</span>
        <span>Diversity</span><span>{d.shannon_diversity != null ? d.shannon_diversity.toFixed(2) : '–'}</span>
        <span>Dominant sp.</span><span>{d.dominant_species ?? '–'}</span>
        <span>Nest success</span><span>{d.nest_success_ratio != null ? (d.nest_success_ratio * 100).toFixed(0) + '%' : '–'}</span>
        <span>Trajectory</span><span className={`trajectory ${d.colony_trajectory}`}>{d.colony_trajectory ?? '–'}</span>
        <span>Nest Δ</span><span>{d.active_nests_pct_change != null ? (d.active_nests_pct_change * 100).toFixed(1) + '%' : '–'}</span>
      </div>
    </div>
  )
}

function GapFillCard({ yearData, index }) {
  if (!yearData) return null
  const sensorLabel = yearData.sensor ? yearData.sensor.toUpperCase() : null
  return (
    <div className="survey-card gap-fill-card">
      <h4>{yearData.year} — Satellite only{sensorLabel ? ` (${sensorLabel})` : ''}</h4>
      <p className="index-value">
        {index ?? 'NDVI'}: {yearData.value != null ? yearData.value.toFixed(3) : 'No data'}
      </p>
      {yearData.gap_fill_text && (
        <p className={`gap-fill-text signal-${yearData.gap_fill_signal ?? 'stable'}`}>
          {yearData.gap_fill_text}
        </p>
      )}
      <p className="gap-fill-note">Satellite-derived — qualitative estimate, not field verified.</p>
    </div>
  )
}

// ─── Year insights strip ──────────────────────────────────────────────────────

function YearInsights({ year, yearData, surveyData }) {
  const d = surveyData?.[String(year)]
  const isSurvey = yearData?.has_survey
  return (
    <div className="year-insights">
      <span className="insight-year">{year ?? '–'}</span>
      {isSurvey && d ? (
        <div className="insight-stats">
          {d.total_birds        != null && <span className="insight-stat"><em>Birds</em>{d.total_birds.toLocaleString()}</span>}
          {d.active_nests       != null && <span className="insight-stat"><em>Nests</em>{d.active_nests}</span>}
          {d.n_species          != null && <span className="insight-stat"><em>Species</em>{d.n_species}</span>}
          {d.dominant_species            && <span className="insight-stat"><em>Dominant</em>{d.dominant_species}</span>}
          {d.colony_trajectory           && <span className={`insight-stat traj-${d.colony_trajectory}`}><em>Trend</em>{d.colony_trajectory}</span>}
          {d.active_nests_pct_change != null && <span className="insight-stat"><em>Nest Δ</em>{(d.active_nests_pct_change * 100).toFixed(1)}%</span>}
          {d.shannon_diversity  != null && <span className="insight-stat"><em>Diversity</em>{d.shannon_diversity.toFixed(2)}</span>}
        </div>
      ) : (
        <div className="insight-placeholder">
          {year ? 'Satellite inference — data loading' : 'Select a year on the seeker'}
        </div>
      )}
    </div>
  )
}

// ─── Video seeker ─────────────────────────────────────────────────────────────

function VideoSeeker({ years, selectedYear, onSelectYear }) {
  const trackRef = useRef(null)
  const yearList = useMemo(() => years?.map(y => y.year) ?? [], [years])
  const currentIdx = yearList.indexOf(selectedYear)
  const progress = yearList.length > 1 ? currentIdx / (yearList.length - 1) : 0

  const seekToX = useCallback((clientX) => {
    const rect = trackRef.current?.getBoundingClientRect()
    if (!rect) return
    const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width))
    const idx = Math.round(ratio * (yearList.length - 1))
    if (yearList[idx] !== undefined) onSelectYear(yearList[idx])
  }, [yearList, onSelectYear])

  const handleMouseDown = useCallback((e) => {
    e.preventDefault()
    seekToX(e.clientX)
    const onMove = (ev) => seekToX(ev.clientX)
    const onUp = () => {
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }, [seekToX])

  if (!years?.length) return null

  return (
    <div className="video-seeker">
      <div className="seeker-markers">
        {years.map((y, i) => {
          const pct = yearList.length > 1 ? (i / (yearList.length - 1)) * 100 : 0
          let markVariant
          if (y.has_survey) markVariant = 'survey'
          else if (y.value == null) markVariant = 'no-data'
          else markVariant = y.gap_fill_signal ?? 'stable'
          return (
            <div
              key={y.year}
              className={`seeker-mark ${markVariant}`}
              style={{ left: `${pct}%` }}
              title={String(y.year)}
            />
          )
        })}
      </div>
      <div className="seeker-track" ref={trackRef} onMouseDown={handleMouseDown}>
        <div className="seeker-fill" style={{ width: `${progress * 100}%` }} />
        <div className="seeker-thumb" style={{ left: `${progress * 100}%` }}>
          <span className="seeker-year-label">{selectedYear}</span>
        </div>
      </div>
      <div className="seeker-endpoints">
        <span>{yearList[0]}</span>
        <span>{yearList[yearList.length - 1]}</span>
      </div>
    </div>
  )
}

// ─── Chatbot sidebar ──────────────────────────────────────────────────────────

function ChatSidebar({ selectedSlug, onCollapse }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const bottomRef = useRef(null)
  const inputRef = useRef(null)

  // Scroll to bottom on new content
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const sendMessage = useCallback(async () => {
    const text = input.trim()
    if (!text || streaming) return

    const userMsg = { role: 'user', content: text }
    const nextMessages = [...messages, userMsg]
    setMessages(nextMessages)
    setInput('')
    setStreaming(true)

    // Placeholder for assistant response
    setMessages(prev => [...prev, { role: 'assistant', content: '' }])

    try {
      const res = await fetch(`${API}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: nextMessages,
          colony_slug: selectedSlug || null,
        }),
      })

      if (!res.ok) throw new Error(`HTTP ${res.status}`)

      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })

        const lines = buffer.split('\n')
        buffer = lines.pop() // keep incomplete line

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue
          const payload = line.slice(6)
          if (payload === '[DONE]') break
          try {
            const chunk = JSON.parse(payload)
            setMessages(prev => {
              const updated = [...prev]
              updated[updated.length - 1] = {
                role: 'assistant',
                content: updated[updated.length - 1].content + chunk,
              }
              return updated
            })
          } catch { /* ignore parse errors */ }
        }
      }
    } catch (e) {
      setMessages(prev => {
        const updated = [...prev]
        updated[updated.length - 1] = {
          role: 'assistant',
          content: 'Something went wrong. Please try again.',
        }
        return updated
      })
    } finally {
      setStreaming(false)
      inputRef.current?.focus()
    }
  }, [input, messages, streaming, selectedSlug])

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="chat-sidebar">
      <div className="chat-header">
        <span className="chat-title">Research Assistant</span>
        {selectedSlug && <span className="chat-context-badge">Colony loaded</span>}
        <button className="chat-collapse" onClick={onCollapse} title="Collapse">‹</button>
      </div>

      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-empty">
            <p>Ask about survey trends, species data, habitat change, or grant writing.</p>
            {!selectedSlug && (
              <p className="chat-empty-hint">Select a colony on the map for colony-specific answers.</p>
            )}
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`chat-bubble ${m.role}`}>
            {m.role === 'assistant' ? (
              <div className="bubble-text">
                {m.content
                  ? <ReactMarkdown>{m.content}</ReactMarkdown>
                  : (streaming && i === messages.length - 1 ? <span className="typing-indicator"><span/><span/><span/></span> : '')}
              </div>
            ) : (
              <span className="bubble-text">{m.content}</span>
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      <div className="chat-input-row">
        <textarea
          ref={inputRef}
          className="chat-input"
          placeholder={selectedSlug ? 'Ask about this colony…' : 'Ask about any colony…'}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={1}
          disabled={streaming}
        />
        <button
          className="chat-send"
          onClick={sendMessage}
          disabled={streaming || !input.trim()}
        >
          {streaming ? '…' : '↑'}
        </button>
      </div>
    </div>
  )
}

// ─── Colony name search bar (fuzzy, no AI) ────────────────────────────────────

function ColonySearchBar({ onSelect }) {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState([])
  const [open, setOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const barRef = useRef(null)
  const debounceRef = useRef(null)

  // Click-outside closes dropdown
  useEffect(() => {
    const handler = (e) => {
      if (barRef.current && !barRef.current.contains(e.target)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const search = useCallback((q) => {
    const trimmed = q.trim()
    if (trimmed.length < 2) { setResults([]); setOpen(false); return }

    setLoading(true)
    fetch(`${API}/colonies/search?q=${encodeURIComponent(trimmed)}`)
      .then(r => r.json())
      .then(data => { setResults(data); setOpen(true) })
      .catch(console.error)
      .finally(() => setLoading(false))
  }, [])

  const handleChange = (e) => {
    const q = e.target.value
    setQuery(q)
    clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => search(q), 180)
  }

  const handleSelect = (colony) => {
    onSelect(colony)
    setQuery(colony.name)
    setOpen(false)
  }

  return (
    <div className="colony-search-bar" ref={barRef}>
      <div className="colony-search-row">
        <input
          className="colony-search-input"
          placeholder="Search colonies…"
          value={query}
          onChange={handleChange}
          onFocus={() => results.length > 0 && setOpen(true)}
        />
        {loading && <span className="colony-search-spinner">…</span>}
      </div>

      {open && results.length > 0 && (
        <div className="colony-search-results">
          {results.slice(0, 12).map((c) => (
            <button key={c.slug} className="colony-search-result" onClick={() => handleSelect(c)}>
              <span className="colony-result-name">{c.name}</span>
              {c.survey_years?.length > 0 && (
                <span className="colony-result-surveys">{c.survey_years.length} surveys</span>
              )}
            </button>
          ))}
          {results.length > 12 && (
            <p className="colony-search-more">+{results.length - 12} more — keep typing</p>
          )}
        </div>
      )}
    </div>
  )
}

// ─── Main component ───────────────────────────────────────────────────────────

export default function ResearchPage() {
  const mapRef = useRef(null)

  // Colony list
  const [colonies, setColonies] = useState([])
  const [selectedSlug, setSelectedSlug] = useState(null)
  const [profile, setProfile] = useState(null)

  // Sidebar
  const [sidebarOpen, setSidebarOpen] = useState(true)

  // Bottom panel
  const [activeTab, setActiveTab] = useState('timeline')
  const [panelHeight, setPanelHeight] = useState(220)
  const [dragging, setDragging] = useState(false)
  const dragRef = useRef(null)

  // Timeline / animation
  const [timelineData, setTimelineData] = useState(null)
  const [selectedYear, setSelectedYear] = useState(null)
  const [selectedIndex, setSelectedIndex] = useState('NDVI')
  const [playing, setPlaying] = useState(false)
  const [playSpeed, setPlaySpeed] = useState(1)
  const playRef = useRef(null)

  // Map frames
  const [mapFrameGeoJSON, setMapFrameGeoJSON] = useState(null)

  // Map view
  const [viewState, setViewState] = useState({ longitude: -89.5, latitude: 28.5, zoom: 6 })
  const [mapStyle, setMapStyle] = useState('mapbox://styles/mapbox/satellite-streets-v12')
  const [mapStyleLoaded, setMapStyleLoaded] = useState(false)
  const isSatellite = mapStyle.includes('satellite')

  // Loading
  const [profileLoading, setProfileLoading] = useState(false)
  const [timelineLoading, setTimelineLoading] = useState(false)

  // ── Resize map after sidebar animation ───────────────────────────────────
  useEffect(() => {
    const timer = setTimeout(() => {
      mapRef.current?.getMap()?.resize()
    }, 300)
    return () => clearTimeout(timer)
  }, [sidebarOpen])

  // ── Load colony list ──────────────────────────────────────────────────────
  useEffect(() => {
    fetch(`${API}/colonies`)
      .then(r => r.json())
      .then(setColonies)
      .catch(console.error)
  }, [])

  // ── Colonies as GeoJSON for clustering ───────────────────────────────────
  const coloniesGeoJSON = useMemo(() => ({
    type: 'FeatureCollection',
    features: colonies
      .filter(c => c.lat != null && c.lon != null)
      .map(c => ({
        type: 'Feature',
        geometry: { type: 'Point', coordinates: [c.lon, c.lat] },
        properties: {
          slug: c.slug,
          name: c.name,
          state: c.state,
          has_surveys: (c.survey_years?.length ?? 0) > 0,
        },
      })),
  }), [colonies])

  // ── Colony selection ─────────────────────────────────────────────────────
  const handleColonySelect = useCallback(async (slug) => {
    if (slug === selectedSlug) return
    setSelectedSlug(slug)
    setProfile(null)
    setTimelineData(null)
    setSelectedYear(null)
    setMapFrameGeoJSON(null)
    setPlaying(false)
    setActiveTab('timeline')

    const colony = colonies.find(c => c.slug === slug)
    if (colony?.lat && colony?.lon) {
      setViewState(v => ({ ...v, longitude: colony.lon, latitude: colony.lat, zoom: 13 }))
    }

    setProfileLoading(true)
    try {
      const res = await fetch(`${API}/colony/${slug}/profile`)
      if (res.ok) setProfile(await res.json())
    } catch (e) { console.error(e) }
    finally { setProfileLoading(false) }

    setTimelineLoading(true)
    try {
      const res = await fetch(`${API}/colony/${slug}/timeline?index=${selectedIndex}`)
      if (res.ok) {
        const data = await res.json()
        setTimelineData(data)
        const firstYear = data.years?.find(y => y.value !== null)?.year ?? data.years?.[0]?.year
        if (firstYear) setSelectedYear(firstYear)
      }
    } catch (e) { console.error(e) }
    finally { setTimelineLoading(false) }
  }, [colonies, selectedIndex, selectedSlug])

  // Search bar colony select (comes in as a colony object with slug)
  const handleSearchBarSelect = useCallback((colony) => {
    handleColonySelect(colony.slug)
  }, [handleColonySelect])

  const handleDeselect = () => {
    setSelectedSlug(null)
    setProfile(null)
    setTimelineData(null)
    setSelectedYear(null)
    setMapFrameGeoJSON(null)
    setPlaying(false)
    setViewState(v => ({ ...v, zoom: 6, longitude: -89.5, latitude: 28.5 }))
  }

  // ── Map click handler ────────────────────────────────────────────────────
  const handleMapClick = useCallback((e) => {
    const map = mapRef.current?.getMap()
    if (!map) return

    if (map.getLayer('clusters')) {
      const clusterFeatures = map.queryRenderedFeatures(e.point, { layers: ['clusters'] })
      if (clusterFeatures.length > 0) {
        const clusterId = clusterFeatures[0].properties.cluster_id
        const source = map.getSource('colonies')
        source.getClusterExpansionZoom(clusterId, (err, zoom) => {
          if (err) return
          setViewState(v => ({
            ...v,
            longitude: clusterFeatures[0].geometry.coordinates[0],
            latitude: clusterFeatures[0].geometry.coordinates[1],
            zoom: zoom + 0.5,
          }))
        })
        return
      }
    }

    if (map.getLayer('colony-markers')) {
      const markerFeatures = map.queryRenderedFeatures(e.point, { layers: ['colony-markers'] })
      if (markerFeatures.length > 0) {
        handleColonySelect(markerFeatures[0].properties.slug)
      }
    }
  }, [handleColonySelect])

  // ── Map frame on year/index change ───────────────────────────────────────
  useEffect(() => {
    if (!selectedSlug || !selectedYear) return
    fetch(`${API}/colony/${selectedSlug}/map_frames?index=${selectedIndex}&year=${selectedYear}`)
      .then(r => r.ok ? r.json() : null)
      .then(setMapFrameGeoJSON)
      .catch(console.error)
  }, [selectedSlug, selectedYear, selectedIndex])

  // ── Reload timeline on index change ─────────────────────────────────────
  useEffect(() => {
    if (!selectedSlug) return
    setTimelineLoading(true)
    fetch(`${API}/colony/${selectedSlug}/timeline?index=${selectedIndex}`)
      .then(r => r.ok ? r.json() : null)
      .then(data => { if (data) setTimelineData(data) })
      .catch(console.error)
      .finally(() => setTimelineLoading(false))
  }, [selectedIndex, selectedSlug])

  // ── Animation ────────────────────────────────────────────────────────────
  useEffect(() => {
    if (playing && timelineData?.years) {
      const years = timelineData.years.map(y => y.year)
      playRef.current = setInterval(() => {
        setSelectedYear(prev => {
          const idx = years.indexOf(prev)
          return years[(idx + 1) % years.length]
        })
      }, 1000 / playSpeed)
    }
    return () => clearInterval(playRef.current)
  }, [playing, timelineData, playSpeed])

  // ── Panel drag-to-resize ──────────────────────────────────────────────────
  const handleDragStart = useCallback((e) => {
    e.preventDefault()
    dragRef.current = { startY: e.clientY, startHeight: panelHeight }
    setDragging(true)

    const onMove = (ev) => {
      const delta = dragRef.current.startY - ev.clientY
      const maxH = Math.floor(window.innerHeight * MAX_HEIGHT_RATIO)
      const newH = Math.max(120, Math.min(maxH, dragRef.current.startHeight + delta))
      setPanelHeight(newH)
    }
    const onUp = () => {
      setDragging(false)
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }, [panelHeight])

  const handleDragDoubleClick = useCallback(() => {
    const maxH = Math.floor(window.innerHeight * MAX_HEIGHT_RATIO)
    const heights = [...SNAP_HEIGHTS, maxH]
    const next = heights.find(h => h > panelHeight + 10) ?? SNAP_HEIGHTS[0]
    setPanelHeight(next)
  }, [panelHeight])

  // ── Year navigation ───────────────────────────────────────────────────────
  const goToPrevYear = useCallback(() => {
    const years = timelineData?.years?.map(y => y.year) ?? []
    const idx = years.indexOf(selectedYear)
    if (idx > 0) { setSelectedYear(years[idx - 1]); setPlaying(false) }
  }, [timelineData, selectedYear])

  const goToNextYear = useCallback(() => {
    const years = timelineData?.years?.map(y => y.year) ?? []
    const idx = years.indexOf(selectedYear)
    if (idx < years.length - 1) { setSelectedYear(years[idx + 1]); setPlaying(false) }
  }, [timelineData, selectedYear])

  // ── Draw ──────────────────────────────────────────────────────────────────
  const handleDrawCreate = useCallback((feature) => {
    console.log('AOI drawn:', feature)
  }, [])

  // ── Derived ───────────────────────────────────────────────────────────────
  const selectedYearData = timelineData?.years?.find(y => y.year === selectedYear)
  const isSurveyYear = selectedYearData?.has_survey
  const panelOpen = !!selectedSlug

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="research-page">

      {/* ── Chatbot sidebar — collapsible ── */}
      <div className={`chat-sidebar-wrapper ${sidebarOpen ? 'open' : ''}`}>
        <ChatSidebar
          selectedSlug={selectedSlug}
          onCollapse={() => setSidebarOpen(false)}
        />
      </div>

      {/* Expand button — visible when sidebar is collapsed */}
      {!sidebarOpen && (
        <button className="chat-expand" onClick={() => setSidebarOpen(true)} title="Open assistant">
          ›
        </button>
      )}

      {/* ── Main area: map on top, colony panel on bottom ── */}
      <div className="research-main">

        {/* Map */}
        <div className="research-map">
          <ColonySearchBar onSelect={handleSearchBarSelect} />

          <Map
            ref={mapRef}
            {...viewState}
            onMove={e => setViewState(e.viewState)}
            onClick={handleMapClick}
            interactiveLayerIds={['clusters', 'colony-markers']}
            mapStyle={mapStyle}
            style={{ width: '100%', height: '100%' }}
            cursor="pointer"
            onLoad={(e) => {
              setMapStyleLoaded(true)
              // 'style.load' fires after every subsequent style switch (including imports).
              // 'onLoad' only fires once, so we register the persistent listener here.
              e.target.on('style.load', () => setMapStyleLoaded(true))
            }}
          >
            {mapStyleLoaded && coloniesGeoJSON.features.length > 0 && (
              <Source
                id="colonies"
                type="geojson"
                data={coloniesGeoJSON}
                cluster={true}
                clusterMaxZoom={11}
                clusterRadius={50}
              >
                <Layer {...clusterLayer} />
                <Layer {...clusterCountLayer} />
                <Layer {...markerLayer} />
              </Source>
            )}

            {mapStyleLoaded && mapFrameGeoJSON?.features?.length > 0 && (
              <Source id="spectral-overlay" type="geojson" data={mapFrameGeoJSON}>
                <Layer {...spectralOverlayLayer} />
              </Source>
            )}

            <NavigationControl position="bottom-right" showCompass={false} />
            <DrawControl onDrawCreate={handleDrawCreate} />
          </Map>

          {/* Map style toggle */}
          <div className="map-style-toggle">
            <button
              className={`map-style-toggle-btn ${isSatellite ? 'active' : ''}`}
              onClick={() => { setMapStyleLoaded(false); setMapStyle('mapbox://styles/mapbox/satellite-streets-v12') }}
            >Satellite</button>
            <span className="map-style-divider" />
            <button
              className={`map-style-toggle-btn ${!isSatellite ? 'active' : ''}`}
              onClick={() => { setMapStyleLoaded(false); setMapStyle('mapbox://styles/mapbox/standard') }}
            >Map</button>
          </div>

          {!panelOpen && (
            <div className="map-hint"><span>Click a colony to explore</span></div>
          )}

          {selectedYear && (
            <div className="year-badge">
              <span>{selectedYear}</span>
              {isSurveyYear && <span className="survey-badge">Survey</span>}
              {selectedYearData?.sensor && <span className="sensor-label">{selectedYearData.sensor}</span>}
            </div>
          )}
        </div>

        {/* ── Colony bottom panel — video player style ── */}
        <div
          className={`colony-bottom-panel ${panelOpen ? 'open' : ''}`}
          style={{
            height: panelOpen ? `${panelHeight}px` : 0,
            transition: dragging ? 'none' : undefined,
          }}
        >
          {panelOpen && (
            <>
              <div
                className="panel-drag-handle"
                onMouseDown={handleDragStart}
                onDoubleClick={handleDragDoubleClick}
                title="Drag to resize · Double-click to snap"
              />

              <div className="bottom-panel-inner">

                {/* Left: colony header + tabs + detail card */}
                <div className="bottom-panel-left">
                  <div className="bottom-panel-header">
                    <div className="panel-colony-name">
                      {profileLoading ? 'Loading…' : (profile?.colony ?? '…')}
                      {profile?.state && <span className="panel-state">{profile.state}</span>}
                    </div>
                    {profile && (
                      <div className="panel-meta-inline">
                        {profile.primary_habitat && <span>{profile.primary_habitat}</span>}
                        {profile.geo_region && <span>{profile.geo_region}</span>}
                        <span>{Object.keys(profile.survey_data ?? {}).length} surveys</span>
                      </div>
                    )}
                    <button className="panel-close" onClick={handleDeselect}>✕</button>
                  </div>
                  <div className="panel-tabs">
                    {PANEL_TABS.map(tab => (
                      <button
                        key={tab}
                        className={`panel-tab ${activeTab === tab ? 'active' : ''}`}
                        onClick={() => setActiveTab(tab)}
                      >
                        {tab.charAt(0).toUpperCase() + tab.slice(1)}
                      </button>
                    ))}
                  </div>
                  <div className="bottom-detail">
                    {activeTab === 'timeline' && selectedYear && selectedYearData?.has_survey && (
                      <SurveyCard data={profile?.survey_data} year={selectedYear} />
                    )}
                    {activeTab === 'timeline' && selectedYear && !selectedYearData?.has_survey && (
                      <GapFillCard yearData={selectedYearData} index={selectedIndex} />
                    )}
                    {activeTab === 'data' && profile && (
                      <div className="data-rows">
                        {Object.entries(profile.survey_data ?? {}).sort().map(([yr, d]) => (
                          <div key={yr} className="data-year-row">
                            <span className="data-year">{yr}</span>
                            <span>{d.total_birds?.toLocaleString() ?? '–'} birds</span>
                            <span>{d.active_nests?.toLocaleString() ?? '–'} nests</span>
                            <span className={`trajectory ${d.colony_trajectory}`}>{d.colony_trajectory ?? '–'}</span>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>

                {/* Right: video player (insights + seeker + controls) */}
                <div className="colony-player">
                  <YearInsights
                    year={selectedYear}
                    yearData={selectedYearData}
                    surveyData={profile?.survey_data}
                  />
                  {timelineLoading && <div className="player-loading">Loading timeline…</div>}
                  {timelineData && (
                    <VideoSeeker
                      years={timelineData.years}
                      selectedYear={selectedYear}
                      onSelectYear={y => { setSelectedYear(y); setPlaying(false) }}
                    />
                  )}
                  <div className="player-controls">
                    <select className="index-select" value={selectedIndex} onChange={e => setSelectedIndex(e.target.value)}>
                      {INDICES.map(i => <option key={i} value={i}>{i}</option>)}
                    </select>
                    <div className="player-transport">
                      <button className="transport-btn" onClick={goToPrevYear} disabled={!timelineData} title="Previous year">⏮</button>
                      <button className="transport-btn play" onClick={() => setPlaying(p => !p)} disabled={!timelineData}>
                        {playing ? '⏸' : '▶'}
                      </button>
                      <button className="transport-btn" onClick={goToNextYear} disabled={!timelineData} title="Next year">⏭</button>
                    </div>
                    <select className="speed-select" value={playSpeed} onChange={e => setPlaySpeed(Number(e.target.value))}>
                      <option value={0.5}>0.5×</option>
                      <option value={1}>1×</option>
                      <option value={2}>2×</option>
                    </select>
                  </div>
                </div>

              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
