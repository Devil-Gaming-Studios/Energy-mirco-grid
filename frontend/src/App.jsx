import { useState, useEffect, useCallback, useRef } from 'react'
import axios from 'axios'
import {
  ResponsiveContainer, ComposedChart, Line, Area, XAxis, YAxis,
  CartesianGrid, Tooltip, Legend,
} from 'recharts'
import Scene3D from './Scene3D.jsx'

const API_URL = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '')

const DEFAULTS = {
  lat: 23.2599,
  lon: 77.4126,
  solar_capacity_kw: 50,
  wind_capacity_kw: 30,
  battery_kwh: 100,
  load_scale_kw: 50,
}

// ---- Browser-side Open-Meteo fetch --------------------------------------
// Open-Meteo rate-limits by IP. The backend on a shared host (e.g. Render) shares
// its IP with strangers, so we fetch from the user's own browser (own IP, own
// free quota, CORS is allowed) and hand the result to the backend's cache.
const OPEN_METEO = 'https://api.open-meteo.com/v1/forecast'
const WEATHER_TTL_MS = 3 * 60 * 60 * 1000

async function fetchOpenMeteoInBrowser(lat, lon) {
  const key = `om:${lat.toFixed(2)},${lon.toFixed(2)}`
  let stale = null
  try {
    const hit = JSON.parse(localStorage.getItem(key) || 'null')
    if (hit) {
      if (Date.now() - hit.t < WEATHER_TTL_MS) return hit.data
      stale = hit.data
    }
  } catch { /* ignore corrupt cache */ }

  try {
    const res = await axios.get(OPEN_METEO, {
      timeout: 15000,
      params: {
        latitude: lat,
        longitude: lon,
        hourly: 'shortwave_radiation,cloudcover,windspeed_10m,temperature_2m',
        past_days: 3,
        forecast_days: 2,
        timezone: 'auto',
      },
    })
    try {
      localStorage.setItem(key, JSON.stringify({ t: Date.now(), data: res.data }))
    } catch { /* storage full/blocked: fine */ }
    return res.data
  } catch (err) {
    if (stale) return stale          // stale browser copy beats nothing
    throw err
  }
}

// Fetch in the browser and push to the backend cache. Never throws: if the
// browser can't reach Open-Meteo either, the backend still tries its own fallbacks.
async function pushBrowserWeather(lat, lon) {
  try {
    const data = await fetchOpenMeteoInBrowser(lat, lon)
    await axios.post(`${API_URL}/api/weather/ingest`, { lat, lon, data })
    return true
  } catch (err) {
    console.warn('Browser weather fetch/push failed; backend will use its own sources.', err)
    return false
  }
}

function clearSkyIrradiance(hourOfDay) {
  const x = Math.max(0, Math.sin(((hourOfDay - 6) / 12) * Math.PI))
  return 1000 * Math.pow(x, 1.1)
}

function formatHour(ts) {
  const d = new Date(ts)
  return d.getHours().toString().padStart(2, '0') + ':00'
}

export default function App() {
  const [mode, setMode] = useState('weather')
  const [params, setParams] = useState(DEFAULTS)
  const [sim, setSim] = useState({
    timeOfDay: 12,
    cloudCover: 20,
    windSpeed: 40,
  })
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [sensorStatus, setSensorStatus] = useState(null)
  const [forecastMethod, setForecastMethod] = useState('sensors')
  const [modelStatus, setModelStatus] = useState(null)
  const [uploadStatus, setUploadStatus] = useState(null)
  const [historyCount, setHistoryCount] = useState(null)
  const [playState, setPlayState] = useState(null)
  const [backfillStatus, setBackfillStatus] = useState(null)
  const [csvStatus, setCsvStatus] = useState(null)

  const debounceRef = useRef(null)
  const playTimerRef = useRef(null)

  const loadForecast = useCallback(async (source, p) => {
    setLoading(true)
    setError(null)

    try {
      if (source === 'weather') {
        await pushBrowserWeather(p.lat, p.lon)
      }
      const res = await axios.get(`${API_URL}/api/forecast`, {
        params: {
          source,
          ...p,
        },
      })

      setData(res.data)
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        err.message ||
        'Failed to load forecast'
      )
    } finally {
      setLoading(false)
    }
  }, [])

  const refreshHistoryStatus = useCallback(async () => {
    try {
      const res = await axios.get(`${API_URL}/api/sensors/history`, {
        params: {
          limit: 72,
        },
      })

      setHistoryCount(res.data.count)
    } catch {
      // Non-critical status display
    }
  }, [])

  const pushSensorReading = useCallback(
    async (simState, capacityParams, method) => {
      const irradiance =
        clearSkyIrradiance(simState.timeOfDay) *
        (1 - (simState.cloudCover / 100) * 0.85)

      const temperature =
        18 +
        10 *
          Math.max(
            0,
            Math.sin(((simState.timeOfDay - 6) / 12) * Math.PI)
          )

      const payload = {
        cloud_cover_pct: simState.cloudCover,
        wind_speed_kmh: simState.windSpeed,
        solar_irradiance_wm2: Math.round(irradiance),
        temperature_c: Math.round(temperature * 10) / 10,
        sim_hour: simState.timeOfDay,
        source: '3d-twin',
      }

      try {
        await axios.post(
          `${API_URL}/api/sensors/ingest`,
          payload,
          {
            params: {
              update_latest: true,
            },
          }
        )

        setSensorStatus({
          ok: true,
          payload,
        })

        refreshHistoryStatus()

        await loadForecast(
          method,
          capacityParams
        )
      } catch (err) {
        setSensorStatus({
          ok: false,
          error:
            err.response?.data?.detail ||
            err.message,
        })
      }
    },
    [loadForecast, refreshHistoryStatus]
  )

  const generateThreeDayTrace = useCallback(
    (startHour, startCloud, startWind) => {
      let cloud = startCloud
      let wind = startWind
      const trace = []

      for (let i = 0; i < 72; i++) {
        cloud = Math.min(
          100,
          Math.max(
            0,
            cloud + (Math.random() - 0.5) * 16
          )
        )

        wind = Math.min(
          100,
          Math.max(
            0,
            wind + (Math.random() - 0.5) * 8
          )
        )

        const hourOfDay = (startHour + i) % 24

        const irradiance =
          clearSkyIrradiance(hourOfDay) *
          (1 - (cloud / 100) * 0.85)

        const temperature =
          18 +
          10 *
            Math.max(
              0,
              Math.sin(((hourOfDay - 6) / 12) * Math.PI)
            )

        trace.push({
          hour: hourOfDay,
          cloudCover: Math.round(cloud),
          windSpeed: Math.round(wind),
          irradiance: Math.round(irradiance),
          temperature:
            Math.round(temperature * 10) / 10,
        })
      }

      return trace
    },
    []
  )

  const handlePlayThreeDays = useCallback(() => {
    if (playState?.running) return

    const trace = generateThreeDayTrace(
      sim.timeOfDay,
      sim.cloudCover,
      sim.windSpeed
    )

    setPlayState({
      running: true,
      step: 0,
      total: trace.length,
    })

    let i = 0

    playTimerRef.current = setInterval(() => {
      if (i >= trace.length) {
        clearInterval(playTimerRef.current)

        const readings = trace.map((p) => ({
          cloud_cover_pct: p.cloudCover,
          wind_speed_kmh: p.windSpeed,
          solar_irradiance_wm2: p.irradiance,
          temperature_c: p.temperature,
          sim_hour: p.hour,
          source: '3d-twin-playback',
        }))

        axios
          .post(
            `${API_URL}/api/sensors/ingest-batch`,
            {
              readings,
              replace_history: true,
            }
          )
          .then(() => {
            setPlayState({
              running: false,
              step: trace.length,
              total: trace.length,
            })

            refreshHistoryStatus()

            loadForecast(
              forecastMethod,
              params
            )
          })
          .catch((err) => {
            setPlayState(null)

            setSensorStatus({
              ok: false,
              error:
                err.response?.data?.detail ||
                err.message,
            })
          })

        return
      }

      const p = trace[i]

      setSim({
        timeOfDay: p.hour,
        cloudCover: p.cloudCover,
        windSpeed: p.windSpeed,
      })

      setPlayState({
        running: true,
        step: i + 1,
        total: trace.length,
      })

      i += 1
    }, 70)
  }, [
    sim,
    generateThreeDayTrace,
    forecastMethod,
    params,
    playState,
    loadForecast,
    refreshHistoryStatus,
  ])

  useEffect(() => {
    return () => {
      if (playTimerRef.current) {
        clearInterval(playTimerRef.current)
      }

      if (debounceRef.current) {
        clearTimeout(debounceRef.current)
      }
    }
  }, [])

  const handleBackfillWeather = useCallback(async () => {
    setBackfillStatus({
      loading: true,
    })

    try {
      await pushBrowserWeather(params.lat, params.lon)
      const res = await axios.post(
        `${API_URL}/api/sensors/backfill-from-weather`,
        null,
        {
          params: {
            lat: params.lat,
            lon: params.lon,
            hours: 72,
          },
        }
      )

      setBackfillStatus({
        ok: true,
        count: res.data.count,
        dataSource: res.data.data_source,
      })

      await refreshHistoryStatus()

      await loadForecast(
        forecastMethod,
        params
      )
    } catch (err) {
      setBackfillStatus({
        ok: false,
        error:
          err.response?.data?.detail ||
          err.message,
      })
    }
  }, [
    params,
    forecastMethod,
    loadForecast,
    refreshHistoryStatus,
  ])

  const handleCsvUpload = async (e) => {
    const file = e.target.files?.[0]

    if (!file) return

    const formData = new FormData()
    formData.append('file', file)

    setCsvStatus({
      loading: true,
    })

    try {
      const res = await axios.post(
        `${API_URL}/api/sensors/ingest-csv`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          params: {
            replace_history: true,
          },
        }
      )

      setCsvStatus({
        ok: true,
        count: res.data.count,
      })

      await refreshHistoryStatus()

      await loadForecast(
        forecastMethod,
        params
      )
    } catch (err) {
      setCsvStatus({
        ok: false,
        error:
          err.response?.data?.detail ||
          err.message,
      })
    }
  }

  useEffect(() => {
    if (mode === 'weather') {
      loadForecast(
        'weather',
        params
      )
    } else {
      pushSensorReading(
        sim,
        params,
        forecastMethod
      )
    }

    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode])

  useEffect(() => {
    axios
      .get(`${API_URL}/api/model/status`)
      .then((res) => {
        setModelStatus(res.data)
      })
      .catch(() => {})
  }, [])

  const handleParamChange = (key) => (e) => {
    const next = {
      ...params,
      [key]: parseFloat(e.target.value),
    }

    setParams(next)
  }

  const handleWeatherSubmit = (e) => {
    e.preventDefault()

    loadForecast(
      'weather',
      params
    )
  }

  const handleSimChange = (key) => (e) => {
    const next = {
      ...sim,
      [key]: parseFloat(e.target.value),
    }

    setSim(next)

    if (debounceRef.current) {
      clearTimeout(debounceRef.current)
    }

    debounceRef.current = setTimeout(() => {
      pushSensorReading(
        next,
        params,
        forecastMethod
      )
    }, 250)
  }

  const handleCapacityChangeSim = (key) => (e) => {
    const next = {
      ...params,
      [key]: parseFloat(e.target.value),
    }

    setParams(next)

    if (debounceRef.current) {
      clearTimeout(debounceRef.current)
    }

    debounceRef.current = setTimeout(() => {
      pushSensorReading(
        sim,
        next,
        forecastMethod
      )
    }, 250)
  }

  const handleMethodChange = (method) => {
    setForecastMethod(method)

    pushSensorReading(
      sim,
      params,
      method
    )
  }

  const handleUploadWeights = async (e) => {
    const file = e.target.files?.[0]

    if (!file) return

    const formData = new FormData()
    formData.append('file', file)

    setUploadStatus({
      loading: true,
    })

    try {
      const res = await axios.post(
        `${API_URL}/api/model/upload-weights`,
        formData,
        {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        }
      )

      setUploadStatus({
        ok: true,
      })

      setModelStatus({
        loaded: true,
        weights_path: res.data.weights_path,
      })
    } catch (err) {
      setUploadStatus({
        ok: false,
        error:
          err.response?.data?.detail ||
          err.message,
      })
    }
  }

  const chartData = data
    ? data.timestamps.map((ts, i) => ({
        time:
          mode === 'weather'
            ? formatHour(ts)
            : `+${i}h`,
        Solar: data.solar_kw[i],
        Wind: data.wind_kw[i],
        Load: data.load_kw[i],
        Battery: data.battery_soc_kwh[i],
        GridDraw: data.grid_draw_kw[i],
      }))
    : []

  const totalSolar = data
    ? data.solar_kw
        .slice(0, 24)
        .reduce((a, b) => a + b, 0)
    : 0

  const totalWind = data
    ? data.wind_kw
        .slice(0, 24)
        .reduce((a, b) => a + b, 0)
    : 0

  const totalLoad = data
    ? data.load_kw
        .slice(0, 24)
        .reduce((a, b) => a + b, 0)
    : 0

  const renewShare =
    totalLoad > 0
      ? Math.min(
          100,
          ((totalSolar + totalWind) / totalLoad) * 100
        )
      : 0

  return (
    <div className="app">
      <header>
        <div>
          <h1>GridSense</h1>

          <p>
            Solar &amp; wind forecast, battery dispatch,
            and load-scheduling recommendations —
            from live weather or a 3D digital-twin
            sensor simulation.
          </p>
        </div>
      </header>

      <div className="mode-toggle">
        <button
          className={mode === 'weather' ? 'active' : ''}
          onClick={() => setMode('weather')}
        >
          Live weather
        </button>

        <button
          className={mode === 'sensors' ? 'active' : ''}
          onClick={() => setMode('sensors')}
        >
          3D sensor simulation
        </button>
      </div>

      {mode === 'weather' && (
        <form
          className="controls"
          onSubmit={handleWeatherSubmit}
        >
          <div className="field">
            <label>Latitude</label>

            <input
              type="number"
              step="0.0001"
              value={params.lat}
              onChange={handleParamChange('lat')}
            />
          </div>

          <div className="field">
            <label>Longitude</label>

            <input
              type="number"
              step="0.0001"
              value={params.lon}
              onChange={handleParamChange('lon')}
            />
          </div>

          <div className="field">
            <label>Solar capacity (kW)</label>

            <input
              type="number"
              min="1"
              value={params.solar_capacity_kw}
              onChange={handleParamChange(
                'solar_capacity_kw'
              )}
            />
          </div>

          <div className="field">
            <label>Wind capacity (kW)</label>

            <input
              type="number"
              min="0"
              value={params.wind_capacity_kw}
              onChange={handleParamChange(
                'wind_capacity_kw'
              )}
            />
          </div>

          <div className="field">
            <label>Battery capacity (kWh)</label>

            <input
              type="number"
              min="1"
              value={params.battery_kwh}
              onChange={handleParamChange(
                'battery_kwh'
              )}
            />
          </div>

          <div className="field">
            <label>Site load scale (kW)</label>

            <input
              type="number"
              min="1"
              value={params.load_scale_kw}
              onChange={handleParamChange(
                'load_scale_kw'
              )}
            />
          </div>

          <button
            type="submit"
            disabled={loading}
          >
            {loading
              ? 'Loading…'
              : 'Update forecast'}
          </button>
        </form>
      )}

      {mode === 'sensors' && (
        <div className="sim-layout">
          <div className="scene-wrap">
            <Scene3D
              timeOfDay={sim.timeOfDay}
              cloudCover={sim.cloudCover}
              windSpeed={sim.windSpeed}
            />
          </div>

          <div className="sim-controls">
            <h2>Environment controls</h2>

            <div className="slider-field">
              <label>
                Time of day
                <span className="num">
                  {sim.timeOfDay.toFixed(1)}:00
                </span>
              </label>

              <input
                type="range"
                min="0"
                max="24"
                step="0.1"
                value={sim.timeOfDay}
                onChange={handleSimChange(
                  'timeOfDay'
                )}
              />
            </div>

            <div className="slider-field">
              <label>
                Cloud cover
                <span className="num">
                  {sim.cloudCover}%
                </span>
              </label>

              <input
                type="range"
                min="0"
                max="100"
                value={sim.cloudCover}
                onChange={handleSimChange(
                  'cloudCover'
                )}
              />
            </div>

            <div className="slider-field">
              <label>
                Wind speed
                <span className="num">
                  {sim.windSpeed} km/h
                </span>
              </label>

              <input
                type="range"
                min="0"
                max="100"
                value={sim.windSpeed}
                onChange={handleSimChange(
                  'windSpeed'
                )}
              />
            </div>

            <h2>
              72-hour sequence data (for the LSTM)
            </h2>

            <div className="history-status">
              <div className="history-bar">
                <div
                  className="history-fill"
                  style={{
                    width: `${Math.min(
                      100,
                      ((historyCount ?? 0) / 72) *
                        100
                    )}%`,
                  }}
                />
              </div>

              <span className="num">
                {historyCount ?? 0} / 72 hours buffered
              </span>
            </div>

            <button
              type="button"
              className="play-btn"
              onClick={handlePlayThreeDays}
              disabled={playState?.running}
            >
              {playState?.running
                ? `Simulating hour ${playState.step}/${playState.total}…`
                : '▶ Play 3 simulated days'}
            </button>

            <p className="hint">
              Animates the scene through 3 fictional
              days and bulk-loads them as real sequential
              history — the fastest way to get a genuine
              72-hour window for the LSTM.
            </p>

            <div className="backfill-row">
              <div className="field small">
                <label>Lat</label>

                <input
                  type="number"
                  step="0.0001"
                  value={params.lat}
                  onChange={handleParamChange(
                    'lat'
                  )}
                />
              </div>

              <div className="field small">
                <label>Lon</label>

                <input
                  type="number"
                  step="0.0001"
                  value={params.lon}
                  onChange={handleParamChange(
                    'lon'
                  )}
                />
              </div>

              <button
                type="button"
                onClick={handleBackfillWeather}
                disabled={backfillStatus?.loading}
              >
                {backfillStatus?.loading
                  ? 'Fetching…'
                  : 'Backfill from real weather'}
              </button>
            </div>

            <p className="hint">
              Pulls the actual last 3 days of hourly
              weather for this location from Open-Meteo —
              real data, not simulated.
            </p>

            {backfillStatus?.ok && (
              <div className="upload-note ok">
                {backfillStatus.dataSource &&
                backfillStatus.dataSource !== 'open-meteo'
                  ? `Loaded ${backfillStatus.count} hours of ESTIMATED weather (Open-Meteo is rate-limiting; using ${backfillStatus.dataSource}). Past hours are approximations, not measurements.`
                  : `Loaded ${backfillStatus.count} real hours from Open-Meteo.`}
              </div>
            )}

            {backfillStatus?.ok === false && (
              <div className="upload-note bad">
                Backfill failed:{' '}
                {backfillStatus.error}
              </div>
            )}

            <label className="upload-btn">
              Upload 72h history (.csv)

              <input
                type="file"
                accept=".csv"
                onChange={handleCsvUpload}
                hidden
              />
            </label>

            <p className="hint">
              Columns: hour, wind_speed_kmh,
              solar_irradiance_wm2 (optional:
              cloud_cover_pct, temperature_c).{' '}
              <a
                href="/sample_history_72h.csv"
                download
              >
                Download a sample file
              </a>.
            </p>

            {csvStatus?.loading && (
              <div className="upload-note">
                Uploading…
              </div>
            )}

            {csvStatus?.ok && (
              <div className="upload-note ok">
                Loaded {csvStatus.count} hours from CSV.
              </div>
            )}

            {csvStatus?.ok === false && (
              <div className="upload-note bad">
                Upload failed: {csvStatus.error}
              </div>
            )}

            <h2>Microgrid capacity</h2>

            <div className="slider-field">
              <label>
                Solar capacity
                <span className="num">
                  {params.solar_capacity_kw} kW
                </span>
              </label>

              <input
                type="range"
                min="5"
                max="200"
                value={params.solar_capacity_kw}
                onChange={handleCapacityChangeSim(
                  'solar_capacity_kw'
                )}
              />
            </div>

            <div className="slider-field">
              <label>
                Wind capacity
                <span className="num">
                  {params.wind_capacity_kw} kW
                </span>
              </label>

              <input
                type="range"
                min="0"
                max="150"
                value={params.wind_capacity_kw}
                onChange={handleCapacityChangeSim(
                  'wind_capacity_kw'
                )}
              />
            </div>

            <div className="slider-field">
              <label>
                Battery capacity
                <span className="num">
                  {params.battery_kwh} kWh
                </span>
              </label>

              <input
                type="range"
                min="10"
                max="300"
                value={params.battery_kwh}
                onChange={handleCapacityChangeSim(
                  'battery_kwh'
                )}
              />
            </div>

            {sensorStatus && (
              <div
                className={`sensor-status ${
                  sensorStatus.ok ? 'ok' : 'bad'
                }`}
              >
                {sensorStatus.ok
                  ? `Sensor reading sent — irradiance ${sensorStatus.payload.solar_irradiance_wm2} W/m², wind ${sensorStatus.payload.wind_speed_kmh} km/h`
                  : `Sensor push failed: ${sensorStatus.error}`}
              </div>
            )}

            <h2>Forecast method</h2>

            <div className="method-toggle">
              <button
                className={
                  forecastMethod === 'sensors'
                    ? 'active'
                    : ''
                }
                onClick={() =>
                  handleMethodChange('sensors')
                }
              >
                Persistence (rule-based)
              </button>

              <button
                className={
                  forecastMethod === 'ml'
                    ? 'active'
                    : ''
                }
                onClick={() =>
                  handleMethodChange('ml')
                }
              >
                Trained ML model
              </button>
            </div>

            {forecastMethod === 'ml' && (
              <div className="model-upload">
                <div
                  className={`model-badge ${
                    modelStatus?.loaded
                      ? 'ok'
                      : 'bad'
                  }`}
                >
                  {modelStatus?.loaded
                    ? 'Model weights loaded'
                    : 'No trained weights loaded yet'}
                </div>

                <label className="upload-btn">
                  Upload trained model weights (.pth)

                  <input
                    type="file"
                    accept=".pth"
                    onChange={handleUploadWeights}
                    hidden
                  />
                </label>

                {uploadStatus?.loading && (
                  <div className="upload-note">
                    Uploading…
                  </div>
                )}

                {uploadStatus?.ok && (
                  <div className="upload-note ok">
                    Weights loaded — forecast updated
                    below.
                  </div>
                )}

                {uploadStatus?.ok === false && (
                  <div className="upload-note bad">
                    Upload failed:{' '}
                    {uploadStatus.error}
                  </div>
                )}

                <p className="hint">
                  Train weights in{' '}
                  <code>
                    colab/train_forecaster.ipynb
                  </code>
                  , download the .pth, and upload it
                  here.
                </p>
              </div>
            )}
          </div>
        </div>
      )}

      {data?.data_source &&
        data.data_source !== 'open-meteo' &&
        data.source === 'weather' && (
          <div className="upload-note bad">
            Open-Meteo is rate-limiting; showing fallback weather ({data.data_source}).
            Radiation is estimated from cloud cover. It will switch back automatically.
          </div>
        )}

      {error && (
        <div className="error">
          Could not reach the backend: {error}
        </div>
      )}

      {data && (
        <>
          <section className="kpis">
            <div className="kpi">
              <div className="label">
                Forecast solar (24h)
              </div>

              <div className="val">
                {totalSolar.toFixed(1)} kWh
              </div>
            </div>

            <div className="kpi">
              <div className="label">
                Forecast wind (24h)
              </div>

              <div className="val">
                {totalWind.toFixed(1)} kWh
              </div>
            </div>

            <div className="kpi">
              <div className="label">
                Renewable share of load
              </div>

              <div className="val">
                {renewShare.toFixed(0)}%
              </div>
            </div>

            <div className="kpi">
              <div className="label">
                Grid draw reduction
              </div>

              <div className="val">
                {data.stats.grid_draw_reduction_pct}%
              </div>

              <div className="sub">
                vs. no-battery baseline
              </div>
            </div>
          </section>

          <section className="card">
            <h2>
              48-hour forecast{' '}
              <span className="source-badge">
                {data.source === 'ml'
                  ? 'trained ML model'
                  : data.source === 'sensors'
                    ? '3D twin (persistence)'
                    : 'live weather'}
              </span>
            </h2>

            {data.sensor_meta?.real_readings_used !==
              undefined && (
              <p className="sensor-meta-note">
                LSTM input window:{' '}
                {data.sensor_meta.real_readings_used}{' '}
                real hours,{' '}
                {data.sensor_meta.padded_readings}{' '}
                padded/extrapolated.
              </p>
            )}

            <div
              style={{
                width: '100%',
                height: 340,
              }}
            >
              <ResponsiveContainer>
                <ComposedChart data={chartData}>
                  <CartesianGrid
                    strokeDasharray="3 3"
                    stroke="#2A3847"
                  />

                  <XAxis
                    dataKey="time"
                    tick={{
                      fill: '#8FA3B5',
                      fontSize: 12,
                    }}
                    interval={3}
                  />

                  <YAxis
                    tick={{
                      fill: '#8FA3B5',
                      fontSize: 12,
                    }}
                    label={{
                      value: 'kW / kWh',
                      angle: -90,
                      position: 'insideLeft',
                      fill: '#8FA3B5',
                    }}
                  />

                  <Tooltip
                    contentStyle={{
                      background: '#1C2836',
                      border: '1px solid #2A3847',
                      color: '#E7EDF3',
                    }}
                  />

                  <Legend />

                  <Area
                    type="monotone"
                    dataKey="Battery"
                    stroke="#6FCF97"
                    fill="#6FCF9720"
                    strokeWidth={1.5}
                  />

                  <Line
                    type="monotone"
                    dataKey="Solar"
                    stroke="#F2A93B"
                    dot={false}
                    strokeWidth={2}
                  />

                  <Line
                    type="monotone"
                    dataKey="Wind"
                    stroke="#4FB6C7"
                    dot={false}
                    strokeWidth={2}
                  />

                  <Line
                    type="monotone"
                    dataKey="Load"
                    stroke="#B98CE0"
                    dot={false}
                    strokeWidth={2}
                    strokeDasharray="4 3"
                  />

                  <Line
                    type="monotone"
                    dataKey="GridDraw"
                    stroke="#E5735A"
                    dot={false}
                    strokeWidth={1.5}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </section>

          <section className="card">
            <h2>Recommended actions</h2>

            <div className="recs">
              {data.recommendations.map((r, i) => (
                <div
                  className={`rec ${r.severity}`}
                  key={i}
                >
                  <div className="t">
                    {r.title}
                  </div>

                  <div className="d">
                    {r.detail}
                  </div>
                </div>
              ))}
            </div>
          </section>
        </>
      )}

      <footer>
        {mode === 'weather'
          ? 'Weather data from Open-Meteo (open, no API key). Load is a synthetic daily curve.'
          : "Environment data comes from the 3D scene's sliders, sent to the backend /api/sensors/ingest exactly as a real weather station would, then projected forward by the backend."}
      </footer>
    </div>
  )
}