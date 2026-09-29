# GridSense — Renewable Forecast & Microgrid Optimization

A working prototype: a **Python (FastAPI)** backend that pulls **real, live weather
forecasts** from [Open-Meteo](https://open-meteo.com) (free, no API key needed) and
converts them into solar/wind power estimates and a battery dispatch plan, plus a
**React** frontend that visualizes it and shows plain-language scheduling
recommendations.

## Two data sources, one API contract

`GET /api/forecast` accepts a `source` parameter and returns the *same shape*
either way:

- **`source=weather`** — real solar radiation, cloud cover, wind speed and
  temperature, fetched live from Open-Meteo for your exact latitude/longitude.
- **`source=sensors`** — the latest reading posted to `POST /api/sensors/ingest`,
  projected forward 48 hours using a persistence model (see below).
- **`source=ml`** — a rolling window of ingested sensor readings run through a
  trained LSTM (see "Training and using your own forecasting model" below).

`POST /api/sensors/ingest` is a genuine sensor-ingestion endpoint — the shape
a real on-site weather station (pyranometer + anemometer) would call. It has
no idea whether the reading came from real hardware or from something else.
That "something else" is the included **3D digital twin**: a Three.js scene
in the React app with sliders for time-of-day, cloud cover, and wind speed.
Moving a slider does two things at once:
1. Updates the 3D scene instantly (sky color, sun position, cloud density,
   turbine blade speed, solar panel glow) — pure visualization, client-side.
2. Computes the corresponding sensor readings (irradiance from a clear-sky
   model attenuated by cloud cover, wind speed directly, a simple diurnal
   temperature curve) and POSTs them to `/api/sensors/ingest`, then asks
   `/api/forecast?source=sensors` to re-read from that reading.

So the dashboard is, mechanically, reading sensor data through the exact same
code path it would use for a real site — the 3D environment is just standing
in for the physical sensors during development/demos. Swapping in real
hardware later means pointing a gateway at `POST /api/sensors/ingest` with the
same payload shape; nothing else in the pipeline changes.

**Persistence forecasting from a single live reading:** with only one
instantaneous sensor reading you can't get a real 48-hour NWP forecast, but
you can build a legitimate short-horizon projection — the same technique
weather services use as a baseline. The backend (`backend/sensors.py`) infers
a "cloud attenuation factor" by comparing the live irradiance reading to a
clear-sky model for that hour, then projects that same attenuation forward
across the clear-sky diurnal curve; wind is held at its last measured value
(wind persistence). This is what `sensor_meta.inferred_cloud_attenuation` in
the API response shows.

## What's real vs. simulated

- **Real (weather mode):** solar radiation, cloud cover, wind speed,
  temperature — fetched live per-location from Open-Meteo's forecast API.
- **Real ingestion path, simulated source (sensor mode):** the 3D twin
  computes physically-motivated readings and sends them through the same
  `/api/sensors/ingest` endpoint a real gateway would use.
- **Trained on real data, validated on real held-out data (ML mode):** the
  LSTM in `colab/train_forecaster.ipynb` trains and backtests on real NASA
  POWER hourly history. In the running app, `source=ml` uses that trained
  model but feeds it whatever's in the live sensor-history buffer, which is
  real once enough readings have been ingested and padded/extrapolated
  before that — see "Training and using your own forecasting model" below.
- **Derived from either source:** solar PV output (irradiance → kW via a
  standard performance-ratio model) and wind turbine output (wind speed → kW
  via a cut-in/rated/cut-out power curve) — `backend/forecast.py`.
- **Simulated (swap for your real data):** the load/demand curve, which uses a
  typical daily consumption shape. Replace `synthetic_load()` with a call to
  your smart-meter or utility data source.
- **Rule-based:** the battery optimizer is a greedy hourly dispatch algorithm
  (charge on surplus, discharge on deficit, draw remainder from grid). A
  production version could use linear programming (e.g. `PuLP` or
  `scipy.optimize`) to plan the full horizon jointly instead of hour-by-hour.

## Project structure

```
energy-microgrid-project/
├── backend/
│   ├── main.py           # FastAPI app: /api/forecast, /api/sensors/*
│   ├── forecast.py        # weather fetch, power models, battery optimizer
│   ├── sensors.py         # sensor ingestion store + persistence projection
│   └── requirements.txt
└── frontend/
    ├── src/
    │   ├── App.jsx         # dashboard UI, mode toggle, sensor push logic
    │   ├── Scene3D.jsx      # Three.js digital-twin scene
    │   ├── App.css
    │   └── main.jsx
    ├── index.html
    ├── package.json
    └── vite.config.js
```

## Running it locally

You'll need **Python 3.9+** and **Node.js 18+** installed.

### 1. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Check it's working: open http://127.0.0.1:8000/api/forecast in your browser —
you should get a JSON forecast for the default location (Bhopal, India).

### 2. Frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open the URL Vite prints (usually http://localhost:5173). The dev server proxies
`/api/*` requests to the backend on port 8000, so both need to be running.

### 3. Try it

**Live weather mode:** change latitude/longitude to your actual site, adjust
solar/wind capacity, battery size, and expected load, then click **Update
forecast**. The chart and recommendations update using the live weather
forecast for that location.

**3D sensor simulation mode:** click "3D sensor simulation" at the top. Drag
the time-of-day, cloud cover, or wind speed sliders — the 3D scene updates
immediately (sun position, sky color, clouds, spinning turbine blades, solar
panel glow), and about a quarter-second later the dashboard below refreshes
with a forecast built from that reading, exactly as it would from a live
sensor feed.

## Training and using your own forecasting model

`colab/train_forecaster.ipynb` trains an **LSTM** (`LSTMForecaster`) that
takes the past 72 hours of solar irradiance, wind speed, and temperature and
predicts the next 48 hours of irradiance and wind speed. Unlike an earlier
version of this project's model, the training data here is **real**: multi-year
hourly historical weather from [NASA POWER](https://power.larc.nasa.gov/) — a
free, no-API-key dataset covering any location on Earth. Run the notebook in
Google Colab (select a GPU runtime, though it trains fine on CPU too), then
either:

- **Drop the file in:** save the downloaded `solar_wind_forecaster.pth` to
  `backend/weights/solar_wind_forecaster.pth` and restart the backend, or
- **Upload it live:** in the frontend, switch to "3D sensor simulation" mode,
  set "Forecast method" to "Trained ML model", and use the upload control —
  it POSTs to `/api/model/upload-weights` and the backend loads it immediately,
  no restart needed.

Once loaded, `GET /api/forecast?source=ml` (and the "Trained ML model" toggle
in the sim panel) reads a rolling window of ingested sensor readings from
`backend/sensors.py` and runs it through the LSTM — so you can directly
compare it against the rule-based persistence projection (`source=sensors`)
from the same underlying readings.

**Why an LSTM specifically:** the model's input is now a genuine sequence —
72 consecutive real hourly measurements — which is exactly what recurrent
architectures are built to exploit (learning how conditions evolve over time,
not just their instantaneous value). A single-snapshot model (e.g. a plain
MLP) can't use that sequential structure.

**What the Colab notebook actually proves, and what it doesn't:**
- It fetches real NASA POWER hourly data, trains on an earlier period, and
  backtests on a **later, held-out real period the model never saw during
  training** — reporting MAE/RMSE in physical units (W/m², km/h), and
  comparing against a naive persistence baseline on the same data. That's a
  legitimate accuracy claim for the location you train it on.
- It does **not** validate the model against your actual deployment site's
  real sensors, or against ground-truth measurements (NASA POWER is itself a
  modeled reanalysis product, not a direct measurement). It's also
  location-specific — retrain per site for real deployment.
- The backend's live sensor-history buffer (`sensors.get_history_for_ml`)
  starts empty and pads with extrapolated values until 72 real hours have
  been ingested; the API response's `sensor_meta.real_readings_used` /
  `padded_readings` fields report this honestly rather than hiding it.
  See "Getting a real 72-hour window" below for three ways to fill it
  quickly instead of waiting three real days.

**Architecture note:** `backend/ml_model.py`'s `LSTMForecaster` class,
`SEQ_LEN`, and `HORIZON` must stay identical to the Colab notebook's — since a
`.pth` file is just a `state_dict`, and PyTorch loads it by matching parameter
shapes, not by re-reading any architecture description. If you change one,
change the other.

## Getting a real 72-hour window

The LSTM needs 72 consecutive hourly readings, and a single slider drag or a
single real sensor reading only gives you one. Three ways to fill the buffer
without waiting three real days:

1. **"Play 3 simulated days"** (3D sensor simulation panel) — animates the
   scene through 72 fictional hours (a bounded random walk for cloud cover
   and wind speed, same clear-sky irradiance formula as everywhere else) and
   bulk-loads them via `POST /api/sensors/ingest-batch`. Fastest way to see
   `source=ml` run on a genuine window, but the data is still simulated.
2. **"Backfill from real weather"** (same panel, or `POST
   /api/sensors/backfill-from-weather?lat=..&lon=..`) — fetches the *actual*
   last 3 days of hourly weather for a location from Open-Meteo
   (`forecast.py`'s `fetch_recent_history`, using its `past_days` parameter)
   and loads it as history. This is real data, not simulated — the closest
   thing to "what would a freshly-installed sensor already have seen."
3. **CSV upload** (`POST /api/sensors/ingest-csv`, or the upload control in
   the same panel) — for a real site's own logs. Required columns: `hour`,
   `wind_speed_kmh`, `solar_irradiance_wm2` (optional: `cloud_cover_pct`,
   `temperature_c`). See `backend/sample_history_72h.csv` for the exact
   format, or download it from the running frontend.

A normal sensor gateway would just call `POST /api/sensors/ingest` once per
hour and the buffer would fill itself over 3 real days — these three options
exist only to make that wait skippable for development and demos.

Dragging a slider in the 3D twin **updates the current hour in place**
(`update_latest=true` on `/api/sensors/ingest`) rather than adding a new hour
per drag — otherwise fiddling with a slider would silently corrupt the
72-hour timeline with dozens of readings from the same instant.

## Extending this for a full submission

- **Better load forecasting:** train a small model (Prophet, XGBoost, or an
  LSTM) on historical smart-meter data instead of the synthetic curve.
- **Better optimization:** replace the greedy dispatch with a linear program
  that minimizes cost/grid draw over the full 48-hour window at once, and can
  account for time-of-use electricity pricing.
- **Historical accuracy tracking:** store each day's forecast vs. actual
  generation to show a forecast-accuracy metric — this is a strong thing to
  demo to judges.
- **Multi-site / microgrid federation:** extend the API to handle several
  buildings sharing one battery and grid connection.
- **Deployment:** containerize backend with Docker, deploy frontend as a static
  build (`npm run build`) behind the FastAPI app or on any static host.
#   E n e r g y - m i r c o - g r i d  
 