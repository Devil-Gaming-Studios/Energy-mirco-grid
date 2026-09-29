"""
GridSense backend — FastAPI service that turns environmental data into solar
and wind power estimates, runs a battery dispatch optimizer, and returns
everything the frontend needs in one call.

Three independent data sources feed the same /api/forecast contract:
  - source=weather  -> live NWP forecast from Open-Meteo (real-world data)
  - source=sensors  -> the last reading posted to POST /api/sensors/ingest,
                        projected forward with a persistence model
  - source=ml       -> a rolling window of ingested sensor readings, projected
                        forward by a trained LSTM (see backend/ml_model.py and
                        the companion Colab notebook in colab/)

POST /api/sensors/ingest is a genuine sensor-ingestion endpoint: it validates
and stores a reading exactly as it would from a physical on-site weather
station / anemometer / pyranometer. The 3D digital-twin frontend calls this
same endpoint to simulate that hardware, but the endpoint itself has no idea
the data isn't real — that's what makes it a drop-in replacement for actual
sensors later.

Run:
    pip install -r requirements.txt
    uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from typing import List
import csv
import io
import shutil
import os

import sensors
import ml_model
from forecast import (
    store_browser_weather,
    fetch_recent_history,
    fetch_weather,
    solar_power_kw,
    wind_power_kw,
    synthetic_load,
    optimize_battery,
    generate_recommendations,
)


class SensorReading(BaseModel):
    """
    Payload shape a real on-site sensor gateway would POST. The 3D twin
    frontend sends exactly this shape too, computed from its scene state.
    """
    cloud_cover_pct: float = Field(..., ge=0, le=100)
    wind_speed_kmh: float = Field(..., ge=0)
    solar_irradiance_wm2: float = Field(..., ge=0)
    temperature_c: float = 20.0
    sim_hour: float = Field(..., ge=0, lt=24, description="Local hour-of-day the reading was taken at")
    source: str = Field("3d-twin", description="Where the reading came from, e.g. 'physical-sensor' or '3d-twin'")

app = FastAPI(title="GridSense API", version="1.0.0")

# Allow the local React dev server (and any origin, for simplicity in a hackathon demo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SensorBatch(BaseModel):
    """Many hourly readings at once, oldest first (e.g. a site's logs, or the 3D twin's 3-day run)."""
    readings: List[SensorReading] = Field(..., min_length=1, max_length=500)
    replace_history: bool = True


class BrowserWeather(BaseModel):
    """Open-Meteo JSON fetched client-side (past_days=3, forecast_days=2, timezone=auto)."""
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    data: dict


@app.post("/api/weather/ingest")
def ingest_browser_weather(body: BrowserWeather):
    """Cache weather the browser fetched from Open-Meteo, so this server (often on a
    shared, rate-limited IP such as Render) doesn't have to. /api/forecast and
    /api/sensors/backfill-from-weather then read it from the cache."""
    try:
        n = store_browser_weather(body.lat, body.lon, body.data)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"Invalid weather payload: {exc}")
    return {"status": "stored", "hours": n}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/sensors/ingest")
def ingest_sensor_reading(reading: SensorReading,
                           update_latest: bool = Query(False, description="Overwrite the most recent history entry instead of appending a new hour")):
    """
    Real sensor-ingestion endpoint. A physical weather station on site would
    call this exact endpoint with this exact payload shape (one reading per
    hour, appended to history); the 3D twin frontend calls it too, standing in
    for that hardware during development and demos.
    """
    stored = sensors.ingest_reading(reading.model_dump(), replace_latest=update_latest)
    return {"status": "stored", "reading": stored}


@app.post("/api/sensors/ingest-batch")
def ingest_sensor_batch(batch: SensorBatch):
    """
    Bulk-load hourly readings (oldest first). This is how a real site would
    backfill its logs so the LSTM has a genuine 72-hour input window from day one.
    """
    stored = sensors.ingest_batch([r.model_dump() for r in batch.readings],
                                  replace_history=batch.replace_history)
    return {"status": "stored", "count": len(stored), "history_size": len(sensors.get_history())}


@app.post("/api/sensors/ingest-csv")
async def ingest_sensor_csv(file: UploadFile = File(...), replace_history: bool = Query(True)):
    """
    Upload hourly history as CSV (oldest row first). Required columns:
    hour (0-23), wind_speed_kmh, solar_irradiance_wm2.
    Optional: cloud_cover_pct (default 0), temperature_c (default 20).
    See backend/sample_history_72h.csv for the exact format.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Expected a .csv file")
    try:
        text = (await file.read()).decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="CSV must be UTF-8 text")

    reader = csv.DictReader(io.StringIO(text))
    required = {"hour", "wind_speed_kmh", "solar_irradiance_wm2"}
    if not reader.fieldnames or not required.issubset({c.strip() for c in reader.fieldnames}):
        raise HTTPException(status_code=400,
                            detail=f"CSV needs columns: {sorted(required)} (optional: cloud_cover_pct, temperature_c)")

    parsed = []
    for line_no, row in enumerate(reader, start=2):
        row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items() if k}
        try:
            reading = SensorReading(
                sim_hour=float(row["hour"]),
                wind_speed_kmh=float(row["wind_speed_kmh"]),
                solar_irradiance_wm2=float(row["solar_irradiance_wm2"]),
                cloud_cover_pct=float(row.get("cloud_cover_pct") or 0),
                temperature_c=float(row.get("temperature_c") or 20),
                source="csv-upload",
            )
        except (ValueError, ValidationError) as exc:
            raise HTTPException(status_code=422, detail=f"Row {line_no}: {exc}")
        parsed.append(reading.model_dump())

    if not parsed:
        raise HTTPException(status_code=400, detail="CSV has no data rows")
    parsed = parsed[-500:]
    sensors.ingest_batch(parsed, replace_history=replace_history)
    return {"status": "stored", "count": len(parsed), "history_size": len(sensors.get_history())}


@app.post("/api/sensors/backfill-from-weather")
def backfill_from_weather(
    lat: float = Query(23.2599, description="Latitude, default Bhopal, India"),
    lon: float = Query(77.4126, description="Longitude, default Bhopal, India"),
    hours: int = Query(72, ge=1, le=72),
):
    """
    Fill the sensor history with the last `hours` of REAL recent hourly weather
    for a location (Open-Meteo), replacing whatever was buffered. After this,
    source=ml runs the LSTM on a genuine 72-hour window instead of padded history.
    """
    try:
        w = fetch_recent_history(lat, lon, hours)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Weather fetch failed: {exc}")

    from datetime import datetime
    readings = []
    for i, ts in enumerate(w["timestamps"]):
        dt = datetime.fromisoformat(ts)
        readings.append({
            "cloud_cover_pct": w["cloudcover"][i],
            "wind_speed_kmh": w["windspeed"][i],
            "solar_irradiance_wm2": w["radiation"][i],
            "temperature_c": w["temperature"][i],
            "sim_hour": dt.hour + dt.minute / 60.0,
            "source": "open-meteo-backfill",
        })
    sensors.ingest_batch(readings, replace_history=True)
    return {
        "status": "stored",
        "count": len(readings),
        "window_start": w["timestamps"][0],
        "window_end": w["timestamps"][-1],
        "timezone": w["timezone"],
        "stale": w.get("stale", False),
        "data_source": w.get("data_source", "open-meteo"),
    }


@app.get("/api/sensors/latest")
def latest_sensor_reading():
    """What a real sensor-monitoring panel would poll to show current conditions."""
    reading = sensors.get_latest_reading()
    if reading is None:
        raise HTTPException(status_code=404, detail="No sensor reading has been received yet")
    return reading


@app.get("/api/sensors/history")
def sensor_history(limit: int = Query(72, ge=1, le=500)):
    """
    Raw ingested reading history (most recent last). Useful for checking how
    many real readings have accumulated toward the LSTM's required window
    (see ml_model.SEQ_LEN) versus how much of a source=ml forecast would
    currently rely on padded/extrapolated history.
    """
    history = sensors.get_history(limit=limit)
    return {"count": len(history), "readings": history}


@app.get("/api/model/status")
def model_status():
    return {"loaded": ml_model.is_loaded(), "weights_path": ml_model.WEIGHTS_PATH}


@app.post("/api/model/upload-weights")
async def upload_model_weights(file: UploadFile = File(...)):
    """
    Upload a .pth file trained in the companion Colab notebook. Saved to
    backend/weights/solar_wind_forecaster.pth and loaded immediately —
    no server restart needed.
    """
    if not file.filename.endswith(".pth"):
        raise HTTPException(status_code=400, detail="Expected a .pth file")

    os.makedirs(os.path.dirname(ml_model.WEIGHTS_PATH), exist_ok=True)
    with open(ml_model.WEIGHTS_PATH, "wb") as out:
        shutil.copyfileobj(file.file, out)

    try:
        ok = ml_model.load_model()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Weights file didn't match the expected model architecture: {exc}")

    if not ok:
        raise HTTPException(status_code=500, detail="Weights saved but failed to load")

    return {"status": "loaded", "weights_path": ml_model.WEIGHTS_PATH}


@app.get("/api/forecast")
def get_forecast(
    source: str = Query("weather", pattern="^(weather|sensors|ml)$",
                         description="'weather' = live Open-Meteo forecast, 'sensors' = persistence projection from latest reading, 'ml' = trained model projection from latest reading"),
    lat: float = Query(23.2599, description="Latitude, default Bhopal, India (used when source=weather)"),
    lon: float = Query(77.4126, description="Longitude, default Bhopal, India (used when source=weather)"),
    solar_capacity_kw: float = Query(50.0, gt=0),
    wind_capacity_kw: float = Query(30.0, ge=0),
    battery_kwh: float = Query(100.0, gt=0),
    load_scale_kw: float = Query(50.0, gt=0),
):
    """
    Returns a 48-hour forecast: solar & wind power, a load curve, battery
    dispatch simulation, and plain-language scheduling recommendations.
    The environmental data behind it comes from either live weather
    (source=weather) or the latest ingested sensor/digital-twin reading
    (source=sensors) — everything downstream is identical either way.
    """
    if source == "sensors":
        weather = sensors.project_from_sensors(hours=48)
        if weather is None:
            raise HTTPException(
                status_code=404,
                detail="No sensor reading yet. POST one to /api/sensors/ingest first "
                       "(e.g. by moving a slider in the 3D environment).",
            )
    elif source == "ml":
        history_info = sensors.get_history_for_ml(seq_len=ml_model.SEQ_LEN)
        if history_info is None:
            raise HTTPException(
                status_code=404,
                detail="No sensor reading yet. POST one to /api/sensors/ingest first.",
            )
        if not ml_model.is_loaded():
            raise HTTPException(
                status_code=503,
                detail="No trained model weights loaded. Train one in the Colab notebook "
                       "and POST it to /api/model/upload-weights, or drop the .pth file at "
                       f"{ml_model.WEIGHTS_PATH}",
            )
        weather = ml_model.predict_ml(history_info["sequence"])
        weather["meta"]["real_readings_used"] = history_info["real_count"]
        weather["meta"]["padded_readings"] = history_info["padded_count"]
    else:
        try:
            weather = fetch_weather(lat, lon, hours=48)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Weather fetch failed: {exc}")

    solar = solar_power_kw(weather["radiation"], solar_capacity_kw)
    wind = wind_power_kw(weather["windspeed"], wind_capacity_kw)
    load = synthetic_load(weather["timestamps"], load_scale_kw)

    battery = optimize_battery(solar, wind, load, battery_kwh)
    recs = generate_recommendations(
        weather["timestamps"], solar, wind, load, battery["battery_soc"], battery_kwh
    )

    return {
        "source": source,
        "data_source": weather.get("data_source"),
        "timestamps": weather["timestamps"],
        "timezone": weather["timezone"],
        "sensor_meta": weather.get("meta"),
        "solar_kw": solar,
        "wind_kw": wind,
        "load_kw": load,
        "cloudcover_pct": weather["cloudcover"],
        "windspeed_kmh": weather["windspeed"],
        "battery_soc_kwh": battery["battery_soc"],
        "grid_draw_kw": battery["grid_draw"],
        "surplus_curtailed_kw": battery["surplus_curtailed"],
        "stats": {
            "total_grid_draw_kwh": battery["total_grid_draw_kwh"],
            "baseline_grid_draw_kwh": battery["baseline_grid_draw_kwh"],
            "grid_draw_reduction_pct": battery["grid_draw_reduction_pct"],
        },
        "recommendations": recs,
    }
