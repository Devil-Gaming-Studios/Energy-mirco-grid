"""
Sensor ingestion layer.

This module is deliberately source-agnostic: it exposes an endpoint contract
that a real IoT gateway (a physical weather station + anemometer + pyranometer
on site) would call, and it does not care whether the numbers it receives came
from real hardware or from the 3D digital-twin simulator in the frontend. That
is the point — swap the data source without touching the ingestion contract,
the storage, or the forecasting logic downstream of it.

In a production deployment `_latest_reading` would be a row in a time-series
database (e.g. TimescaleDB, InfluxDB) keyed by site_id; here it's an in-memory
single-site store, which is enough for a hackathon demo / single microgrid site.

Forecasting from a live instantaneous reading uses a *persistence* model: a
standard short-term forecasting technique that assumes near-term conditions
resemble the most recently observed one. Concretely:
  1. Compare the current measured solar irradiance to a clear-sky model for
     the current time of day -> infer a "cloud attenuation factor".
  2. Project that attenuation factor forward across a clear-sky curve for the
     next N hours, so the diurnal solar shape is still physically correct
     even though we only have one live reading.
  3. Hold wind speed at its last measured value (with small variation), which
     is the standard wind persistence baseline.
Two forecasting methods read from this module:
  1. project_from_sensors() — persistence projection from the single latest
     reading (no history needed, works from the very first reading).
  2. get_history_for_ml() — a rolling window of past readings for the LSTM
     model in ml_model.py, which needs a sequence rather than one point.
"""

import math
from collections import deque
from datetime import datetime, timedelta

_latest_reading = None  # single-site in-memory store; replace with a DB for multi-site

# Rolling history of ingested readings, used by the LSTM ML model, which needs
# a sequence of past hours rather than a single instantaneous value. In a real
# deployment this would be a time-series DB query ("last N hours for site X");
# here it's an in-memory ring buffer, which is enough for a single-site demo.
HISTORY_MAXLEN = 500
_history = deque(maxlen=HISTORY_MAXLEN)


def _build_reading(payload: dict) -> dict:
    return {
        "cloud_cover_pct": float(payload["cloud_cover_pct"]),
        "wind_speed_kmh": float(payload["wind_speed_kmh"]),
        "solar_irradiance_wm2": float(payload["solar_irradiance_wm2"]),
        "temperature_c": float(payload.get("temperature_c", 20.0)),
        "sim_hour": float(payload.get("sim_hour", datetime.utcnow().hour)),
        "source": payload.get("source", "unknown"),
        "received_at": datetime.utcnow().isoformat(),
    }


def ingest_reading(payload: dict, replace_latest: bool = False) -> dict:
    """
    Store a sensor reading. Called identically by real hardware or the 3D sim.

    A real station posts one reading per hour, so the default is to append a
    new hour to the history. `replace_latest=True` instead overwrites the most
    recent entry ("what if the current hour looked like this?") — used by the
    3D twin's sliders so dragging them doesn't keep shifting the 72-hour window.
    """
    reading = _build_reading(payload)
    global _latest_reading
    _latest_reading = reading
    if replace_latest and _history:
        _history[-1] = reading
    else:
        _history.append(reading)
    return reading


def ingest_batch(readings: list, replace_history: bool = True) -> list:
    """
    Store many hourly readings at once, oldest first — for backfilling a
    site's logs, a CSV upload, a weather-API backfill, or the 3D twin's
    "play 3 days" run. With replace_history=True the buffer is cleared first,
    so the LSTM's 72-hour window is exactly this timeline.
    """
    global _latest_reading
    if replace_history:
        _history.clear()
    stored = []
    for payload in readings:
        r = _build_reading(payload)
        _history.append(r)
        stored.append(r)
    if stored:
        _latest_reading = stored[-1]
    return stored


def get_latest_reading():
    return _latest_reading


def get_history(limit: int = None):
    """Raw ingested history, most recent last. Used by /api/sensors/history."""
    items = list(_history)
    return items[-limit:] if limit else items


def clear_sky_irradiance(hour_of_day: float) -> float:
    """Simple bell-curve clear-sky model peaking at ~1000 W/m^2 at solar noon."""
    x = max(0.0, math.sin(((hour_of_day - 6.0) / 12.0) * math.pi))
    return 1000.0 * (x ** 1.1)


def get_history_for_ml(seq_len: int):
    """
    Return exactly `seq_len` hourly feature points ending at the most recent
    ingested reading, for the LSTM model's input window.

    If fewer than `seq_len` real readings have been ingested yet (typical in a
    fresh demo — the 3D twin sends one reading per slider move, not a full
    week of history), the window is padded backward using the same
    persistence-style extrapolation as project_from_sensors(): a clear-sky
    curve attenuated by the earliest real reading's inferred cloud
    attenuation, with wind held near its value. Padded points are clearly
    flagged in the returned metadata — they are not real measurements, and
    the model's accuracy for a mostly-padded window should be treated
    accordingly.

    Returns None if no reading has ever been ingested.
    """
    history = list(_history)
    if not history:
        return None

    real = history[-seq_len:]
    real_count = len(real)
    padded_count = seq_len - real_count

    if padded_count <= 0:
        sequence = [
            {"hour": r["sim_hour"] % 24, "irradiance": r["solar_irradiance_wm2"],
             "wind_kmh": r["wind_speed_kmh"], "temperature_c": r["temperature_c"]}
            for r in real
        ]
        return {"sequence": sequence, "real_count": real_count, "padded_count": 0}

    # Pad backward from the earliest real reading using persistence/clear-sky extrapolation
    earliest = real[0]
    clear_now = clear_sky_irradiance(earliest["sim_hour"] % 24)
    attenuation = earliest["solar_irradiance_wm2"] / clear_now if clear_now > 1.0 else 1.0
    attenuation = max(0.0, min(1.0, attenuation))

    padding = []
    for i in range(padded_count, 0, -1):
        hod = (earliest["sim_hour"] - i) % 24
        rad = clear_sky_irradiance(hod) * attenuation
        wind = max(0.0, earliest["wind_speed_kmh"] + math.sin(i * 0.7) * 3.0)
        padding.append({"hour": hod, "irradiance": rad, "wind_kmh": wind,
                         "temperature_c": earliest["temperature_c"]})

    real_seq = [
        {"hour": r["sim_hour"] % 24, "irradiance": r["solar_irradiance_wm2"],
         "wind_kmh": r["wind_speed_kmh"], "temperature_c": r["temperature_c"]}
        for r in real
    ]
    return {"sequence": padding + real_seq, "real_count": real_count, "padded_count": padded_count}


def project_from_sensors(hours: int = 48):
    """
    Build a 48-hour forward projection from the single latest live sensor
    reading, in the same shape fetch_weather() returns so the rest of the
    pipeline (solar_power_kw, wind_power_kw, optimize_battery, ...) doesn't
    need to know or care where the data came from.
    """
    reading = get_latest_reading()
    if reading is None:
        return None

    sim_hour = reading["sim_hour"]
    clear_now = clear_sky_irradiance(sim_hour % 24)
    attenuation = reading["solar_irradiance_wm2"] / clear_now if clear_now > 1.0 else 1.0
    attenuation = max(0.0, min(1.0, attenuation))

    timestamps, radiation, windspeed, cloudcover, temperature = [], [], [], [], []
    base_time = datetime.utcnow()

    for h in range(hours):
        hod = (sim_hour + h) % 24
        rad = clear_sky_irradiance(hod) * attenuation
        wind = max(0.0, reading["wind_speed_kmh"] + math.sin(h * 0.7) * 3.0)

        timestamps.append((base_time + timedelta(hours=h)).isoformat())
        radiation.append(round(rad, 1))
        windspeed.append(round(wind, 1))
        cloudcover.append(reading["cloud_cover_pct"])
        temperature.append(reading["temperature_c"])

    return {
        "timestamps": timestamps,
        "radiation": radiation,
        "windspeed": windspeed,
        "cloudcover": cloudcover,
        "temperature": temperature,
        "timezone": "site-local (from sensor)",
        "meta": {
            "inferred_cloud_attenuation": round(attenuation, 3),
            "sim_hour_at_reading": sim_hour,
            "reading_source": reading["source"],
            "reading_received_at": reading["received_at"],
        },
    }
