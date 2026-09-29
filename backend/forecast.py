"""
Core forecasting + optimization logic.

Real weather data comes from Open-Meteo (https://open-meteo.com), a free
weather API that requires no API key. We pull hourly shortwave solar
radiation and wind speed for the next 48 hours and convert them into
estimated power output using simple, physically-motivated models.

Load is currently a synthetic "typical daily campus/building" curve —
swap `synthetic_load()` for a call to your real smart-meter / utility
data source when you have one.
"""

import math
from datetime import datetime, timedelta
import requests

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_weather(lat: float, lon: float, hours: int = 48) -> dict:
    """Fetch hourly solar radiation, wind speed, cloud cover and temperature."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "shortwave_radiation,cloudcover,windspeed_10m,temperature_2m",
        "forecast_days": 2,
        "timezone": "auto",
    }
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    hourly = data["hourly"]
    return {
        "timestamps": hourly["time"][:hours],
        "radiation": hourly["shortwave_radiation"][:hours],   # W/m^2
        "cloudcover": hourly["cloudcover"][:hours],            # %
        "windspeed": hourly["windspeed_10m"][:hours],          # km/h
        "temperature": hourly["temperature_2m"][:hours],       # deg C
        "timezone": data.get("timezone", "UTC"),
    }


def select_recent_window(hourly: dict, utc_offset_seconds: int, now_utc: datetime, hours: int = 72) -> dict:
    """
    From an Open-Meteo hourly block covering past days + today, pick the
    `hours` most recent hours up to (and including) the current local hour.
    Split out from the HTTP call so it can be tested without network access.
    """
    now_local = now_utc + timedelta(seconds=utc_offset_seconds)
    times = [datetime.fromisoformat(t) for t in hourly["time"]]
    eligible = [i for i, t in enumerate(times) if t <= now_local]
    if not eligible:
        raise ValueError("Weather API returned no past hours")
    last = eligible[-1]
    start = max(0, last - hours + 1)

    def col(name, default=0.0):
        vals = hourly[name][start:last + 1]
        out, prev = [], default
        for v in vals:
            prev = prev if v is None else v  # carry the last good value across API gaps
            out.append(float(prev))
        return out

    return {
        "timestamps": hourly["time"][start:last + 1],
        "radiation": col("shortwave_radiation"),
        "cloudcover": col("cloudcover"),
        "windspeed": col("windspeed_10m"),
        "temperature": col("temperature_2m", 20.0),
    }


def fetch_recent_history(lat: float, lon: float, hours: int = 72) -> dict:
    """
    Fetch the last `hours` (max 72) of hourly weather for a location: real
    recent conditions used to fill the LSTM's input window. Open-Meteo's
    `past_days` parameter returns the previous days alongside today.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "shortwave_radiation,cloudcover,windspeed_10m,temperature_2m",
        "past_days": 3,
        "forecast_days": 1,
        "timezone": "auto",
    }
    resp = requests.get(OPEN_METEO_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    window = select_recent_window(
        data["hourly"], data.get("utc_offset_seconds", 0), datetime.utcnow(), hours
    )
    window["timezone"] = data.get("timezone", "UTC")
    return window


def solar_power_kw(radiation_wm2: list[float], capacity_kw: float,
                    panel_efficiency: float = 0.18, performance_ratio: float = 0.8) -> list[float]:
    """
    Estimate solar PV output from shortwave radiation.
    Standard test conditions assume 1000 W/m^2 = rated capacity output.
    """
    out = []
    for r in radiation_wm2:
        frac_of_rated = max(0.0, r) / 1000.0
        power = capacity_kw * frac_of_rated * performance_ratio
        out.append(round(min(power, capacity_kw), 3))
    return out


def wind_power_kw(windspeed_kmh: list[float], capacity_kw: float,
                   cut_in=10.0, rated=45.0, cut_out=90.0) -> list[float]:
    """
    Simple wind turbine power curve (speeds in km/h):
    - below cut_in: 0 output
    - between cut_in and rated: cubic ramp-up
    - between rated and cut_out: full rated output
    - above cut_out: 0 (turbine shuts down for safety)
    """
    out = []
    for v in windspeed_kmh:
        if v < cut_in or v > cut_out:
            power = 0.0
        elif v < rated:
            frac = ((v - cut_in) / (rated - cut_in)) ** 3
            power = capacity_kw * frac
        else:
            power = capacity_kw
        out.append(round(power, 3))
    return out


def synthetic_load(timestamps: list[str], scale_kw: float = 50.0) -> list[float]:
    """Typical daily demand curve: morning ramp + evening peak, repeated per day."""
    out = []
    for ts in timestamps:
        hour = datetime.fromisoformat(ts).hour
        morning = 0.5 * math.exp(-((hour - 8) ** 2) / 8)
        evening = 0.8 * math.exp(-((hour - 19) ** 2) / 6)
        base = 0.3 + morning + evening
        out.append(round(base * scale_kw, 3))
    return out


def optimize_battery(solar: list[float], wind: list[float], load: list[float],
                      battery_kwh: float, charge_eff=0.95, discharge_eff=0.95) -> dict:
    """
    Simple greedy hourly dispatch:
    - if generation > load: charge battery with surplus (up to capacity)
    - if generation < load: discharge battery to cover deficit (down to empty)
    - remaining shortfall is drawn from the grid
    Returns per-hour battery state of charge, grid draw, and total stats.
    """
    soc = battery_kwh * 0.5  # start half-charged
    soc_series, grid_draw, surplus_curtailed = [], [], []
    baseline_grid_draw = 0.0

    for s, w, l in zip(solar, wind, load):
        gen = s + w
        baseline_grid_draw += max(0.0, l - gen)
        net = gen - l

        if net >= 0:
            charge_room = battery_kwh - soc
            charge_amount = min(net * charge_eff, charge_room)
            soc += charge_amount
            leftover = net - (charge_amount / charge_eff if charge_eff else 0)
            surplus_curtailed.append(round(max(0.0, leftover), 3))
            grid_draw.append(0.0)
        else:
            deficit = -net
            discharge_amount = min(deficit / discharge_eff, soc)
            soc -= discharge_amount
            remaining = deficit - (discharge_amount * discharge_eff)
            grid_draw.append(round(max(0.0, remaining), 3))
            surplus_curtailed.append(0.0)

        soc_series.append(round(soc, 3))

    total_grid_draw = round(sum(grid_draw), 3)
    grid_saved_pct = 0.0
    if baseline_grid_draw > 0:
        grid_saved_pct = round((1 - total_grid_draw / baseline_grid_draw) * 100, 1)

    return {
        "battery_soc": soc_series,
        "grid_draw": grid_draw,
        "surplus_curtailed": surplus_curtailed,
        "total_grid_draw_kwh": total_grid_draw,
        "baseline_grid_draw_kwh": round(baseline_grid_draw, 3),
        "grid_draw_reduction_pct": grid_saved_pct,
    }


def generate_recommendations(timestamps, solar, wind, load, soc_series, battery_kwh) -> list[dict]:
    """Turn the numbers into a handful of plain-language scheduling suggestions."""
    recs = []

    # Best surplus window in the next 24h -> good time for flexible/shiftable loads
    best_idx, best_surplus = None, -1e9
    for i in range(min(24, len(timestamps))):
        surplus = (solar[i] + wind[i]) - load[i]
        if surplus > best_surplus:
            best_surplus, best_idx = surplus, i
    if best_idx is not None and best_surplus > 0:
        t = datetime.fromisoformat(timestamps[best_idx]).strftime("%H:%M")
        recs.append({
            "title": f"Shift flexible loads to {t}",
            "detail": f"Forecast surplus of about {round(best_surplus, 1)} kW from solar+wind — "
                      f"a good window for EV charging, water heating, or batch processing.",
            "severity": "info",
        })

    # Look for an evening/overnight deficit hour where battery is low
    for i in range(len(timestamps)):
        gen = solar[i] + wind[i]
        deficit = load[i] - gen
        if deficit > 0.3 * (max(load) if load else 1) and soc_series[i] < 0.15 * battery_kwh:
            t = datetime.fromisoformat(timestamps[i]).strftime("%H:%M")
            recs.append({
                "title": f"Pre-charge battery before {t}",
                "detail": f"A demand peak of about {round(deficit,1)} kW is forecast with low battery "
                          f"reserve at that time — charge earlier in the day to avoid a grid draw spike.",
                "severity": "warn",
            })
            break

    # Peak solar hour
    if solar:
        peak_i = solar.index(max(solar[:24])) if len(solar) >= 24 else solar.index(max(solar))
        t = datetime.fromisoformat(timestamps[peak_i]).strftime("%H:%M")
        recs.append({
            "title": f"Peak solar output near {t}",
            "detail": "Good time to schedule maintenance or high-draw processes directly on solar "
                      "rather than drawing from the battery.",
            "severity": "info",
        })

    if not recs:
        recs.append({
            "title": "No major deficit windows detected",
            "detail": "Forecast generation and battery capacity comfortably cover predicted load.",
            "severity": "info",
        })

    return recs
