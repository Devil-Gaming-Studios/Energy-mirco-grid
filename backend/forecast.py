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
import json
import math
import os
import threading
import time
from datetime import datetime, timedelta

import requests

OPEN_METEO_FREE_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_PAID_URL = "https://customer-api.open-meteo.com/v1/forecast"


def _open_meteo_endpoint() -> tuple[str, dict]:
    """Paid/commercial key (env OPEN_METEO_API_KEY) -> dedicated customer host; else free host."""
    key = os.environ.get("OPEN_METEO_API_KEY", "").strip()
    if key:
        return OPEN_METEO_PAID_URL, {"apikey": key}
    return OPEN_METEO_FREE_URL, {}


HOURLY_VARS = "shortwave_radiation,cloudcover,windspeed_10m,temperature_2m"

CACHE_TTL_S = 3 * 3600          # serve cached data for 3h without touching the API
                                # (NWP hourly data barely changes; saves quota)
FALLBACK_TTL_S = 600           # fallback data is re-tried against Open-Meteo after 10 min
MAX_RETRY_WAIT_S = 10           # never block a request longer than this per retry
CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".weather_cache.json")

_cache: dict = {}               # key -> (fetched_at_epoch, raw_open_meteo_json)
_cache_lock = threading.Lock()
_cache_loaded = False


class WeatherUnavailable(RuntimeError):
    """Open-Meteo is rate-limiting / down AND there is no cached data to fall back on."""


def _load_disk_cache():
    global _cache_loaded
    if _cache_loaded:
        return
    _cache_loaded = True
    try:
        with open(CACHE_FILE) as f:
            for k, v in json.load(f).items():
                _cache[k] = (v["t"], v["data"])
    except (OSError, ValueError, KeyError):
        pass


def _save_disk_cache():
    try:
        with open(CACHE_FILE, "w") as f:
            json.dump({k: {"t": t, "data": d} for k, (t, d) in _cache.items()}, f)
    except OSError:
        pass  # cache persistence is best-effort


MET_NO_URL = "https://api.met.no/weatherapi/locationforecast/2.0/compact"


def _utc_offset_hours(lat: float, lon: float) -> float:
    """Best-effort timezone offset when Open-Meteo (which reports it) is unavailable.
    Override with env WEATHER_UTC_OFFSET_HOURS for exact control."""
    env = os.environ.get("WEATHER_UTC_OFFSET_HOURS")
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    if 6 <= lat <= 36 and 68 <= lon <= 98:      # India / Sri Lanka: UTC+5:30
        return 5.5
    return round(lon / 15.0 * 2) / 2


def _clear_sky_ghi(lat: float, lon: float, utc_dt: datetime) -> float:
    """Approximate clear-sky global horizontal irradiance (W/m^2) from solar geometry."""
    doy = utc_dt.timetuple().tm_yday
    decl = math.radians(23.44) * math.sin(2 * math.pi * (284 + doy) / 365)
    solar_time = utc_dt.hour + utc_dt.minute / 60 + lon / 15.0
    ha = math.radians((solar_time - 12) * 15)
    la = math.radians(lat)
    sin_elev = math.sin(la) * math.sin(decl) + math.cos(la) * math.cos(decl) * math.cos(ha)
    return 1050.0 * sin_elev ** 1.15 if sin_elev > 0 else 0.0


def _fallback_raw(lat: float, lon: float) -> dict:
    """
    Last-resort weather when Open-Meteo refuses us, returned in Open-Meteo's own
    JSON shape so nothing downstream changes.
      - Forecast hours: real MET Norway (api.met.no) cloud / wind / temperature.
      - Radiation: NOT provided by MET Norway, so it is estimated from solar
        geometry x cloud attenuation (Kasten-Czeplak).
      - Hours before 'now' (needed for the 72h backfill): MET Norway has no
        history, so the earliest forecast values are held constant. These are
        ESTIMATES and are labelled as such via `_source`.
    """
    offset = _utc_offset_hours(lat, lon)
    steps, source = [], "estimated"
    try:
        resp = requests.get(
            MET_NO_URL, params={"lat": round(lat, 4), "lon": round(lon, 4)}, timeout=15,
            headers={"User-Agent": "EnergyMicroGrid/1.0 github.com/energy-microgrid"},
        )
        resp.raise_for_status()
        for ts in resp.json()["properties"]["timeseries"]:
            d = ts["data"]["instant"]["details"]
            steps.append((
                datetime.fromisoformat(ts["time"].replace("Z", "+00:00")).replace(tzinfo=None),
                d.get("cloud_area_fraction", 25.0),
                d.get("wind_speed", 3.3) * 3.6,                 # m/s -> km/h
                d.get("air_temperature", 25.0),
            ))
        if steps:
            source = "met.no+estimated-history"
    except (requests.RequestException, ValueError, KeyError):
        pass
    if not steps:                                               # no network source at all
        steps = [(datetime(2000, 1, 1), 25.0, 12.0, 25.0)]

    now_local = datetime.utcnow() + timedelta(hours=offset)
    start = (now_local - timedelta(days=3)).replace(hour=0, minute=0, second=0, microsecond=0)
    out = {"time": [], "shortwave_radiation": [], "cloudcover": [],
           "windspeed_10m": [], "temperature_2m": []}
    for i in range(120):                                        # 3 past days + 2 forecast days
        t_local = start + timedelta(hours=i)
        t_utc = t_local - timedelta(hours=offset)
        pick = steps[0]
        for s in steps:                                         # latest step at or before t_utc
            if s[0] <= t_utc:
                pick = s
            else:
                break
        _, cloud, wind, temp = pick
        c = min(max(cloud / 100.0, 0.0), 1.0)
        ghi = _clear_sky_ghi(lat, lon, t_utc) * (1 - 0.75 * c ** 3.4)
        out["time"].append(t_local.strftime("%Y-%m-%dT%H:00"))
        out["shortwave_radiation"].append(round(ghi, 1))
        out["cloudcover"].append(cloud)
        out["windspeed_10m"].append(round(wind, 1))
        out["temperature_2m"].append(temp)
    return {"hourly": out, "utc_offset_seconds": int(offset * 3600),
            "timezone": f"UTC{offset:+g}", "_source": source}


def _fetch_raw(lat: float, lon: float) -> tuple[dict, bool]:
    """
    One Open-Meteo request per location, shared by BOTH the forecast and the
    72h backfill (3 past days + 2 forecast days covers everything), cached so
    repeated UI clicks / dev-server reloads don't burn API quota.

    Returns (raw_json, is_stale). On 429 / network errors it honours
    Retry-After (bounded), retries, and finally falls back to ANY cached copy
    (even expired) before giving up.
    """
    key = f"{lat:.2f},{lon:.2f}"          # ~1 km grid; NWP resolution is coarser anyway
    with _cache_lock:
        _load_disk_cache()
        cached = _cache.get(key)
    if cached:
        ttl = FALLBACK_TTL_S if cached[1].get("_source") else CACHE_TTL_S
        if time.time() - cached[0] < ttl:
            return cached[1], False

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": HOURLY_VARS,
        "past_days": 3,
        "forecast_days": 2,
        "timezone": "auto",
    }
    url, extra = _open_meteo_endpoint()
    params.update(extra)
    last_err = "unknown error"
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=15,
                                headers={"User-Agent": "EnergyMicroGrid/1.0"})
            if resp.status_code == 429:
                try:
                    last_err = "Open-Meteo rate limit reached: " + resp.json().get("reason", "")
                except ValueError:
                    last_err = "Open-Meteo rate limit reached"
                try:
                    wait = float(resp.headers.get("Retry-After", 2 ** (attempt + 1)))
                except ValueError:
                    wait = 2 ** (attempt + 1)
                if attempt < 2:
                    time.sleep(min(wait, MAX_RETRY_WAIT_S))
                continue
            resp.raise_for_status()
            data = resp.json()
            _ = data["hourly"]["time"]     # sanity check
            with _cache_lock:
                _cache[key] = (time.time(), data)
                _save_disk_cache()
            return data, False
        except (requests.RequestException, ValueError, KeyError) as exc:
            last_err = str(exc)
            if attempt < 2:
                time.sleep(2 ** attempt)

    if cached:                              # stale data beats no data
        return cached[1], True
    print(f"[weather] Open-Meteo unavailable ({last_err}); using fallback source")
    data = _fallback_raw(lat, lon)
    with _cache_lock:
        _cache[key] = (time.time(), data)
        _save_disk_cache()
    return data, False


def store_browser_weather(lat: float, lon: float, raw: dict) -> int:
    """
    Accept an Open-Meteo forecast response that the user's BROWSER fetched
    (its own IP has its own free quota) and put it in the shared cache, so the
    server never has to call Open-Meteo itself from a shared/throttled IP.
    Validates shape; returns the number of hourly rows stored.
    """
    hourly = raw.get("hourly") if isinstance(raw, dict) else None
    if not isinstance(hourly, dict):
        raise ValueError("missing 'hourly' block")
    cols = ["time", "shortwave_radiation", "cloudcover", "windspeed_10m", "temperature_2m"]
    for c in cols:
        if not isinstance(hourly.get(c), list):
            raise ValueError(f"hourly.{c} missing")
    n = len(hourly["time"])
    if not (24 <= n <= 400) or any(len(hourly[c]) != n for c in cols):
        raise ValueError("hourly arrays must be equal length (24-400 rows)")
    for c in cols[1:]:
        if any(v is not None and not isinstance(v, (int, float)) for v in hourly[c]):
            raise ValueError(f"hourly.{c} must be numeric")
    for t in (hourly["time"][0], hourly["time"][-1]):
        datetime.fromisoformat(t)                      # raises ValueError if malformed
    clean = {
        "hourly": {c: hourly[c] for c in cols},
        "utc_offset_seconds": int(raw.get("utc_offset_seconds", 0)),
        "timezone": str(raw.get("timezone", "UTC"))[:64],
        "_via": "browser",
    }
    with _cache_lock:
        _load_disk_cache()
        _cache[f"{lat:.2f},{lon:.2f}"] = (time.time(), clean)
        _save_disk_cache()
    return n


def fetch_weather(lat: float, lon: float, hours: int = 48) -> dict:
    """Hourly forecast starting at today's local midnight, from the shared cached request."""
    data, stale = _fetch_raw(lat, lon)
    hourly = data["hourly"]
    now_local = datetime.utcnow() + timedelta(seconds=data.get("utc_offset_seconds", 0))
    times = hourly["time"]
    start = next((i for i, t in enumerate(times)
                  if datetime.fromisoformat(t).date() >= now_local.date()), 0)
    sl = slice(start, start + hours)
    return {
        "timestamps": times[sl],
        "radiation": [v or 0.0 for v in hourly["shortwave_radiation"][sl]],
        "cloudcover": [v or 0.0 for v in hourly["cloudcover"][sl]],
        "windspeed": [v or 0.0 for v in hourly["windspeed_10m"][sl]],
        "temperature": [20.0 if v is None else v for v in hourly["temperature_2m"][sl]],
        "timezone": data.get("timezone", "UTC"),
        "stale": stale,
        "data_source": data.get("_source", "open-meteo"),
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
    Last `hours` (max 72) of hourly weather: real recent conditions used to fill
    the LSTM's input window. Shares the cached request with fetch_weather, so
    forecast + backfill together cost one API call per location per few hours.
    """
    data, stale = _fetch_raw(lat, lon)
    window = select_recent_window(
        data["hourly"], data.get("utc_offset_seconds", 0), datetime.utcnow(), hours
    )
    window["timezone"] = data.get("timezone", "UTC")
    window["stale"] = stale
    window["data_source"] = data.get("_source", "open-meteo")
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
