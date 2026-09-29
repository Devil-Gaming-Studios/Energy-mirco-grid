"""
ML-based forecaster: an LSTM that takes a rolling window of past hourly
readings (irradiance, wind speed, temperature, time-of-day) and predicts the
next 48 hours of solar irradiance and wind speed.

IMPORTANT: the model architecture below (LSTMForecaster), SEQ_LEN, HORIZON,
and the normalization constants must exactly match the Colab notebook
(colab/train_forecaster.ipynb), or a .pth file trained there will either fail
to load (shape mismatch) or load fine but produce meaningless predictions
(mismatched normalization). If you change one, change the other.

Why an LSTM (and not the simpler single-snapshot MLP this project used
earlier): the training data is now real, long hourly time series from NASA
POWER (https://power.larc.nasa.gov), which is exactly the kind of sequential
data an LSTM is built to exploit — it can learn how irradiance and wind speed
typically evolve hour-to-hour, not just their instantaneous value. See the
notebook for how the training data is fetched and prepared, including a
genuine held-out-time-period backtest with MAE/RMSE reported in physical
units, so its accuracy claim isn't hand-waved.

What this model is NOT: it has not been validated against this specific
project's actual deployment site, and the backend's rolling history buffer
(see backend/sensors.py get_history_for_ml) will contain simulated/padded
readings until real sensor data has been flowing in for at least SEQ_LEN
hours. Treat early-life predictions accordingly — the API response's
`sensor_meta` field reports how many of the input readings were real vs.
padded so this is visible, not hidden.
"""

import os
import math
from datetime import datetime, timedelta

import torch
import torch.nn as nn

SEQ_LEN = 72        # hours of history fed into the LSTM
HORIZON = 48         # hours predicted forward
INPUT_DIM = 5        # hour_sin, hour_cos, irradiance_norm, wind_norm, temp_norm
HIDDEN_DIM = 64
NUM_LAYERS = 2
OUTPUT_DIM = 2        # irradiance_norm, wind_norm, per predicted hour

WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "weights", "solar_wind_forecaster.pth")


class LSTMForecaster(nn.Module):
    """Must be identical to the class of the same name in the Colab notebook."""
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM,
                 num_layers=NUM_LAYERS, horizon=HORIZON, output_dim=OUTPUT_DIM):
        super().__init__()
        self.horizon = horizon
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # final layer's hidden state, shape (batch, hidden_dim)
        out = self.head(last_hidden)
        return out.view(-1, self.horizon, self.output_dim)


_model = None  # lazily loaded singleton


def _normalize_point(hour, irradiance, wind_kmh, temperature_c):
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)
    irr_norm = max(0.0, min(1.0, irradiance / 1000.0))
    wind_norm = max(0.0, min(1.0, wind_kmh / 100.0))
    temp_norm = max(0.0, min(1.0, temperature_c / 40.0))
    return [hour_sin, hour_cos, irr_norm, wind_norm, temp_norm]


def load_model(weights_path: str = WEIGHTS_PATH):
    """Load weights from disk. Returns True if a model is now loaded, False otherwise."""
    global _model
    if not os.path.exists(weights_path):
        _model = None
        return False
    model = LSTMForecaster()
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    _model = model
    return True


def is_loaded() -> bool:
    return _model is not None


def predict_ml(history_sequence: list):
    """
    history_sequence: list of SEQ_LEN dicts, oldest first, each with
    {hour, irradiance, wind_kmh, temperature_c} — exactly what
    sensors.get_history_for_ml(SEQ_LEN) returns.

    Returns a dict in the same shape fetch_weather()/project_from_sensors()
    return, so it drops into the existing power-modeling pipeline unchanged.
    """
    if _model is None:
        return None
    if len(history_sequence) != SEQ_LEN:
        raise ValueError(f"Expected a history window of {SEQ_LEN} hours, got {len(history_sequence)}")

    features = [
        _normalize_point(p["hour"], p["irradiance"], p["wind_kmh"], p["temperature_c"])
        for p in history_sequence
    ]
    x = torch.tensor([features], dtype=torch.float32)  # (1, seq_len, input_dim)

    with torch.no_grad():
        out = _model(x)[0]  # (horizon, 2)

    last_hour = history_sequence[-1]["hour"]
    timestamps, radiation, windspeed, cloudcover, temperature = [], [], [], [], []
    base_time = datetime.utcnow()

    for h in range(HORIZON):
        irradiance_norm, wind_norm = out[h].tolist()
        timestamps.append((base_time + timedelta(hours=h + 1)).isoformat())
        radiation.append(round(irradiance_norm * 1000.0, 1))
        windspeed.append(round(wind_norm * 100.0, 1))
        cloudcover.append(None)  # the LSTM predicts irradiance directly, not a cloud-cover proxy
        temperature.append(history_sequence[-1]["temperature_c"])  # held from most recent real/padded reading

    return {
        "timestamps": timestamps,
        "radiation": radiation,
        "windspeed": windspeed,
        "cloudcover": cloudcover,
        "temperature": temperature,
        "timezone": "site-local (from LSTM model)",
        "meta": {
            "model": "LSTMForecaster",
            "seq_len_hours": SEQ_LEN,
            "trained_on": "NASA POWER real hourly historical data (see colab/train_forecaster.ipynb)",
            "seed_last_hour": last_hour,
        },
    }


# Attempt to load weights at import time so a restart picks up a previously uploaded file.
load_model()
