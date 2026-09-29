# ⚡ Energy Micro-Grid System

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8+-green.svg)
![Status](https://img.shields.io/badge/Status-Active_Development-orange)

An intelligent, real-time Micro-Grid Energy Management and Control System. This platform simulates, optimizes, and coordinates distributed energy resources (DERs)—including renewable generators, energy storage systems, and dynamic dynamic load demands—to achieve grid stability and economic energy dispatch.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage & Configuration](#-usage--configuration)
- [Core Components](#-core-components)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🌟 Overview

As local power networks integrate more variable energy sources (such as solar and wind), maintaining power quality and frequency/voltage stability becomes challenging.

The **Energy Micro-Grid** software models real-time generation and load dynamics. It implements automated control strategies for:

1. **Demand-Side Management (DSM):** Balancing peak demand loads against available supply.
2. **Battery Energy Storage Optimization (BESS):** Managing State-of-Charge (SoC) cycles to prolong battery life and optimize peak shaving.
3. **Islanded & Grid-Tied Operation:** Simulating seamless transitions between main grid connection and autonomous micro-grid functioning during grid outages.

---

## ✨ Key Features

- **☀️ Distributed Renewable Generation:** Real-time generation simulation for PV Solar, Wind Turbines, and Auxiliary Power Units.
- **🔋 Battery Management System (BMS):** Configurable charge/discharge limits, depth-of-discharge (DoD) protection, and efficiency parameters.
- **📊 Dynamic Demand Response:** Tracks fluctuating loads across residential, commercial, and industrial sub-grids.
- **⚖️ Economic Dispatch Engine:** Optimizes energy routing based on cost curves, feed-in tariffs, and real-time electricity pricing.
- **🛡️ Fault Detection & Resilience:** Monitors grid frequency/voltage anomalies to trigger load shedding or islanding protocols.

---

## 🏗️ System Architecture

```text
               +-----------------------+
               | Main Utility Grid     |
               +-----------+-----------+
                           |
                     [ Grid Tie Switch ]
                           |
 +-------------------------+-------------------------+
 |                 Micro-Grid Bus                    |
 +-------+-----------------+-----------------+-------+
         |                 |                 |
  +------+------+   +------+------+   +------+------+
  | Solar / Wind|   | Battery BESS |   | Grid Loads  |
  | Generation  |   | Storage System|  | (Res/Ind)   |
  +-------------+   +-------------+   +-------------+
         ^                 ^                 ^
         |                 |                 |
 +-------+-----------------+-----------------+-------+
 |             Micro-Grid Controller                 |
 |          (Optimization & Scheduling)              |
 +---------------------------------------------------+
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+** (or compatible runtime)
- **Git** installed on your system

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Devil-Gaming-Studios/Energy-mirco-grid.git
   cd Energy-mirco-grid
   ```

2. **Create and activate a virtual environment:**
   - **Linux/macOS:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - **Windows:**
     ```cmd
     python -m venv venv
     venv\Scripts\activate
     ```

3. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage & Configuration

### Running the Micro-Grid Engine

To launch the primary simulation engine with default micro-grid parameters:

```bash
python main.py
```

### Configuration

System params (such as battery capacity, solar peak rating, and load curves) are defined within `config/grid_config.json` (or `config.py` depending on structure).

Example configuration snippet:

```json
{
  "grid_mode": "grid_tied",
  "battery": {
    "capacity_kwh": 100,
    "max_charge_kw": 20,
    "min_soc_pct": 20,
    "max_soc_pct": 95
  },
  "pv_solar": {
    "peak_capacity_kw": 50
  }
}
```

---

## 🤝 Contributing

Contributions are always welcome!

1. **Fork** the repository.
2. **Create** a branch for your feature (`git checkout -b feature/NewGridFeature`).
3. **Commit** your updates (`git commit -m 'Add NewGridFeature'`).
4. **Push** to the branch (`git push origin feature/NewGridFeature`).
5. **Open** a Pull Request for review.

---
