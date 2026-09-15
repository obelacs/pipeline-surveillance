import numpy as np
import pandas as pd
import geopandas as gpd
import os

# ── Load your pipeline shapefile ──────────────────────────────────────────────
shp_path = "../data/gis_maps/pipeline_corridor.shp"
pipeline = gpd.read_file(shp_path)
print("Pipeline loaded. CRS:", pipeline.crs)
print("Features:", len(pipeline))
print(pipeline.head())

# ── Configuration ─────────────────────────────────────────────────────────────
np.random.seed(42)
N_NORMAL    = 1000   # normal readings
N_ANOMALY   = 200    # anomaly events
TICKS       = 1200   # simulation time steps

os.makedirs("../data/sensor_logs", exist_ok=True)

# ── 1. Pressure Sensor Data ───────────────────────────────────────────────────
# Model from Chapter 3: P(x,t) = P0 - kx - αt
P0 = 50.0   # initial pressure (bar)
k  = 0.02   # friction loss per km
alpha = 0.001  # natural decay rate

time    = np.arange(TICKS)
x_pos   = np.linspace(0, 100, TICKS)   # position along pipeline (km)
noise   = np.random.normal(0, 0.3, TICKS)

pressure_normal = P0 - k * x_pos - alpha * time + noise

# Inject anomaly events (sudden pressure drops = leaks)
pressure_anomaly = pressure_normal.copy()
leak_positions = np.random.choice(range(200, TICKS-100), size=15, replace=False)
for pos in leak_positions:
    drop = np.random.uniform(5, 20)    # pressure drops 5-20 bar suddenly
    pressure_anomaly[pos:pos+50] -= drop * np.exp(-0.05 * np.arange(50))

# Label: 0 = normal, 1 = anomaly
labels_pressure = np.zeros(TICKS)
for pos in leak_positions:
    labels_pressure[pos:pos+50] = 1

pressure_df = pd.DataFrame({
    "tick": time,
    "x_position_km": x_pos,
    "pressure_bar": pressure_anomaly,
    "label": labels_pressure.astype(int)
})
pressure_df.to_csv("../data/sensor_logs/pressure_data.csv", index=False)
print(f"Pressure data saved — {int(labels_pressure.sum())} anomaly ticks")

# ── 2. Vessel AIS Movement Data ───────────────────────────────────────────────
vessel_records = []

# Normal vessels — pass through quickly, don't loiter
for v_id in range(60):
    start_lat = np.random.uniform(4.5, 5.5)
    start_lon = np.random.uniform(5.0, 7.5)
    speed     = np.random.uniform(8, 18)    # knots — moving purposefully
    heading   = np.random.uniform(0, 360)
    for t in range(30):
        vessel_records.append({
            "vessel_id": f"V_NORMAL_{v_id:03d}",
            "tick": t,
            "lat": start_lat + t * 0.005 * np.cos(np.radians(heading)),
            "lon": start_lon + t * 0.005 * np.sin(np.radians(heading)),
            "speed_knots": speed + np.random.normal(0, 0.5),
            "near_pipeline": np.random.choice([0, 1], p=[0.8, 0.2]),
            "label": 0   # normal
        })

# Suspicious vessels — loiter, slow down, circle near pipeline
for v_id in range(30):
    # Place them near the pipeline zone
    base_lat = np.random.uniform(4.8, 5.2)
    base_lon = np.random.uniform(5.8, 6.5)
    for t in range(60):
        angle = t * 12   # circles slowly
        vessel_records.append({
            "vessel_id": f"V_SUSPECT_{v_id:03d}",
            "tick": t,
            "lat": base_lat + 0.01 * np.cos(np.radians(angle)),
            "lon": base_lon + 0.01 * np.sin(np.radians(angle)),
            "speed_knots": np.random.uniform(0.5, 3.0),   # very slow / loitering
            "near_pipeline": 1,
            "label": 1   # anomaly
        })

vessel_df = pd.DataFrame(vessel_records)
vessel_df.to_csv("../data/sensor_logs/vessel_ais_data.csv", index=False)
print(f"Vessel AIS data saved — {len(vessel_df)} records")

# ── 3. Acoustic / Vibration Data ──────────────────────────────────────────────
from scipy import signal as sig

sampling_rate = 100  # Hz
duration      = TICKS / sampling_rate

t_acoustic = np.linspace(0, duration, TICKS)

# Normal vibration: low-frequency water flow hum
normal_vibration = (
    0.5 * np.sin(2 * np.pi * 2 * t_acoustic) +    # 2 Hz baseline hum
    0.2 * np.sin(2 * np.pi * 5 * t_acoustic) +    # 5 Hz harmonic
    np.random.normal(0, 0.1, TICKS)               # noise
)

# Drilling signature: high-frequency burst (40-60 Hz)
drilling_vibration = normal_vibration.copy()
drill_events = np.random.choice(range(200, TICKS-200), size=8, replace=False)
for pos in drill_events:
    burst_len = np.random.randint(80, 150)
    burst = 3.0 * np.sin(2 * np.pi * 50 * t_acoustic[pos:pos+burst_len])
    drilling_vibration[pos:pos+burst_len] += burst

labels_acoustic = np.zeros(TICKS)
for pos in drill_events:
    labels_acoustic[pos:pos+150] = 1

acoustic_df = pd.DataFrame({
    "tick": np.arange(TICKS),
    "vibration_amplitude": drilling_vibration,
    "label": labels_acoustic.astype(int)
})
acoustic_df.to_csv("../data/sensor_logs/acoustic_data.csv", index=False)
print(f"Acoustic data saved — {int(labels_acoustic.sum())} anomaly ticks")

# ── 4. Chemical / Oil Spill Sensor Data ──────────────────────────────────────
hydrocarbon_baseline = np.random.exponential(0.05, TICKS)   # near-zero normally

spill_events = np.random.choice(range(100, TICKS-100), size=10, replace=False)
for pos in spill_events:
    spike_len = np.random.randint(30, 80)
    spike = np.random.uniform(8, 25) * np.exp(-0.03 * np.arange(spike_len))
    hydrocarbon_baseline[pos:pos+spike_len] += spike

labels_chemical = np.zeros(TICKS)
for pos in spill_events:
    labels_chemical[pos:pos+80] = 1

chemical_df = pd.DataFrame({
    "tick": np.arange(TICKS),
    "hydrocarbon_ppm": hydrocarbon_baseline,
    "label": labels_chemical.astype(int)
})
chemical_df.to_csv("../data/sensor_logs/chemical_data.csv", index=False)
print(f"Chemical data saved — {int(labels_chemical.sum())} anomaly ticks")

print("\n✅ All sensor datasets generated successfully.")
print("Files saved to: data/sensor_logs/")