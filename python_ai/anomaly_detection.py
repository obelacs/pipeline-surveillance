import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

os.makedirs("../models", exist_ok=True)
os.makedirs("../outputs/plots", exist_ok=True)

# ─── 1. PRESSURE DROP DETECTION ──────────────────────────────────────────────
# Implements P(x,t) = P0 - kx - αt model from Chapter 3
# Detects sudden drops indicating leaks or illegal tapping

print("=" * 60)
print("ALGORITHM 1: PRESSURE DROP DETECTION")
print("=" * 60)

df_pressure = pd.read_csv("../data/sensor_logs/pressure_data.csv")

print(f"Loaded {len(df_pressure)} pressure readings")
print(f"Normal ticks:  {(df_pressure['label'] == 0).sum()}")
print(f"Anomaly ticks: {(df_pressure['label'] == 1).sum()}")

# Feature engineering — what the ML model actually learns from
df_pressure['pressure_diff']    = df_pressure['pressure_bar'].diff().fillna(0)
df_pressure['rolling_mean_10']  = df_pressure['pressure_bar'].rolling(10).mean().bfill()
df_pressure['rolling_std_10']   = df_pressure['pressure_bar'].rolling(10).std().fillna(0)
df_pressure['deviation']        = df_pressure['pressure_bar'] - df_pressure['rolling_mean_10']
df_pressure['rate_of_change']   = df_pressure['pressure_diff'].rolling(5).mean().fillna(0)

features = ['pressure_bar', 'pressure_diff', 'rolling_mean_10',
            'rolling_std_10', 'deviation', 'rate_of_change']

X = df_pressure[features].values
y = df_pressure['label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Isolation Forest — ideal for anomaly detection without needing labelled
# training data (matches the operational scenario where normal data dominates)
iso_forest = IsolationForest(
    contamination=0.15,   # approx fraction of anomalies expected
    n_estimators=200,
    random_state=42
)
iso_forest.fit(X_scaled)

# Isolation Forest returns -1 (anomaly) or 1 (normal) — convert to 0/1
y_pred_raw = iso_forest.predict(X_scaled)
y_pred = np.where(y_pred_raw == -1, 1, 0)

print("\n--- Pressure Drop Detection Results ---")
print(classification_report(y, y_pred, target_names=["Normal", "Anomaly"]))

cm = confusion_matrix(y, y_pred)
TP = cm[1][1]; FP = cm[0][1]; TN = cm[0][0]; FN = cm[1][0]
detection_accuracy = (TP + TN) / (TP + TN + FP + FN)
false_alarm_rate   = FP / (FP + TN) if (FP + TN) > 0 else 0

print(f"Detection Accuracy : {detection_accuracy:.3f} ({detection_accuracy*100:.1f}%)")
print(f"False Alarm Rate   : {false_alarm_rate:.3f} ({false_alarm_rate*100:.1f}%)")

# Save model and scaler
joblib.dump(iso_forest, "../models/pressure_detector.pkl")
joblib.dump(scaler,     "../models/pressure_scaler.pkl")
print("Model saved to ../models/pressure_detector.pkl")

# Plot
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

axes[0].plot(df_pressure['tick'], df_pressure['pressure_bar'],
             color='steelblue', linewidth=0.8, label='Pressure (bar)')
axes[0].set_title('Pipeline Pressure Over Time')
axes[0].set_ylabel('Pressure (bar)')
axes[0].legend()

axes[1].plot(df_pressure['tick'], y, color='green',
             linewidth=0.8, label='Ground Truth')
axes[1].plot(df_pressure['tick'], y_pred, color='red',
             linewidth=0.8, alpha=0.7, linestyle='--', label='Predicted')
axes[1].set_title('Ground Truth vs Predicted Anomalies')
axes[1].set_ylabel('Anomaly (1=Yes)')
axes[1].legend()

axes[2].plot(df_pressure['tick'], df_pressure['deviation'],
             color='orange', linewidth=0.8, label='Pressure Deviation')
axes[2].axhline(y=0, color='black', linestyle='--', linewidth=0.5)
axes[2].set_title('Pressure Deviation from Rolling Mean')
axes[2].set_ylabel('Deviation (bar)')
axes[2].set_xlabel('Simulation Tick')
axes[2].legend()

plt.tight_layout()
plt.savefig("../outputs/plots/pressure_detection.png", dpi=150)
print("Plot saved to ../outputs/plots/pressure_detection.png")
plt.close()


# ─── 2. VESSEL ANOMALY DETECTION ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ALGORITHM 2: VESSEL ANOMALY DETECTION")
print("=" * 60)

df_vessel = pd.read_csv("../data/sensor_logs/vessel_ais_data.csv")

print(f"Loaded {len(df_vessel)} AIS records")
print(f"Unique vessels: {df_vessel['vessel_id'].nunique()}")

# Aggregate per vessel — classify each vessel as suspicious or not
vessel_features = df_vessel.groupby('vessel_id').agg(
    mean_speed      = ('speed_knots', 'mean'),
    min_speed       = ('speed_knots', 'min'),
    max_speed       = ('speed_knots', 'max'),
    std_speed       = ('speed_knots', 'std'),
    time_near_pipe  = ('near_pipeline', 'sum'),
    total_ticks     = ('tick', 'count'),
    label           = ('label', 'max')   # if any tick was suspicious, vessel is suspicious
).reset_index()

vessel_features['pct_near_pipeline'] = (
    vessel_features['time_near_pipe'] / vessel_features['total_ticks']
)
vessel_features['speed_range'] = (
    vessel_features['max_speed'] - vessel_features['min_speed']
)

X_v = vessel_features[['mean_speed', 'min_speed', 'std_speed',
                         'pct_near_pipeline', 'speed_range']].values
y_v = vessel_features['label'].values

scaler_v = StandardScaler()
X_v_scaled = scaler_v.fit_transform(X_v)

iso_vessel = IsolationForest(contamination=0.33, n_estimators=200, random_state=42)
iso_vessel.fit(X_v_scaled)

y_v_pred_raw = iso_vessel.predict(X_v_scaled)
y_v_pred = np.where(y_v_pred_raw == -1, 1, 0)

print("\n--- Vessel Anomaly Detection Results ---")
print(classification_report(y_v, y_v_pred, target_names=["Normal", "Suspicious"]))

cm_v = confusion_matrix(y_v, y_v_pred)
TP = cm_v[1][1]; FP = cm_v[0][1]; TN = cm_v[0][0]; FN = cm_v[1][0]
accuracy_v = (TP + TN) / (TP + TN + FP + FN)
far_v      = FP / (FP + TN) if (FP + TN) > 0 else 0

print(f"Detection Accuracy : {accuracy_v:.3f} ({accuracy_v*100:.1f}%)")
print(f"False Alarm Rate   : {far_v:.3f} ({far_v*100:.1f}%)")

joblib.dump(iso_vessel, "../models/vessel_detector.pkl")
joblib.dump(scaler_v,   "../models/vessel_scaler.pkl")
print("Model saved to ../models/vessel_detector.pkl")


# ─── 3. ACOUSTIC / DRILLING DETECTION ────────────────────────────────────────
print("\n" + "=" * 60)
print("ALGORITHM 3: DRILLING / ACOUSTIC DETECTION")
print("=" * 60)

df_acoustic = pd.read_csv("../data/sensor_logs/acoustic_data.csv")

# FFT-based feature extraction — drilling has a distinct 40-60 Hz signature
window_size = 50
fft_features = []

for i in range(0, len(df_acoustic) - window_size, window_size):
    window  = df_acoustic['vibration_amplitude'].values[i:i+window_size]
    label   = df_acoustic['label'].values[i:i+window_size].max()
    fft_mag = np.abs(np.fft.rfft(window))
    freqs   = np.fft.rfftfreq(window_size, d=0.01)  # sampling rate=100Hz

    low_energy  = fft_mag[(freqs >= 0)  & (freqs < 10)].mean()
    mid_energy  = fft_mag[(freqs >= 10) & (freqs < 30)].mean()
    high_energy = fft_mag[(freqs >= 30) & (freqs < 70)].mean()  # drilling band
    peak_freq   = freqs[np.argmax(fft_mag)]
    total_power = np.sum(fft_mag ** 2)

    fft_features.append({
        'low_energy':   low_energy,
        'mid_energy':   mid_energy,
        'high_energy':  high_energy,  # this is the drilling signature
        'peak_freq':    peak_freq,
        'total_power':  total_power,
        'label':        label
    })

df_fft = pd.DataFrame(fft_features)

X_a = df_fft[['low_energy', 'mid_energy', 'high_energy',
               'peak_freq', 'total_power']].values
y_a = df_fft['label'].values

scaler_a = StandardScaler()
X_a_scaled = scaler_a.fit_transform(X_a)

iso_acoustic = IsolationForest(contamination=0.2, n_estimators=200, random_state=42)
iso_acoustic.fit(X_a_scaled)

y_a_pred_raw = iso_acoustic.predict(X_a_scaled)
y_a_pred = np.where(y_a_pred_raw == -1, 1, 0)

print("\n--- Drilling Detection Results ---")
print(classification_report(y_a, y_a_pred, target_names=["Normal", "Drilling"]))

cm_a = confusion_matrix(y_a, y_a_pred)
TP = cm_a[1][1]; FP = cm_a[0][1]; TN = cm_a[0][0]; FN = cm_a[1][0]
accuracy_a = (TP + TN) / (TP + TN + FP + FN)
far_a      = FP / (FP + TN) if (FP + TN) > 0 else 0

print(f"Detection Accuracy : {accuracy_a:.3f} ({accuracy_a*100:.1f}%)")
print(f"False Alarm Rate   : {far_a:.3f} ({far_a*100:.1f}%)")

joblib.dump(iso_acoustic, "../models/acoustic_detector.pkl")
joblib.dump(scaler_a,     "../models/acoustic_scaler.pkl")
print("Model saved to ../models/acoustic_detector.pkl")


# ─── 4. CHEMICAL / OIL SPILL DETECTION ───────────────────────────────────────
print("\n" + "=" * 60)
print("ALGORITHM 4: OIL SPILL / CHEMICAL DETECTION")
print("=" * 60)

df_chem = pd.read_csv("../data/sensor_logs/chemical_data.csv")

# Z-score based detection — spills are extreme outliers in hydrocarbon ppm
df_chem['rolling_mean'] = df_chem['hydrocarbon_ppm'].rolling(20).mean().bfill()
df_chem['rolling_std']  = df_chem['hydrocarbon_ppm'].rolling(20).std().fillna(1)
df_chem['z_score']      = ((df_chem['hydrocarbon_ppm'] - df_chem['rolling_mean'])
                            / df_chem['rolling_std'])
df_chem['rate_change']  = df_chem['hydrocarbon_ppm'].diff().fillna(0)

X_c = df_chem[['hydrocarbon_ppm', 'z_score', 'rate_change', 'rolling_mean']].values
y_c = df_chem['label'].values

scaler_c = StandardScaler()
X_c_scaled = scaler_c.fit_transform(X_c)

iso_chem = IsolationForest(contamination=0.12, n_estimators=200, random_state=42)
iso_chem.fit(X_c_scaled)

y_c_pred_raw = iso_chem.predict(X_c_scaled)
y_c_pred = np.where(y_c_pred_raw == -1, 1, 0)

print("\n--- Oil Spill Detection Results ---")
print(classification_report(y_c, y_c_pred, target_names=["Normal", "Spill"]))

cm_c = confusion_matrix(y_c, y_c_pred)
TP = cm_c[1][1]; FP = cm_c[0][1]; TN = cm_c[0][0]; FN = cm_c[1][0]
accuracy_c = (TP + TN) / (TP + TN + FP + FN)
far_c      = FP / (FP + TN) if (FP + TN) > 0 else 0

print(f"Detection Accuracy : {accuracy_c:.3f} ({accuracy_c*100:.1f}%)")
print(f"False Alarm Rate   : {far_c:.3f} ({far_c*100:.1f}%)")

joblib.dump(iso_chem, "../models/chemical_detector.pkl")
joblib.dump(scaler_c, "../models/chemical_scaler.pkl")
print("Model saved to ../models/chemical_detector.pkl")


# ─── SUMMARY ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)
print(f"{'Algorithm':<30} {'Accuracy':>10} {'FAR':>10}")
print("-" * 52)
print(f"{'Pressure Drop Detection':<30} {detection_accuracy*100:>9.1f}% {false_alarm_rate*100:>9.1f}%")
print(f"{'Vessel Anomaly Detection':<30} {accuracy_v*100:>9.1f}% {far_v*100:>9.1f}%")
print(f"{'Drilling Detection':<30} {accuracy_a*100:>9.1f}% {far_a*100:>9.1f}%")
print(f"{'Oil Spill Detection':<30} {accuracy_c*100:>9.1f}% {far_c*100:>9.1f}%")
print("=" * 60)
print("\n✅ All 4 detection algorithms trained and saved.")
