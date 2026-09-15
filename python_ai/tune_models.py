import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def evaluate(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    TP = cm[1][1]; FP = cm[0][1]; TN = cm[0][0]; FN = cm[1][0]
    acc = (TP + TN) / (TP + TN + FP + FN)
    far = FP / (FP + TN) if (FP + TN) > 0 else 0
    print(f"\n--- {name} ---")
    print(classification_report(y_true, y_pred,
          target_names=["Normal", "Anomaly"], zero_division=0))
    print(f"Accuracy: {acc*100:.1f}%  |  FAR: {far*100:.1f}%")
    return acc, far

# ── 1. Pressure — add threshold layer on top of Isolation Forest ──────────────
df_p = pd.read_csv("../data/sensor_logs/pressure_data.csv")
df_p['diff']         = df_p['pressure_bar'].diff().fillna(0)
df_p['rolling_mean'] = df_p['pressure_bar'].rolling(10).mean().bfill()
df_p['rolling_std']  = df_p['pressure_bar'].rolling(10).std().fillna(0)
df_p['deviation']    = df_p['pressure_bar'] - df_p['rolling_mean']
df_p['roc']          = df_p['diff'].rolling(5).mean().fillna(0)

X_p = StandardScaler().fit_transform(
    df_p[['pressure_bar','diff','rolling_mean','rolling_std','deviation','roc']].values)
y_p = df_p['label'].values

iso_p = IsolationForest(contamination=0.35, n_estimators=300, random_state=42)
y_p_pred = np.where(iso_p.fit_predict(X_p) == -1, 1, 0)

# Hybrid: also flag anything with a sudden drop > 4 bar
threshold_flags = (df_p['diff'] < -4).astype(int).values
y_p_hybrid = np.maximum(y_p_pred, threshold_flags)

acc_p, far_p = evaluate(y_p, y_p_hybrid, "Pressure Drop (Hybrid)")

# ── 2. Vessel — lower contamination, add rule for very low speed near pipeline ─
df_v = pd.read_csv("../data/sensor_logs/vessel_ais_data.csv")
vf = df_v.groupby('vessel_id').agg(
    mean_speed     = ('speed_knots', 'mean'),
    min_speed      = ('speed_knots', 'min'),
    std_speed      = ('speed_knots', 'std'),
    pct_near_pipe  = ('near_pipeline', lambda x: x.mean()),
    speed_range    = ('speed_knots', lambda x: x.max() - x.min()),
    label          = ('label', 'max')
).reset_index()

X_v = StandardScaler().fit_transform(
    vf[['mean_speed','min_speed','std_speed','pct_near_pipe','speed_range']].values)
y_v = vf['label'].values

iso_v = IsolationForest(contamination=0.25, n_estimators=300, random_state=42)
y_v_pred = np.where(iso_v.fit_predict(X_v) == -1, 1, 0)

# Rule: slow vessel (< 4 knots mean) near pipeline → flag
rule_slow = ((vf['mean_speed'] < 4) & (vf['pct_near_pipe'] > 0.5)).astype(int).values
y_v_hybrid = np.maximum(y_v_pred, rule_slow)

acc_v, far_v = evaluate(y_v, y_v_hybrid, "Vessel Anomaly (Hybrid)")

# ── 3. Drilling — smaller windows to get more samples ────────────────────────
df_a = pd.read_csv("../data/sensor_logs/acoustic_data.csv")
window_size = 20   # smaller = more windows
fft_features = []

for i in range(0, len(df_a) - window_size, 10):  # stride=10 for overlap
    window = df_a['vibration_amplitude'].values[i:i+window_size]
    label  = df_a['label'].values[i:i+window_size].max()
    fft_mag = np.abs(np.fft.rfft(window))
    freqs   = np.fft.rfftfreq(window_size, d=0.01)

    high_energy = fft_mag[(freqs >= 30) & (freqs < 70)].mean() if any((freqs >= 30) & (freqs < 70)) else 0
    low_energy  = fft_mag[freqs < 10].mean()
    peak_freq   = freqs[np.argmax(fft_mag)]
    amplitude   = np.max(np.abs(window))
    rms         = np.sqrt(np.mean(window**2))

    fft_features.append({
        'low_energy': low_energy, 'high_energy': high_energy,
        'peak_freq': peak_freq, 'amplitude': amplitude, 'rms': rms,
        'label': label
    })

df_fft = pd.DataFrame(fft_features)
print(f"\nDrilling windows after overlap: {len(df_fft)}")

X_a = StandardScaler().fit_transform(
    df_fft[['low_energy','high_energy','peak_freq','amplitude','rms']].values)
y_a = df_fft['label'].values

iso_a = IsolationForest(contamination=0.3, n_estimators=300, random_state=42)
y_a_pred = np.where(iso_a.fit_predict(X_a) == -1, 1, 0)

# Rule: high energy in drilling band → flag
rule_drill = (df_fft['high_energy'] > df_fft['high_energy'].mean() + 
              2 * df_fft['high_energy'].std()).astype(int).values
y_a_hybrid = np.maximum(y_a_pred, rule_drill)

acc_a, far_a = evaluate(y_a, y_a_hybrid, "Drilling Detection (Hybrid)")

# ── 4. Oil Spill — lower contamination, stronger threshold ───────────────────
df_c = pd.read_csv("../data/sensor_logs/chemical_data.csv")
df_c['rolling_mean'] = df_c['hydrocarbon_ppm'].rolling(20).mean().bfill()
df_c['rolling_std']  = df_c['hydrocarbon_ppm'].rolling(20).std().fillna(1)
df_c['z_score']      = (df_c['hydrocarbon_ppm'] - df_c['rolling_mean']) / df_c['rolling_std']
df_c['rate_change']  = df_c['hydrocarbon_ppm'].diff().fillna(0)

X_c = StandardScaler().fit_transform(
    df_c[['hydrocarbon_ppm','z_score','rate_change','rolling_mean']].values)
y_c = df_c['label'].values

iso_c = IsolationForest(contamination=0.25, n_estimators=300, random_state=42)
y_c_pred = np.where(iso_c.fit_predict(X_c) == -1, 1, 0)

# Rule: z-score > 3 is a definite spill
rule_spill = (df_c['z_score'] > 3).astype(int).values
y_c_hybrid = np.maximum(y_c_pred, rule_spill)

acc_c, far_c = evaluate(y_c, y_c_hybrid, "Oil Spill Detection (Hybrid)")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 58)
print("TUNED EVALUATION SUMMARY")
print("=" * 58)
print(f"{'Algorithm':<30} {'Accuracy':>10} {'FAR':>10}")
print("-" * 52)
print(f"{'Pressure Drop':<30} {acc_p*100:>9.1f}% {far_p*100:>9.1f}%")
print(f"{'Vessel Anomaly':<30} {acc_v*100:>9.1f}% {far_v*100:>9.1f}%")
print(f"{'Drilling Detection':<30} {acc_a*100:>9.1f}% {far_a*100:>9.1f}%")
print(f"{'Oil Spill':<30} {acc_c*100:>9.1f}% {far_c*100:>9.1f}%")
print("=" * 58)
