"""
detection_v2.py  -  Tuned detection layer for the SABSS simulation

What changed compared to anomaly_detection.py / tune_models.py
  1. Supervised Random Forest classifiers replace unsupervised Isolation Forest.
     (Labels exist in the simulation, and a deployed agent would be calibrated
     on labelled historical incidents anyway.)
  2. Causal features only: each tick uses current + past readings, never future
     ones, so the same code can run inside a live GAMA simulation.
  3. Honest evaluation: blocked 5-fold cross-validation. The model is always
     scored on data it was NOT trained on (the old scripts scored on training data).
  4. Event-level metrics (event detection rate and detection delay) are added,
     which map directly to "response time" in the research questions.
  5. Vessel data is rebuilt with realistic overlap (legal fishing boats and
     licensed bunker barges also move slowly near pipelines).
"""
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix

SEED = 42
FS = 500            # acoustic sampling rate in Hz (see note on regenerating the data)
rng = np.random.default_rng(SEED)

os.makedirs("../models", exist_ok=True)
os.makedirs("../outputs/plots", exist_ok=True)
DATA = "../data/sensor_logs/"


# ───────────────────────── helpers ─────────────────────────
def rf():
    return RandomForestClassifier(
        n_estimators=300, min_samples_leaf=3, class_weight="balanced",
        random_state=SEED, n_jobs=-1)


def metrics(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    far = fp / (fp + tn) if (fp + tn) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    pre = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * pre * rec / (pre + rec) if (pre + rec) else 0.0
    return dict(accuracy=acc, far=far, recall=rec, precision=pre, f1=f1)


def event_metrics(y, p):
    """Event-level view: was each true event detected, and how late (in ticks)?"""
    y = np.asarray(y); p = np.asarray(p)
    starts = np.where((y == 1) & (np.r_[0, y[:-1]] == 0))[0]
    ends = np.where((y == 1) & (np.r_[y[1:], 0] == 0))[0]
    detected, delays = 0, []
    for s, e in zip(starts, ends):
        hit = np.where(p[s:e + 1] == 1)[0]
        if len(hit):
            detected += 1
            delays.append(hit[0])
    n = len(starts)
    return dict(events=n, detected=detected,
                event_recall=detected / n if n else 0.0,
                mean_delay=float(np.mean(delays)) if delays else float("nan"))


def blocked_cv(X, y, n_splits=5):
    """Contiguous (non-shuffled) folds so neighbouring ticks never leak across the split."""
    pred = cross_val_predict(rf(), X, y, cv=KFold(n_splits=n_splits, shuffle=False))
    return pred


def report(name, y, pred, results, event_level=True):
    m = metrics(y, pred)
    row = dict(algorithm=name, **m)
    if event_level:
        row.update(event_metrics(y, pred))
    results.append(row)
    print(f"\n--- {name} ---")
    print(f"Accuracy  : {m['accuracy']*100:5.1f}%")
    print(f"FAR       : {m['far']*100:5.1f}%")
    print(f"Recall    : {m['recall']*100:5.1f}%   Precision: {m['precision']*100:5.1f}%   F1: {m['f1']:.3f}")
    if event_level:
        e = row
        print(f"Events    : {e['detected']}/{e['events']} detected "
              f"(mean detection delay {e['mean_delay']:.1f} ticks)")


def save_model(name, model, features):
    joblib.dump({"model": model, "features": features}, f"../models/{name}.pkl")


def plot_ts(df, signal_col, y, pred, title, fname):
    fig, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    ax[0].plot(df["tick"], df[signal_col], lw=0.7, color="steelblue")
    ax[0].set_ylabel(signal_col); ax[0].set_title(title)
    ax[1].plot(df["tick"], y, label="Ground truth", color="green", lw=1.2)
    ax[1].plot(df["tick"], pred, label="Predicted (out-of-fold)", color="red",
               lw=0.9, ls="--", alpha=0.8)
    ax[1].set_ylabel("Anomaly"); ax[1].set_xlabel("Simulation tick"); ax[1].legend()
    plt.tight_layout(); plt.savefig(f"../outputs/plots/{fname}", dpi=150); plt.close()


results = []

# ───────────────────── 1. PRESSURE DROP ─────────────────────
print("=" * 60); print("1. PRESSURE DROP DETECTION"); print("=" * 60)
dp = pd.read_csv(DATA + "pressure_data.csv")
p = dp["pressure_bar"]
d = p.diff().fillna(0)
# Only RELATIVE features: absolute pressure drifts along the pipeline (P0 - kx - at),
# so the detector must look at local change, not the raw level.
dp["drop_from_max_60"] = p.rolling(60, min_periods=1).max() - p
dp["dev_median_100"] = p - p.rolling(100, min_periods=1).median()
dp["diff_1"] = d
dp["min_diff_10"] = d.rolling(10, min_periods=1).min()
dp["std_10"] = p.rolling(10, min_periods=2).std().fillna(0)
dp["mean5_minus_mean30"] = p.rolling(5, min_periods=1).mean() - p.rolling(30, min_periods=1).mean()
f_p = ["drop_from_max_60", "dev_median_100", "diff_1", "min_diff_10", "std_10", "mean5_minus_mean30"]
y_p = dp["label"].values
pred_p = blocked_cv(dp[f_p].values, y_p)
report("Pressure Drop", y_p, pred_p, results)
save_model("pressure_rf", rf().fit(dp[f_p].values, y_p), f_p)
plot_ts(dp, "pressure_bar", y_p, pred_p, "Pressure drop detection (blocked CV)", "pressure_v2.png")

# ───────────────────── 2. DRILLING (ACOUSTIC) ─────────────────────
print("\n" + "=" * 60); print("2. DRILLING / ACOUSTIC DETECTION"); print("=" * 60)
da = pd.read_csv(DATA + "acoustic_data.csv")
x = da["vibration_amplitude"].values
WIN = 64


def causal_band_fraction(sig, win, fs, lo, hi):
    """Fraction of spectral energy in [lo, hi] Hz over the last `win` samples."""
    out = np.zeros(len(sig))
    freqs = np.fft.rfftfreq(win, d=1 / fs)
    band = (freqs >= lo) & (freqs <= hi)
    hann = np.hanning(win)
    for i in range(win - 1, len(sig)):
        seg = sig[i - win + 1:i + 1]
        seg = (seg - seg.mean()) * hann
        e = np.abs(np.fft.rfft(seg)) ** 2
        out[i] = e[band].sum() / (e.sum() + 1e-12)
    return out


s = pd.Series(x)
da["drill_band_frac"] = causal_band_fraction(x, WIN, FS, 35, 65)
da["rms_32"] = s.rolling(32, min_periods=2).std().fillna(0)
da["absdiff_16"] = s.diff().abs().rolling(16, min_periods=1).mean().fillna(0)
da["peak_32"] = s.abs().rolling(32, min_periods=1).max()
da["band_max_150"] = pd.Series(da["drill_band_frac"]).rolling(150, min_periods=1).max()
da["rms_max_150"] = da["rms_32"].rolling(150, min_periods=1).max()
f_a = ["drill_band_frac", "rms_32", "absdiff_16", "peak_32", "band_max_150", "rms_max_150"]
y_a = da["label"].values
pred_a = blocked_cv(da[f_a].values, y_a)
report("Drilling Detection", y_a, pred_a, results)
save_model("drilling_rf", rf().fit(da[f_a].values, y_a), f_a)
plot_ts(da, "vibration_amplitude", y_a, pred_a, "Drilling detection (blocked CV)", "drilling_v2.png")

# ───────────────────── 3. OIL SPILL (CHEMICAL) ─────────────────────
print("\n" + "=" * 60); print("3. OIL SPILL / CHEMICAL DETECTION"); print("=" * 60)
dc = pd.read_csv(DATA + "chemical_data.csv")
h = dc["hydrocarbon_ppm"]
base = h.rolling(200, min_periods=20).median().bfill()
dc["ppm"] = h
dc["ratio_to_base"] = h / (base + 1e-3)
dc["diff_1"] = h.diff().fillna(0)
dc["max_20"] = h.rolling(20, min_periods=1).max()
dc["max_80"] = h.rolling(80, min_periods=1).max()
dc["mean_20"] = h.rolling(20, min_periods=1).mean()
f_c = ["ppm", "ratio_to_base", "diff_1", "max_20", "max_80", "mean_20"]
y_c = dc["label"].values
pred_c = blocked_cv(dc[f_c].values, y_c)
report("Oil Spill", y_c, pred_c, results)
save_model("chemical_rf", rf().fit(dc[f_c].values, y_c), f_c)
plot_ts(dc, "hydrocarbon_ppm", y_c, pred_c, "Oil spill detection (blocked CV)", "spill_v2.png")

# ───────────────────── 4. VESSEL ANOMALY (legal vs illegal) ─────────────────────
print("\n" + "=" * 60); print("4. VESSEL ANOMALY DETECTION"); print("=" * 60)


def make_vessels(n_each=(150, 90, 60, 90, 45, 45)):
    """
    Per-vessel behaviour features. Classes deliberately overlap:
      legal : cargo/transit, fishing boats, licensed bunker barges
      illegal: loitering tappers, hit-and-run approaches, mother-ship transfers
    Columns are the features a SensorAgent/DecisionAgent could compute from AIS + radar.
    """
    rows = []

    def add(n, label, kind, speed, near, spread, night, gaps, headings):
        for _ in range(n):
            rows.append(dict(
                kind=kind, label=label,
                mean_speed=max(0.2, rng.normal(*speed)),
                frac_near_pipeline=float(np.clip(rng.normal(*near), 0, 1)),
                position_spread_km=max(0.01, rng.normal(*spread)),
                night_fraction=float(np.clip(rng.normal(*night), 0, 1)),
                ais_gap_count=int(max(0, rng.poisson(gaps))),
                heading_changes=max(0, rng.normal(*headings)),
            ))
    c = n_each
    # kind            label  speed      near_pipe   spread     night      gaps  heading chg
    add(c[0], 0, "cargo_transit",   (12, 3.0),  (0.08, 0.08), (9.0, 3.0),  (0.45, 0.2),  0.2, (3, 2))
    add(c[1], 0, "fishing",         (3.5, 1.6), (0.25, 0.18), (1.8, 0.9),  (0.30, 0.2),  0.4, (14, 5))
    add(c[2], 0, "licensed_barge",  (2.5, 1.3), (0.55, 0.20), (0.6, 0.3),  (0.35, 0.2),  0.3, (4, 2))
    add(c[3], 1, "loitering_tapper", (2.2, 1.2), (0.70, 0.18), (0.5, 0.3), (0.70, 0.2),  2.5, (9, 4))
    add(c[4], 1, "hit_and_run",     (9.0, 3.0), (0.30, 0.15), (5.0, 2.0),  (0.80, 0.15), 3.0, (6, 3))
    add(c[5], 1, "mother_ship",     (1.5, 0.9), (0.45, 0.20), (0.8, 0.4),  (0.75, 0.2),  2.0, (3, 2))
    return pd.DataFrame(rows)


dv = make_vessels()
dv.to_csv(DATA + "vessel_features_v2.csv", index=False)
f_v = ["mean_speed", "frac_near_pipeline", "position_spread_km",
       "night_fraction", "ais_gap_count", "heading_changes"]
y_v = dv["label"].values
pred_v = cross_val_predict(rf(), dv[f_v].values, y_v,
                           cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
report("Vessel Anomaly", y_v, pred_v, results, event_level=False)
save_model("vessel_rf", rf().fit(dv[f_v].values, y_v), f_v)
print("Per-class detection rate:")
dv["correct"] = (pred_v == y_v).astype(int)
print(dv.groupby("kind")["correct"].mean().round(3).to_string())

# ───────────────────── SUMMARY ─────────────────────
res = pd.DataFrame(results)
res.to_csv("../outputs/results_summary.csv", index=False)
print("\n" + "=" * 66)
print("TUNED EVALUATION SUMMARY (out-of-sample, blocked cross-validation)")
print("=" * 66)
print(f"{'Algorithm':<22}{'Accuracy':>10}{'FAR':>8}{'Recall':>9}{'F1':>8}{'Evt det.':>10}{'Delay':>8}")
print("-" * 66)
for _, r in res.iterrows():
    ev = f"{int(r['detected'])}/{int(r['events'])}" if "events" in r and not pd.isna(r.get("events")) else "-"
    dl = f"{r['mean_delay']:.1f}" if "mean_delay" in r and not pd.isna(r.get("mean_delay")) else "-"
    print(f"{r['algorithm']:<22}{r['accuracy']*100:>9.1f}%{r['far']*100:>7.1f}%"
          f"{r['recall']*100:>8.1f}%{r['f1']:>8.3f}{ev:>10}{dl:>8}")
print("=" * 66)
print("Saved: ../models/*.pkl, ../outputs/results_summary.csv, ../outputs/plots/*_v2.png")
