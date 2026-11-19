#!/usr/bin/env python3
"""
build_dataset.py  -  SABSS-Sim: scenario-based synthetic dataset generator + exporter
===================================================================================
Smart Agent-Based Surveillance System (SABSS) for illegal pipeline bunkering in
Nigeria's maritime backwaters.

What it does
  * Defines 13 documented simulation scenarios (the SCENARIOS registry below).
  * Simulates many independent runs per scenario on REAL pipeline geometry taken
    from your OpenStreetMap shapefile (falls back to a synthetic corridor).
  * Produces multi-modal, geo-referenced, fully labelled data:
        sensor_readings  - pressure, acoustic features, hydrocarbon, temperature, turbidity
        vessel_tracks    - AIS/radar-fused tracks of legal and illegal vessels
        events           - ground-truth event register (what/where/when/how severe)
        runs, nodes      - per-run metadata and sensor-node coordinates
  * Exports a self-describing dataset package: CSV.gz (+ optional Parquet), scenario
    catalogue (JSON/CSV/Markdown), data dictionary, datasheet, statistics, GeoJSON
    of the pipeline segments used, a checksum manifest and an optional ZIP.

Usage (from python_ai/ with the venv active)
    python build_dataset.py --scale small               # quick test (~1 min)
    python build_dataset.py --scale medium --zip        # default research size
    python build_dataset.py --scale large --parquet --zip
    python build_dataset.py --list-scenarios
Everything is reproducible: same --seed + same options => identical dataset.
"""
import argparse
import gzip
import hashlib
import json
import os
import platform
import sys
import time
import zipfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.signal import lfilter

VERSION = "1.0.0"
FS = 500                      # acoustic sampling rate (Hz) of the synthetic waveform window
WIN = 128                     # samples per acoustic window (0.256 s)
UTM = "EPSG:32632"            # metric CRS used for distances (Niger Delta, ~6-9 E)
KN = 1.852 / 60.0             # km per minute for one knot
STEP_PER_TICK_MIN = 1         # 1 tick = 1 minute

# physical-footprint constants (km) used by the signal models AND the label definitions
LAMBDA_P = 8.0                # pressure-wave attenuation length
LAMBDA_A = 0.6                # acoustic attenuation length (drilling)
LAMBDA_S = 1.5                # hydrocarbon plume attenuation length
R_P, R_A, R_C, R_E = 6.0, 1.5, 4.0, 3.0   # label influence radii (pressure, acoustic, chemical, env.)
CONC_DETECT = 0.3             # ppm: plume concentration above which y_chemical = 1

EVENT_TYPES = ["none", "leak", "hot_tap", "cold_tap", "spill", "env_disturbance"]
FAULT_TYPES = ["none", "dropout", "stuck", "drift", "spike"]

# ═════════════════════════════════════════════════════════════════════════════
# 1. SCENARIO REGISTRY  (single source of truth for the data AND the Chapter 3 text)
# ═════════════════════════════════════════════════════════════════════════════
DEFAULTS = dict(
    nodes=[10], spacing_km=1.5, min_corridor_km=8.0, events=[],
    traffic={"cargo_transit": (2, 5), "fishing": (3, 8), "licensed_barge": (0, 1)},
    noise_mult=1.0, rain_prob=0.25, rain_max=0.8, fault_prob=0.0,
    op_transients=(0, 0), ais_dark_boost=1.0, runs_factor=1.0,
    concurrent_window=None, sequential=False,
)


def S(**kw):
    d = json.loads(json.dumps(DEFAULTS))
    d["traffic"] = {k: tuple(v) for k, v in d["traffic"].items()}
    d.update(kw)
    return d


SCENARIOS = [
    S(id="S01", name="Baseline normal operations", category="Baseline",
      objective="Measure the false alarm rate (FAR) of every detector when nothing is wrong.",
      description="Pipeline operating normally. Legal traffic only (cargo transits, fishing boats, "
                  "an occasional licensed bunker barge). Normal diurnal pressure cycle and tidal turbidity.",
      events=[]),
    S(id="S02", name="Single pipeline leak", category="Pipeline integrity",
      objective="Evaluate pressure-drop detection accuracy, localisation and response time.",
      description="One accidental or corrosion-induced leak at a random point of the monitored section. "
                  "Sudden pressure drop that partially recovers to a sustained lower level until repaired.",
      events=[dict(type="leak", count=(1, 1), duration=(40, 180), severity=(0.3, 1.0), night_bias=0.3)]),
    S(id="S03", name="Hot-tap illegal bunkering (full attack chain)", category="Illegal bunkering",
      objective="Evaluate multi-sensor fusion: vessel behaviour + drilling acoustics + pressure drop.",
      description="A small boat (Cotonou-type) approaches, loiters at the pipeline and drills a live line. "
                  "Drilling acoustics first, then product diversion causes a sustained pressure drop. "
                  "Sediment disturbance raises local turbidity. AIS is frequently switched off.",
      events=[dict(type="hot_tap", count=(1, 1), duration=(60, 240), severity=(0.3, 1.0),
                   night_bias=0.7, vessel="loitering_tapper")]),
    S(id="S04", name="Cold-tap illegal bunkering", category="Illegal bunkering",
      objective="Test detection when the pressure signature is absent (acoustic + vessel only).",
      description="Tapping of a depressurised line (during shut-in). Drilling acoustics and a loitering "
                  "vessel are present but there is no pressure anomaly - a hard case for pressure-based systems.",
      events=[dict(type="cold_tap", count=(1, 1), duration=(45, 150), severity=(0.3, 1.0),
                   night_bias=0.7, vessel="loitering_tapper")]),
    S(id="S05", name="Oil spill from a vandalised pipeline", category="Environmental",
      objective="Evaluate spill detection, plume tracking and downstream alert delay.",
      description="A rupture releases crude. Pressure drops (coupled leak) and a hydrocarbon plume "
                  "advects downstream with the tidal/river current, arriving at downstream sensors with a delay.",
      events=[dict(type="spill", count=(1, 1), duration=(90, 300), severity=(0.3, 1.0),
                   night_bias=0.5, coupled_leak=0.8)]),
    S(id="S06", name="Simultaneous multi-site attacks", category="Illegal bunkering",
      objective="Stress-test agent coordination and resource allocation (drones, patrols).",
      description="A hot tap, a cold tap, a leak and a spill start within about an hour of each other at "
                  "different points of a long monitored section, competing for limited response assets.",
      nodes=[20], min_corridor_km=20.0, concurrent_window=60,
      events=[dict(type="hot_tap", count=(1, 1), duration=(60, 200), severity=(0.4, 1.0), night_bias=0.7,
                   vessel="loitering_tapper"),
              dict(type="cold_tap", count=(1, 1), duration=(60, 160), severity=(0.4, 1.0), night_bias=0.7,
                   vessel="loitering_tapper"),
              dict(type="leak", count=(1, 1), duration=(60, 180), severity=(0.4, 1.0), night_bias=0.5),
              dict(type="spill", count=(1, 1), duration=(90, 240), severity=(0.4, 1.0), night_bias=0.5,
                   coupled_leak=0.5)]),
    S(id="S07", name="Night-time covert hit-and-run tapping", category="Illegal bunkering",
      objective="Evaluate detection of short, covert attacks under reduced visibility.",
      description="Fast boats arrive in the dark, tap for 30-90 minutes and leave. AIS is usually off. "
                  "Models the situation where optical confirmation by UAV is degraded at night.",
      ais_dark_boost=1.5,
      events=[dict(type="hot_tap", count=(1, 1), duration=(30, 90), severity=(0.4, 1.0), night_bias=1.0,
                   vessel="hit_and_run")]),
    S(id="S08", name="Adverse weather and high sea state", category="Environmental stress",
      objective="Measure robustness to noise: storms, rain and rough water raise sensor noise.",
      description="Heavy rain/storm for much of the run, doubled sensor noise, high turbidity. "
                  "Each run contains a hot tap or a leak with probability 0.5 (otherwise it is a pure "
                  "false-alarm test under bad weather).",
      noise_mult=2.0, rain_prob=1.0, rain_max=1.0,
      events=[dict(type="hot_tap", p=0.5, count=(1, 1), duration=(60, 200), severity=(0.4, 1.0),
                   night_bias=0.5, vessel="loitering_tapper"),
              dict(type="leak", p=0.5, count=(1, 1), duration=(40, 160), severity=(0.4, 1.0), night_bias=0.4)]),
    S(id="S09", name="Sensor degradation and faults", category="Robustness",
      objective="Test fault tolerance: dropouts, stuck sensors, drift and spikes.",
      description="30% of sensor nodes develop a fault (communication dropout, stuck value, calibration "
                  "drift or spikes) while a hot tap or leak occurs with probability 0.6. Faults are "
                  "recorded in sensor_ok / fault_type but are NOT labelled as threats.",
      fault_prob=0.3,
      events=[dict(type="hot_tap", p=0.6, count=(1, 1), duration=(60, 200), severity=(0.4, 1.0),
                   night_bias=0.5, vessel="loitering_tapper"),
              dict(type="leak", p=0.6, count=(1, 1), duration=(40, 160), severity=(0.4, 1.0), night_bias=0.4)]),
    S(id="S10", name="Legitimate operations (hard negatives)", category="Baseline",
      objective="Quantify false alarms caused by lawful look-alike activity.",
      description="Licensed bunker barges work close to the pipeline for hours (slow, loitering, engine "
                  "noise) and operators perform valve/pump actions that cause pressure transients. "
                  "No threat exists, so every alert is a false alarm.",
      traffic={"cargo_transit": (2, 4), "fishing": (3, 6), "licensed_barge": (3, 5)},
      op_transients=(2, 4), events=[]),
    S(id="S11", name="Mother-ship supported multi-tap operation", category="Illegal bunkering",
      objective="Evaluate detection of organised, sequential tapping supported by a mother ship.",
      description="An anchored mother ship collects product while skiffs shuttle between it and 2-3 tap "
                  "points along the line, one after another during the night.",
      sequential=True,
      events=[dict(type="hot_tap", count=(2, 3), duration=(60, 150), severity=(0.4, 1.0), night_bias=0.8,
                   vessel="mother_ship")]),
    S(id="S12", name="Scalability stress test", category="Scalability",
      objective="Measure how latency/accuracy scale with the number of agents (sensor nodes and vessels).",
      description="The same hot-tap/leak situation is simulated with 5, 10, 20, 50 and 100 sensor nodes "
                  "and 3-5x denser vessel traffic (node spacing 0.5 km, shorter if the segment is short).",
      nodes=[5, 10, 20, 50, 100], spacing_km=0.5, min_corridor_km=10.0, runs_factor=0.5,
      traffic={"cargo_transit": (5, 15), "fishing": (10, 40), "licensed_barge": (1, 4)},
      events=[dict(type="hot_tap", count=(1, 1), duration=(60, 200), severity=(0.4, 1.0), night_bias=0.6,
                   vessel="loitering_tapper"),
              dict(type="leak", p=0.5, count=(1, 1), duration=(40, 160), severity=(0.4, 1.0), night_bias=0.4)]),
    S(id="S13", name="Non-malicious environmental disturbance", category="Environmental",
      objective="Test discrimination of natural/industrial disturbance (dredging, anchor dragging) from attacks.",
      description="Sediment disturbance from dredging or anchor dragging raises turbidity and low-frequency "
                  "vibration near the pipeline without any tapping. Frequent rain adds natural turbidity.",
      rain_prob=0.6, traffic={"cargo_transit": (2, 5), "fishing": (3, 8), "licensed_barge": (1, 3)},
      events=[dict(type="env_disturbance", count=(1, 2), duration=(30, 150), severity=(0.3, 1.0),
                   night_bias=0.3)]),
]
SC_INDEX = {s["id"]: i for i, s in enumerate(SCENARIOS)}

# ═════════════════════════════════════════════════════════════════════════════
# 2. GEOMETRY
# ═════════════════════════════════════════════════════════════════════════════
class Corridor:
    """A pipeline segment as a metric polyline with chainage (km along the line)."""

    def __init__(self, pid, xy_m, meta, to_ll):
        self.pid, self.meta, self.to_ll = pid, meta, to_ll
        xy_m = np.asarray(xy_m, float)
        keep = np.r_[True, np.hypot(*np.diff(xy_m, axis=0).T) > 0.5]
        xy_m = xy_m[keep]
        self.xs, self.ys = xy_m[:, 0], xy_m[:, 1]
        self.cum = np.r_[0.0, np.cumsum(np.hypot(np.diff(self.xs), np.diff(self.ys)))]
        self.length_km = self.cum[-1] / 1000.0

    def xy(self, s_km, y_km=0.0):
        s_m = np.asarray(s_km, float) * 1000.0
        sc = np.clip(s_m, 0, self.cum[-1])
        x = np.interp(sc, self.cum, self.xs)
        y = np.interp(sc, self.cum, self.ys)
        s1, s2 = np.clip(sc - 50, 0, self.cum[-1]), np.clip(sc + 50, 0, self.cum[-1])
        tx = np.interp(s2, self.cum, self.xs) - np.interp(s1, self.cum, self.xs)
        ty = np.interp(s2, self.cum, self.ys) - np.interp(s1, self.cum, self.ys)
        nrm = np.hypot(tx, ty)
        nrm = np.where(nrm == 0, 1.0, nrm)
        tx, ty = tx / nrm, ty / nrm
        ex = s_m - sc                                  # linear extrapolation past the ends
        x, y = x + ex * tx, y + ex * ty
        yy = np.asarray(y_km, float) * 1000.0
        return x - yy * ty, y + yy * tx                # left normal = (-ty, tx)

    def latlon(self, s_km, y_km=0.0):
        X, Y = self.xy(s_km, y_km)
        lon, lat = self.to_ll.transform(X, Y)
        return np.asarray(lat), np.asarray(lon)


def load_corridors(shp_path, min_km=3.0):
    from pyproj import Transformer
    to_utm = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)
    to_ll = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)
    cors, src = [], "synthetic fallback corridors"
    if shp_path and os.path.exists(shp_path):
        import geopandas as gpd
        from shapely.ops import linemerge
        gdf = gpd.read_file(shp_path)
        gdf = gdf[gdf.geometry.notna()]
        if gdf.crs is not None and str(gdf.crs).upper() not in ("EPSG:4326",):
            gdf = gdf.to_crs("EPSG:4326")
        src = f"OpenStreetMap shapefile: {os.path.basename(shp_path)}"
        for i, (_, row) in enumerate(gdf.iterrows()):
            geom = row.geometry
            if geom.geom_type == "MultiLineString":
                geom = linemerge(geom)
                if geom.geom_type == "MultiLineString":
                    geom = max(geom.geoms, key=lambda g: g.length)
            if geom.geom_type != "LineString":
                continue
            ll = np.asarray(geom.coords)[:, :2]
            X, Y = to_utm.transform(ll[:, 0], ll[:, 1])
            c = Corridor(str(row.get("id", f"seg{i}")), np.c_[X, Y], {}, to_ll)
            if c.length_km >= min_km:
                def g(k):
                    v = row.get(k, None)
                    return "unknown" if v is None or (isinstance(v, float) and np.isnan(v)) or v == "" else str(v)
                c.meta = dict(operator=g("operator"), substance=g("substance"), location=g("location"),
                              name=g("name"))
                c.geom_ll = geom
                cors.append(c)
    if not cors:   # synthetic fallback: five straight-ish corridors near Bonny
        rng = np.random.default_rng(1)
        for i in range(5):
            lon0, lat0 = 7.0 + 0.1 * i, 4.5 + 0.05 * i
            ang = rng.uniform(0, np.pi)
            n = 40
            t = np.linspace(0, 1, n)
            lon = lon0 + 0.35 * t * np.cos(ang) + 0.01 * np.sin(6 * t)
            lat = lat0 + 0.35 * t * np.sin(ang) + 0.01 * np.cos(5 * t)
            X, Y = to_utm.transform(lon, lat)
            c = Corridor(f"synthetic/{i}", np.c_[X, Y],
                         dict(operator="synthetic", substance="oil", location="underwater", name=f"synthetic {i}"),
                         to_ll)
            from shapely.geometry import LineString
            c.geom_ll = LineString(np.c_[lon, lat])
            cors.append(c)
    return cors, src


# ═════════════════════════════════════════════════════════════════════════════
# 3. SIMULATION ENGINE
# ═════════════════════════════════════════════════════════════════════════════
def ar1(rng, T, theta, sigma):
    return lfilter([1.0], [1.0, -(1.0 - theta)], rng.normal(0.0, sigma, T))


class Sig:
    """All physical contributions of events, shape (T, N)."""

    def __init__(self, T, N):
        z = lambda dt=float: np.zeros((T, N), dt)
        self.dP, self.a_drill, self.a_low, self.conc, self.dturb = z(), z(), z(), z(), z()
        self.y_p, self.y_a, self.y_c, self.y_e = z(bool), z(bool), z(bool), z(bool)
        self.ev = np.zeros((T, N), np.int16)


def add_tail(arr, e, end_val, tau, T, span=150):
    k = np.arange(1, min(span, T - 1 - e) + 1)
    if k.size:
        arr[e + 1:e + 1 + k.size] += end_val[None, :] * np.exp(-k / tau)[:, None]


def pick_start(rng, T, dur, night_bias, is_night, lo=90, margin=45):
    hi = T - dur - margin
    lo = min(lo, max(5, hi - 1))
    if rng.random() < night_bias:
        cand = np.flatnonzero(is_night[lo:hi]) + lo
        if cand.size:
            return int(rng.choice(cand))
    return int(rng.integers(lo, max(lo + 1, hi)))


def plan_events(sc, rng, T, is_night, s0, L_mon, run_id):
    events, anchor, prev_end = [], None, None
    for spec in sc["events"]:
        if rng.random() > spec.get("p", 1.0):
            continue
        cnt = int(rng.integers(spec["count"][0], spec["count"][1] + 1))
        for _ in range(cnt):
            dur = int(rng.integers(spec["duration"][0], spec["duration"][1] + 1))
            if sc["sequential"] and prev_end is not None:
                start = prev_end + int(rng.integers(10, 61))
                if start + dur > T - 45:
                    break
            elif sc["concurrent_window"] and anchor is not None:
                start = int(np.clip(anchor + rng.integers(-sc["concurrent_window"], sc["concurrent_window"] + 1),
                                    60, T - dur - 45))
            else:
                start = pick_start(rng, T, dur, spec.get("night_bias", 0.3), is_night)
            if anchor is None:
                anchor = start
            lo, hi = 0.3, max(0.31, L_mon - 0.3)
            s_ev = s0 + rng.uniform(lo, hi)
            for _try in range(25):                       # keep events >= 3 km apart if possible
                if all(abs(s_ev - e["s_km"]) >= 3.0 for e in events) or L_mon < 6:
                    break
                s_ev = s0 + rng.uniform(lo, hi)
            sev = float(rng.uniform(*spec["severity"]))
            ev = dict(type=spec["type"], start=start, end=start + dur - 1, s_km=float(s_ev), severity=sev,
                      vessel=spec.get("vessel"), params={})
            t = ev["type"]
            if t == "leak":
                ev["params"] = dict(D=float(rng.uniform(3, 15) * (0.4 + 0.6 * sev)), tau=float(rng.uniform(20, 60)),
                                    r=float(rng.uniform(0.3, 0.6)))
            elif t == "hot_tap":
                ev["params"] = dict(drill_min=int(rng.integers(15, 51)), D=float(rng.uniform(0.8, 4.0) * sev),
                                    a_d=float(rng.uniform(1.2, 3.5)), tau_rec=15.0)
            elif t == "cold_tap":
                ev["params"] = dict(a_d=float(rng.uniform(1.2, 3.5)), duty=0.6)
            elif t == "spill":
                ev["params"] = dict(C0=float(rng.uniform(5, 30) * (0.4 + 0.6 * sev)),
                                    u=float(rng.uniform(0.01, 0.03)))
            elif t == "env_disturbance":
                ev["params"] = dict(kind=str(rng.choice(["dredging", "anchor_drag"])),
                                    dturb=float(rng.uniform(30, 120) * sev), a_low=float(rng.uniform(0.5, 1.5) * sev))
            events.append(ev)
            prev_end = ev["end"]
            if t == "spill" and rng.random() < spec.get("coupled_leak", 0.0):
                leak = dict(type="leak", start=start, end=ev["end"], s_km=ev["s_km"], severity=sev, vessel=None,
                            parent_idx=len(events) - 1,
                            params=dict(D=float(rng.uniform(3, 12) * (0.4 + 0.6 * sev)),
                                        tau=float(rng.uniform(20, 60)), r=float(rng.uniform(0.3, 0.6))))
                events.append(leak)
    for k, ev in enumerate(events, 1):
        ev["event_id"] = f"{run_id}_E{k:02d}"
        ev["code"] = k
    return events


def mark(sig, sl, mask, which, code):
    getattr(sig, which)[sl][mask] = True
    sig.ev[sl][mask] = code


def apply_event(ev, sig, nodes_s, T, cur_dir, rng):
    s, e = ev["start"], min(T - 1, ev["end"])
    sl = slice(s, e + 1)
    L = e - s + 1
    d = np.abs(nodes_s - ev["s_km"])
    tt = np.arange(L)[:, None]
    p, t = ev["params"], ev["type"]
    if t == "leak":
        prof = p["r"] + (1 - p["r"]) * np.exp(-tt / p["tau"])
        ramp = np.minimum(1.0, (tt + 1) / 3.0)
        amp = -p["D"] * np.exp(-d / LAMBDA_P)[None, :]
        sig.dP[sl] += amp * prof * ramp
        add_tail(sig.dP, e, (amp * prof[-1] * ramp[-1])[0], 10.0, T)
        mark(sig, sl, np.broadcast_to(d <= R_P, (L, len(d))), "y_p", ev["code"])
    elif t in ("hot_tap", "cold_tap"):
        on = rng.random(L) < (0.65 if t == "cold_tap" else 0.65)
        if t == "hot_tap":
            td = min(p["drill_min"], L - 1)
            on[td:] = rng.random(L - td) < 0.25
        a = p["a_d"] * np.exp(-d / LAMBDA_A)
        sig.a_drill[sl] += on[:, None] * a[None, :]
        sig.dturb[sl] += (rng.uniform(3, 12) * np.exp(-d / 1.0))[None, :] * np.minimum(1, (tt + 1) / 10.0)
        mark(sig, sl, np.broadcast_to(d <= R_A, (L, len(d))), "y_a", ev["code"])
        if t == "hot_tap":
            td = min(p["drill_min"], L - 1)
            ramp = np.minimum(1.0, (np.arange(L - td)[:, None] + 1) / 3.0)
            amp = -p["D"] * np.exp(-d / LAMBDA_P)[None, :]
            sig.dP[s + td:e + 1] += amp * ramp
            add_tail(sig.dP, e, (amp * ramp[-1])[0], p["tau_rec"], T)
            m = np.zeros((L, len(d)), bool)
            m[td:, :] = (d <= R_P)[None, :]
            mark(sig, sl, m, "y_p", ev["code"])
    elif t == "spill":
        tend = min(T - 1, e + 240)
        tv = np.arange(s, tend + 1)
        down = (nodes_s - ev["s_km"]) * cur_dir >= 0
        delay = np.where(down, d / p["u"], 1e9)
        tp = tv[:, None] - s - delay[None, :]
        h = np.where(tp > 0, 1 - np.exp(-np.clip(tp, 0, None) / 20.0), 0.0)
        src = np.where(tv <= e, 1.0, np.exp(-(tv - e) / 60.0))[:, None]
        att = p["C0"] * np.exp(-d / LAMBDA_S) * np.where(down, 1.0, 0.02)
        c = att[None, :] * h * src
        sig.conc[s:tend + 1] += c
        sig.dturb[s:tend + 1] += 0.1 * c
        mark(sig, slice(s, tend + 1), (c > CONC_DETECT) & (d <= R_C)[None, :], "y_c", ev["code"])
    elif t == "env_disturbance":
        ramp = np.minimum(1.0, (tt + 1) / 10.0)
        a = np.exp(-d / 2.0)
        sig.dturb[sl] += p["dturb"] * a[None, :] * ramp
        add_tail(sig.dturb, e, p["dturb"] * a * ramp[-1], 30.0, T)
        sig.a_low[sl] += p["a_low"] * np.exp(-d / R_E)[None, :]
        mark(sig, sl, np.broadcast_to(d <= R_E, (L, len(d))), "y_e", ev["code"])


# ── vessels ──────────────────────────────────────────────────────────────────
ENGINE_AMP = dict(cargo_transit=0.9, fishing=0.35, licensed_barge=0.5, loitering_tapper=0.3,
                  hit_and_run=0.5, mother_ship=0.7)
AIS_PARAMS = dict(cargo_transit=(0.0003, 0.3), fishing=(0.002, 0.1), licensed_barge=(0.0005, 0.3),
                  loitering_tapper=(0.02, 0.01), hit_and_run=(0.05, 0.005), mother_ship=(0.01, 0.01))
THREAT_CLASSES = ("loitering_tapper", "hit_and_run", "mother_ship")


def v_cargo(rng, T, lo, hi):
    spd = rng.uniform(8, 16) * KN
    dur = int((hi - lo) / spd) + 1
    t0 = int(rng.integers(-dur, T))
    tt = np.arange(T) - t0
    pres = (tt >= 0) & (tt <= dur)
    s = lo + tt * spd if rng.random() < 0.5 else hi - tt * spd
    y = rng.choice([-1, 1]) * rng.uniform(1.0, 12.0) + np.cumsum(rng.normal(0, 0.004, T))
    return s, y, pres


def v_ou(rng, T, s_c, y_c, theta, sigma, t_in, t_out):
    s = s_c + ar1(rng, T, theta, sigma)
    y = y_c + ar1(rng, T, theta, sigma)
    pres = np.zeros(T, bool)
    pres[max(0, t_in):min(T, t_out)] = True
    return s, y, pres


def v_waypoints(rng, T, wps, jitter=0.003):
    ts = np.array([w[0] for w in wps], float)
    for i in range(1, len(ts)):
        ts[i] = max(ts[i], ts[i - 1] + 1)
    tick = np.arange(T)
    s = np.interp(tick, ts, [w[1] for w in wps]) + ar1(rng, T, 0.05, jitter)
    y = np.interp(tick, ts, [w[2] for w in wps]) + ar1(rng, T, 0.05, jitter)
    return s, y, (tick >= ts[0]) & (tick <= ts[-1])


def threat_vessel(rng, T, ev, cls, origin=None):
    se = ev["s_km"]
    ty = rng.choice([-1, 1]) * rng.uniform(0.03, 0.12)
    if cls == "hit_and_run":
        appr, dist = int(rng.integers(12, 21)), rng.uniform(3, 5)
    else:
        appr, dist = int(rng.integers(15, 46)), rng.uniform(4, 8)
    if origin is None:
        far = (se + rng.choice([-1, 1]) * dist * rng.uniform(0.6, 1.0), rng.choice([-1, 1]) * dist * rng.uniform(0.3, 0.8))
        far2 = (se + rng.choice([-1, 1]) * dist * rng.uniform(0.6, 1.0), rng.choice([-1, 1]) * dist * rng.uniform(0.3, 0.8))
    else:
        far = far2 = origin
    wps = [(ev["start"] - appr, far[0], far[1]),
           (ev["start"], se + rng.normal(0, 0.04), ty),
           (ev["end"], se + rng.normal(0, 0.04), ty),
           (ev["end"] + int(rng.integers(10, 26)), far2[0], far2[1])]
    return v_waypoints(rng, T, wps)


def markov_ais(rng, T, p_off, p_on, start_on):
    u, out, state = rng.random(T), np.empty(T, bool), start_on
    for t in range(T):
        if state:
            if u[t] < p_off[t]:
                state = False
        elif u[t] < p_on:
            state = True
        out[t] = state
    return out


def build_vessels(sc, rng, T, events, s0, L_mon, is_night):
    V = []
    lo, hi = s0 - 8.0, s0 + L_mon + 8.0

    def add(cls, s, y, pres, threat=False, ev=None):
        V.append(dict(cls=cls, s=s, y=y, pres=pres, threat=threat, event_id=ev["event_id"] if ev else "none"))

    for cls, fn in (("cargo_transit", None), ("fishing", None), ("licensed_barge", None)):
        a, b = sc["traffic"][cls]
        for _ in range(int(rng.integers(a, b + 1))):
            if cls == "cargo_transit":
                add(cls, *v_cargo(rng, T, lo, hi))
            elif cls == "fishing":
                t_in = int(rng.integers(-60, T - 120)); t_out = t_in + int(rng.integers(120, 601))
                add(cls, *v_ou(rng, T, s0 + rng.uniform(-1, L_mon + 1), rng.choice([-1, 1]) * rng.uniform(0.3, 3.0),
                               0.01, 0.05, t_in, t_out))
            else:
                t_in = int(rng.integers(-60, T - 120)); t_out = t_in + int(rng.integers(120, 361))
                add(cls, *v_ou(rng, T, s0 + rng.uniform(0, L_mon), rng.choice([-1, 1]) * rng.uniform(0.15, 0.5),
                               0.02, 0.004, t_in, t_out))
    taps = [e for e in events if e["type"] in ("hot_tap", "cold_tap") and e["vessel"]]
    mother = None
    if any(e["vessel"] == "mother_ship" for e in taps):
        e0 = taps[0]
        ms = e0["s_km"] + rng.uniform(-1, 1)
        my = rng.choice([-1, 1]) * rng.uniform(1.5, 3.0)
        t_in, t_out = min(e["start"] for e in taps) - 60, max(e["end"] for e in taps) + 60
        s, y, pres = v_ou(rng, T, ms, my, 0.01, 0.003, t_in, t_out)
        add("mother_ship", s, y, pres, True, e0)
        mother = (ms, my)
    for ev in taps:
        cls = "loitering_tapper" if ev["vessel"] == "mother_ship" else ev["vessel"]
        add(cls, *threat_vessel(rng, T, ev, cls, origin=mother), threat=True, ev=ev)
    for v in V:
        p_off, p_on = AIS_PARAMS[v["cls"]]
        boost = sc["ais_dark_boost"] if v["threat"] else 1.0
        p_off_t = np.minimum(1.0, p_off * boost * np.where(is_night, 1.5, 1.0))
        start_on = not (v["cls"] == "hit_and_run" and rng.random() < 0.7)
        v["ais"] = markov_ais(rng, T, p_off_t, p_on, start_on)
    return V


# ── acoustics ────────────────────────────────────────────────────────────────
def synth_acoustic(rng, T, N, sig, a_traffic, w, nm):
    tau = (np.arange(WIN) / FS).astype(np.float32)
    wave = (rng.normal(0, 1, (T, N, WIN)) * (0.1 * nm * (1 + 1.5 * w))[:, None, None]).astype(np.float32)

    def add(freq, amp):
        if not np.any(amp):
            return
        ph = rng.uniform(0, 2 * np.pi, (T, N, 1)).astype(np.float32)
        f = np.broadcast_to(np.asarray(freq, np.float32), (T, N))[:, :, None]
        wave.__iadd__((amp[:, :, None] * np.sin(2 * np.pi * f * tau[None, None, :] + ph)).astype(np.float32))

    hum = rng.uniform(0.25, 0.5)
    for f, a in ((2.0, 1.0), (5.0, 0.6), (11.0, 0.3)):
        add(f, hum * a * (1 + 0.1 * rng.normal(0, 1, (T, N))))
    add(rng.uniform(25, 45, (T, N)), a_traffic)
    jit = rng.uniform(-1.5, 1.5, (T, N))
    add(50 + jit, sig.a_drill)
    add(100 + 2 * jit, 0.4 * sig.a_drill)
    add(rng.uniform(6, 15, (T, N)), sig.a_low)
    x = wave - wave.mean(axis=2, keepdims=True)
    rms, peak = x.std(axis=2), np.abs(x).max(axis=2)
    spec = np.abs(np.fft.rfft(x * np.hanning(WIN).astype(np.float32), axis=2)) ** 2
    tot = spec[:, :, 1:].sum(2) + 1e-12
    freqs = np.fft.rfftfreq(WIN, 1 / FS)
    dom = freqs[1 + spec[:, :, 1:].argmax(2)]
    return rms, peak, dom, spec[:, :, 9:17].sum(2) / tot, spec[:, :, 1:5].sum(2) / tot


# ── one full run ─────────────────────────────────────────────────────────────
def simulate_run(sc, sc_idx, run_idx, corridors, args):
    run_id = f"{sc['id']}_R{run_idx:04d}"
    rng = np.random.default_rng([args.seed, sc_idx, run_idx])
    T = args.ticks
    N = int(rng.choice(sc["nodes"]))
    spacing = sc["spacing_km"]
    elig = [c for c in corridors if c.length_km >= sc["min_corridor_km"]] or [max(corridors, key=lambda c: c.length_km)]
    cor = elig[int(rng.integers(len(elig)))]
    need = (N - 1) * spacing
    if cor.length_km < need + 0.6:
        spacing = max(0.05, (cor.length_km - 0.6) / (N - 1))
        need = (N - 1) * spacing
    s0 = float(rng.uniform(0.3, max(0.3, cor.length_km - need - 0.3)))
    nodes_s = s0 + np.arange(N) * spacing
    m_nodes = nodes_s - s0
    L_mon = float(need)

    start_dt = datetime(2026, 1, 1) + timedelta(days=int(rng.integers(0, 365)))
    rainy = 4 <= start_dt.month <= 10
    tick = np.arange(T)
    hod = (tick // 60) % 24
    is_night = (hod >= 19) | (hod < 6)
    nm = sc["noise_mult"]

    # weather
    w = np.zeros(T)
    if rng.random() < min(1.0, sc["rain_prob"] * (1.6 if rainy else 0.6)):
        for _ in range(int(rng.integers(1, 4))):
            d = int(rng.integers(30, 181)); st = int(rng.integers(0, T - d))
            w[st:st + d] = np.maximum(w[st:st + d], rng.uniform(0.3, 1.0) * sc["rain_max"] * np.sin(np.pi * np.arange(d) / d) ** 2)

    # pressure baseline: P(x,t) = P0 - k x - alpha t  (Chapter 3) + diurnal cycle + noise
    P0, k, alpha = rng.uniform(35, 70), rng.uniform(0.01, 0.04), rng.uniform(2e-4, 8e-4)
    common = lfilter([1.0], [1.0, -0.98], rng.normal(0, 0.03 * nm, T))
    diur = rng.uniform(0.2, 0.8) * np.sin(2 * np.pi * tick / 1440 + rng.uniform(0, 2 * np.pi))
    base_p = (P0 - k * m_nodes[None, :] - alpha * tick[:, None]
              + diur[:, None] * (1 - 0.01 * m_nodes)[None, :] + common[:, None])

    # events -> physical signals
    cur_dir = int(rng.choice([-1, 1]))
    events = plan_events(sc, rng, T, is_night, s0, L_mon, run_id)
    sig = Sig(T, N)
    for ev in events:
        apply_event(ev, sig, nodes_s, T, cur_dir, rng)
    for ev in events:                                         # coupled-leak parent linkage (id strings)
        ev["parent"] = events[ev["parent_idx"]]["event_id"] if "parent_idx" in ev else "none"

    # operational transients (benign pressure steps)
    op = np.zeros(T, bool)
    dP_op = np.zeros((T, N))
    for _ in range(int(rng.integers(sc["op_transients"][0], sc["op_transients"][1] + 1))):
        dur = int(rng.integers(15, 61)); st = int(rng.integers(60, T - dur - 30))
        mag = rng.choice([-1, 1]) * rng.uniform(1.0, 3.0)
        prof = mag * np.minimum(1.0, (np.arange(dur) + 1) / 5.0)
        att = np.exp(-m_nodes / 60.0)
        dP_op[st:st + dur] += prof[:, None] * att[None, :]
        add_tail(dP_op, st + dur - 1, prof[-1] * att, 8.0, T, span=60)
        op[st:st + dur + 20] = True

    # vessels + engine noise
    V = build_vessels(sc, rng, T, events, s0, L_mon, is_night)
    a_traffic = np.zeros((T, N))
    for v in V:
        spd = np.hypot(np.gradient(v["s"]), np.gradient(v["y"])) / KN
        amp = ENGINE_AMP[v["cls"]] * (0.5 + np.minimum(spd, 15) / 15.0) * v["pres"]
        dist = np.hypot(v["s"][:, None] - nodes_s[None, :], v["y"][:, None])
        a_traffic += amp[:, None] * np.exp(-dist / 0.8)

    # acoustic features from synthetic waveform windows
    rms, peak, dom, b_drill, b_low = synth_acoustic(rng, T, N, sig, a_traffic, w, nm)

    # chemical / environmental channels
    hc = rng.lognormal(np.log(0.08), 0.5, (T, N)) * (1 + 0.5 * w[:, None]) + sig.conc
    temp = (27.5 + rng.normal(0, 0.5) + 0.9 * np.sin(2 * np.pi * (tick / 60 - 9) / 24))[:, None] \
        + rng.normal(0, 0.15, N)[None, :] + rng.normal(0, 0.05 * nm, (T, N))
    turb = (rng.uniform(15, 35) + 5 * np.sin(2 * np.pi * tick / 745) + 60 * w)[:, None] \
        + rng.normal(0, 2, N)[None, :] + rng.normal(0, 1.2 * nm, (T, N)) + sig.dturb
    turb = np.clip(turb, 0, None)
    pressure = base_p + sig.dP + dP_op + rng.normal(0, 0.12 * nm, (T, N))

    # sensor faults (do not change threat labels)
    fcode = np.zeros((T, N), np.int8)
    ok = np.ones((T, N), np.int8)
    channels = dict(pressure=pressure, rms=rms, peak=peak, dom=dom, b_drill=b_drill, b_low=b_low,
                    hc=hc, temp=temp, turb=turb)
    n_faulty = 0
    for n in range(N):
        if rng.random() < sc["fault_prob"]:
            n_faulty += 1
            ft = str(rng.choice(["dropout", "stuck", "drift", "spike"]))
            a = int(rng.integers(60, T - 200)); b = a + int(rng.integers(30, 180))
            if ft == "dropout":
                for arr in channels.values():
                    arr[a:b, n] = np.nan
                fcode[a:b, n], ok[a:b, n] = 1, 0
            elif ft == "stuck":
                for key in ("pressure", "hc", "turb"):
                    channels[key][a:b, n] = channels[key][a - 1, n]
                fcode[a:b, n], ok[a:b, n] = 2, 0
            elif ft == "drift":
                channels["pressure"][a:b, n] += np.linspace(0, rng.choice([-1, 1]) * rng.uniform(0.8, 2.5), b - a)
                fcode[a:b, n], ok[a:b, n] = 3, 0
            else:
                idx = rng.integers(a, b, int(rng.integers(3, 9)))
                channels["pressure"][idx, n] += rng.choice([-1, 1], idx.size) * rng.uniform(5, 15, idx.size)
                fcode[idx, n], ok[idx, n] = 4, 0

    # ── tables ───────────────────────────────────────────────────────────────
    ts = np.datetime64(start_dt, "s") + (tick * 60).astype("timedelta64[s]")
    rows = T * N
    y_any = sig.y_p | sig.y_a | sig.y_c | sig.y_e
    ev_ids = np.array(["none"] + [e["event_id"] for e in events])
    ev_types = np.array(["none"] + [e["type"] for e in events])
    f32 = lambda a: np.asarray(a, np.float32).ravel()
    sensors = pd.DataFrame({
        "run_id": run_id, "scenario_id": sc["id"],
        "tick": np.repeat(tick, N).astype(np.int32), "timestamp": np.repeat(ts, N),
        "node_id": np.tile(np.arange(N), T).astype(np.int16),
        "pressure_bar": f32(channels["pressure"]), "vib_rms": f32(channels["rms"]),
        "vib_peak": f32(channels["peak"]), "vib_dom_freq_hz": f32(channels["dom"]),
        "vib_band_drill": f32(channels["b_drill"]), "vib_band_low": f32(channels["b_low"]),
        "hydrocarbon_ppm": f32(channels["hc"]), "water_temp_c": f32(channels["temp"]),
        "turbidity_ntu": f32(channels["turb"]),
        "sensor_ok": ok.ravel(), "fault_type": pd.Categorical.from_codes(fcode.ravel(), FAULT_TYPES),
        "op_transient": np.repeat(op, N).astype(np.int8),
        "y_pressure": sig.y_p.ravel().astype(np.int8), "y_acoustic": sig.y_a.ravel().astype(np.int8),
        "y_chemical": sig.y_c.ravel().astype(np.int8), "y_env": sig.y_e.ravel().astype(np.int8),
        "y_any": y_any.ravel().astype(np.int8),
        "event_id": ev_ids[sig.ev.ravel()], "event_type": pd.Categorical(ev_types[sig.ev.ravel()], categories=EVENT_TYPES),
    })

    lat_n, lon_n = cor.latlon(nodes_s)
    nodes = pd.DataFrame({"run_id": run_id, "node_id": np.arange(N).astype(np.int16), "latitude": lat_n,
                          "longitude": lon_n, "chainage_km": nodes_s, "monitored_pos_km": m_nodes,
                          "has_fault": [int(np.any(fcode[:, n] > 0)) for n in range(N)]})

    # vessel tracks sampled at the AIS reporting interval
    trk = []
    for vi, v in enumerate(V):
        pres_idx = np.flatnonzero(v["pres"])
        if pres_idx.size < 2:
            continue
        X, Y = cor.xy(v["s"], v["y"])
        X = X + rng.normal(0, 6.0, T); Y = Y + rng.normal(0, 6.0, T)      # GPS/radar jitter (m)
        dx, dy = np.diff(X, prepend=X[0]), np.diff(Y, prepend=Y[0])
        spd = np.hypot(dx, dy) / 60.0 / 0.514444
        hdg = (np.degrees(np.arctan2(dx, dy)) + 360) % 360
        sel = pres_idx[(pres_idx % args.ais_interval) == 0]
        if sel.size == 0:
            continue
        lon, lat = cor.to_ll.transform(X[sel], Y[sel])
        trk.append(pd.DataFrame({
            "run_id": run_id, "scenario_id": sc["id"], "vessel_id": f"V{vi:03d}", "tick": sel.astype(np.int32),
            "timestamp": ts[sel], "latitude": lat, "longitude": lon, "chainage_km": v["s"][sel],
            "cross_track_km": v["y"][sel], "speed_kn": spd[sel], "heading_deg": hdg[sel],
            "ais_transmitting": v["ais"][sel].astype(np.int8), "vessel_class": v["cls"],
            "is_threat_vessel": int(v["threat"]), "event_id": v["event_id"]}))
    tracks = pd.concat(trk, ignore_index=True) if trk else pd.DataFrame(columns=[
        "run_id", "scenario_id", "vessel_id", "tick", "timestamp", "latitude", "longitude", "chainage_km",
        "cross_track_km", "speed_kn", "heading_deg", "ais_transmitting", "vessel_class", "is_threat_vessel", "event_id"])

    ev_rows = []
    for ev in events:
        lat, lon = cor.latlon(ev["s_km"])
        nearest = int(np.argmin(np.abs(nodes_s - ev["s_km"])))
        vids = [f"V{i:03d}" for i, v in enumerate(V) if v["event_id"] == ev["event_id"]]
        ev_rows.append(dict(
            run_id=run_id, scenario_id=sc["id"], event_id=ev["event_id"], event_type=ev["type"],
            parent_event_id=ev.get("parent", "none"), start_tick=ev["start"], end_tick=min(T - 1, ev["end"]),
            duration_min=min(T - 1, ev["end"]) - ev["start"] + 1,
            start_time=str(np.datetime64(start_dt, "s") + np.timedelta64(ev["start"] * 60, "s")).replace("T", " "),
            starts_at_night=int(is_night[ev["start"]]), chainage_km=round(ev["s_km"], 4),
            latitude=float(lat), longitude=float(lon), severity=round(ev["severity"], 3), nearest_node=nearest,
            threat_vessel_ids=";".join(vids) if vids else "none", parameters=json.dumps(
                {k: (round(x, 4) if isinstance(x, float) else x) for k, x in ev["params"].items()})))
    events_df = pd.DataFrame(ev_rows, columns=[
        "run_id", "scenario_id", "event_id", "event_type", "parent_event_id", "start_tick", "end_tick",
        "duration_min", "start_time", "starts_at_night", "chainage_km", "latitude", "longitude", "severity",
        "nearest_node", "threat_vessel_ids", "parameters"])

    runs = pd.DataFrame([dict(
        run_id=run_id, scenario_id=sc["id"], seed=f"{args.seed}-{sc_idx}-{run_idx}", pipeline_id=cor.pid,
        operator=cor.meta.get("operator", "unknown"), substance=cor.meta.get("substance", "unknown"),
        location_type=cor.meta.get("location", "unknown"), corridor_length_km=round(cor.length_km, 3),
        monitored_start_km=round(s0, 3), monitored_length_km=round(L_mon, 3), n_nodes=N,
        node_spacing_km=round(spacing, 4), n_ticks=T, start_time=str(start_dt), rainy_season=int(rainy),
        noise_mult=nm, rain_fraction=round(float((w > 0.05).mean()), 4), current_direction=cur_dir,
        n_events=len(events), n_vessels=len(V), n_threat_vessels=sum(v["threat"] for v in V),
        n_faulty_nodes=n_faulty, P0_bar=round(P0, 3), k_bar_per_km=round(k, 4), alpha_bar_per_min=round(alpha, 6))])

    stats = dict(scenario_id=sc["id"], run_id=run_id, sensor_rows=rows, track_rows=len(tracks),
                 n_events=len(events), pos_any=int(y_any.sum()), pos_pressure=int(sig.y_p.sum()),
                 pos_acoustic=int(sig.y_a.sum()), pos_chemical=int(sig.y_c.sum()), pos_env=int(sig.y_e.sum()),
                 **{f"ev_{t}": sum(e["type"] == t for e in events) for t in EVENT_TYPES[1:]})
    geom = cor
    return dict(sensors=sensors, nodes=nodes, tracks=tracks, events=events_df, runs=runs, stats=stats, cor=geom)


# ═════════════════════════════════════════════════════════════════════════════
# 4. EXPORT
# ═════════════════════════════════════════════════════════════════════════════
DICTIONARY = {
    "sensor_readings": [
        ("run_id", "str", "-", "Unique run identifier <scenario>_R<nnnn>; joins to runs/nodes/events/vessel_tracks"),
        ("scenario_id", "str", "-", "Scenario code S01..S13 (see scenario_catalog)"),
        ("tick", "int", "min", "Simulation minute since run start (1 tick = 1 minute)"),
        ("timestamp", "datetime", "WAT", "Simulated local time of the reading"),
        ("node_id", "int", "-", "Sensor node index along the monitored section (0 = upstream end); joins to nodes"),
        ("pressure_bar", "float", "bar", "Line pressure. Model P(x,t)=P0-kx-alpha*t + diurnal cycle + noise + event effects"),
        ("vib_rms", "float", "a.u.", "RMS of the 128-sample (0.256 s, 500 Hz) geophone/hydrophone window"),
        ("vib_peak", "float", "a.u.", "Peak absolute amplitude of the window"),
        ("vib_dom_freq_hz", "float", "Hz", "Dominant spectral frequency of the window (excluding DC)"),
        ("vib_band_drill", "float", "fraction", "Spectral energy fraction in 35-62.5 Hz (drilling band)"),
        ("vib_band_low", "float", "fraction", "Spectral energy fraction in 3.9-15.6 Hz (flow/hum/anchor-drag band)"),
        ("hydrocarbon_ppm", "float", "ppm", "Dissolved/surface hydrocarbon concentration"),
        ("water_temp_c", "float", "deg C", "Water temperature"),
        ("turbidity_ntu", "float", "NTU", "Water turbidity"),
        ("sensor_ok", "int", "0/1", "0 while the node is in a fault block (dropout/stuck/drift) or at a spike tick"),
        ("fault_type", "category", "-", "none | dropout | stuck | drift | spike (sensor faults are NOT threats)"),
        ("op_transient", "int", "0/1", "1 while a lawful operational pressure transient (valve/pump) is active - hard negative"),
        ("y_pressure", "int", "0/1", f"Ground truth: pressure anomaly footprint (node within {R_P} km of a leak/hot-tap pressure effect)"),
        ("y_acoustic", "int", "0/1", f"Ground truth: drilling/tapping acoustic activity within {R_A} km"),
        ("y_chemical", "int", "0/1", f"Ground truth: plume concentration > {CONC_DETECT} ppm within {R_C} km (downstream)"),
        ("y_env", "int", "0/1", f"Ground truth: non-malicious environmental disturbance within {R_E} km"),
        ("y_any", "int", "0/1", "OR of the four labels above"),
        ("event_id", "str", "-", "Most recently started active event affecting this node-tick, else 'none'"),
        ("event_type", "category", "-", "none | leak | hot_tap | cold_tap | spill | env_disturbance"),
    ],
    "vessel_tracks": [
        ("run_id", "str", "-", "Run identifier"), ("scenario_id", "str", "-", "Scenario code"),
        ("vessel_id", "str", "-", "Vessel identifier unique within the run"),
        ("tick", "int", "min", "Simulation minute of the report (every --ais-interval minutes)"),
        ("timestamp", "datetime", "WAT", "Simulated local time"),
        ("latitude", "float", "deg", "WGS84 latitude (AIS/radar-fused position incl. ~6 m jitter)"),
        ("longitude", "float", "deg", "WGS84 longitude"),
        ("chainage_km", "float", "km", "Position projected along the pipeline segment"),
        ("cross_track_km", "float", "km", "Signed distance from the pipeline (left positive)"),
        ("speed_kn", "float", "knots", "Speed over ground from successive positions"),
        ("heading_deg", "float", "deg", "Course over ground (0 = north)"),
        ("ais_transmitting", "int", "0/1", "0 = AIS switched off ('dark'); position then comes from radar fusion"),
        ("vessel_class", "category", "-", "cargo_transit | fishing | licensed_barge | loitering_tapper | hit_and_run | mother_ship"),
        ("is_threat_vessel", "int", "0/1", "Ground truth: vessel participates in illegal tapping"),
        ("event_id", "str", "-", "Tapping event the vessel is linked to, else 'none'"),
    ],
    "events": [
        ("run_id", "str", "-", "Run identifier"), ("scenario_id", "str", "-", "Scenario code"),
        ("event_id", "str", "-", "Event identifier"), ("event_type", "category", "-", "leak | hot_tap | cold_tap | spill | env_disturbance"),
        ("parent_event_id", "str", "-", "For a leak coupled to a spill: the spill event; else 'none'"),
        ("start_tick", "int", "min", "First active minute"), ("end_tick", "int", "min", "Last active minute (source stops)"),
        ("duration_min", "int", "min", "Event duration"), ("start_time", "datetime", "WAT", "Simulated start time"),
        ("starts_at_night", "int", "0/1", "1 if start is between 19:00 and 06:00"),
        ("chainage_km", "float", "km", "Event location along the pipeline segment"),
        ("latitude", "float", "deg", "Event latitude"), ("longitude", "float", "deg", "Event longitude"),
        ("severity", "float", "0-1", "Relative severity"), ("nearest_node", "int", "-", "Index of the closest sensor node"),
        ("threat_vessel_ids", "str", "-", "Semicolon-separated vessel_ids linked to the event"),
        ("parameters", "json", "-", "Physical parameters used for the event (D, tau, C0, ...)"),
    ],
    "runs": [
        ("run_id", "str", "-", "Run identifier"), ("scenario_id", "str", "-", "Scenario code"),
        ("seed", "str", "-", "<global seed>-<scenario index>-<run index>: reproduces the run"),
        ("pipeline_id", "str", "-", "OpenStreetMap way id of the pipeline segment used"),
        ("operator", "str", "-", "Operator tag from OSM (may be 'unknown')"), ("substance", "str", "-", "Substance tag from OSM"),
        ("location_type", "str", "-", "underground | overground | underwater | ... from OSM"),
        ("corridor_length_km", "float", "km", "Length of the pipeline segment"),
        ("monitored_start_km", "float", "km", "Chainage where the sensor array starts"),
        ("monitored_length_km", "float", "km", "Length of the instrumented section"),
        ("n_nodes", "int", "-", "Number of sensor nodes"), ("node_spacing_km", "float", "km", "Spacing between nodes"),
        ("n_ticks", "int", "min", "Run length"), ("start_time", "datetime", "WAT", "Simulated start"),
        ("rainy_season", "int", "0/1", "April-October"), ("noise_mult", "float", "-", "Sensor noise multiplier"),
        ("rain_fraction", "float", "-", "Fraction of time with rain/storm"), ("current_direction", "int", "-", "+1/-1 water current along increasing chainage"),
        ("n_events", "int", "-", "Number of ground-truth events"), ("n_vessels", "int", "-", "Vessels simulated"),
        ("n_threat_vessels", "int", "-", "Illegal vessels simulated"), ("n_faulty_nodes", "int", "-", "Nodes with a sensor fault"),
        ("P0_bar", "float", "bar", "Pressure model parameter P0"), ("k_bar_per_km", "float", "bar/km", "Friction-loss parameter k"),
        ("alpha_bar_per_min", "float", "bar/min", "Decay parameter alpha"),
    ],
    "nodes": [
        ("run_id", "str", "-", "Run identifier"), ("node_id", "int", "-", "Node index"),
        ("latitude", "float", "deg", "Node latitude (on the real pipeline geometry)"), ("longitude", "float", "deg", "Node longitude"),
        ("chainage_km", "float", "km", "Absolute chainage on the segment"), ("monitored_pos_km", "float", "km", "Distance from start of monitored section"),
        ("has_fault", "int", "0/1", "Node had a sensor fault during the run"),
    ],
}
ROUND = dict(pressure_bar=4, vib_rms=5, vib_peak=5, vib_dom_freq_hz=2, vib_band_drill=5, vib_band_low=5,
             hydrocarbon_ppm=5, water_temp_c=3, turbidity_ntu=3, latitude=6, longitude=6, chainage_km=4,
             cross_track_km=4, speed_kn=3, heading_deg=1, monitored_pos_km=4)


class Sink:
    def __init__(self, outdir, parquet):
        self.out, self.parquet, self.seen, self.pqw, self.counts = outdir, parquet, set(), {}, {}

    def write(self, table, scen, df):
        folder = os.path.join(self.out, "data", table)
        os.makedirs(folder, exist_ok=True)
        df = df.round({c: d for c, d in ROUND.items() if c in df.columns})
        key = (table, scen)
        path = os.path.join(folder, f"{scen}.csv.gz")
        first = key not in self.seen
        with gzip.open(path, "wt" if first else "at", newline="", compresslevel=6) as f:
            df.to_csv(f, header=first, index=False)
        self.counts[key] = self.counts.get(key, 0) + len(df)
        if self.parquet:
            import pyarrow as pa, pyarrow.parquet as pq
            tbl = pa.Table.from_pandas(df, preserve_index=False)
            pp = os.path.join(folder, f"{scen}.parquet")
            if first:
                self.pqw[key] = pq.ParquetWriter(pp, tbl.schema, compression="zstd")
            self.pqw[key].write_table(tbl.cast(self.pqw[key].schema))
        self.seen.add(key)

    def close(self):
        for w in self.pqw.values():
            w.close()


def scenario_catalog_rows():
    rows = []
    for s in SCENARIOS:
        evs = "; ".join(f"{e['type']} x{e['count'][0]}-{e['count'][1]}" + (f" (p={e['p']})" if 'p' in e else "") +
                        (f" [{e['vessel']}]" if e.get('vessel') else "") for e in s["events"]) or "none"
        rows.append(dict(
            scenario_id=s["id"], name=s["name"], category=s["category"], objective=s["objective"],
            description=s["description"], threat_events=evs,
            sensor_nodes="/".join(map(str, s["nodes"])), node_spacing_km=s["spacing_km"],
            min_corridor_km=s["min_corridor_km"], noise_multiplier=s["noise_mult"],
            rain_probability=s["rain_prob"], sensor_fault_probability=s["fault_prob"],
            operational_transients=f"{s['op_transients'][0]}-{s['op_transients'][1]}",
            legal_traffic="; ".join(f"{k} {v[0]}-{v[1]}" for k, v in s["traffic"].items()),
            ais_dark_boost=s["ais_dark_boost"], runs_factor=s["runs_factor"]))
    return rows


def write_docs(out, args, stats_df, runs_total, src, elapsed, files_info):
    cat = scenario_catalog_rows()
    pd.DataFrame(cat).to_csv(os.path.join(out, "scenario_catalog.csv"), index=False)
    with open(os.path.join(out, "scenario_catalog.json"), "w") as f:
        json.dump(SCENARIOS, f, indent=2, default=list)
    with open(os.path.join(out, "scenario_catalog.md"), "w") as f:
        f.write("# SABSS-Sim scenario catalogue\n\n")
        for r in cat:
            f.write(f"## {r['scenario_id']} - {r['name']}\n*Category:* {r['category']}\n\n"
                    f"**Objective.** {r['objective']}\n\n**Description.** {r['description']}\n\n"
                    f"- Threat events: {r['threat_events']}\n- Sensor nodes: {r['sensor_nodes']} (spacing {r['node_spacing_km']} km)\n"
                    f"- Noise multiplier: {r['noise_multiplier']}; rain probability: {r['rain_probability']}; "
                    f"sensor fault probability: {r['sensor_fault_probability']}\n- Lawful traffic: {r['legal_traffic']}\n"
                    f"- Operational transients: {r['operational_transients']}\n\n")
    dd = [dict(table=t, column=c, dtype=d, unit=u, description=desc) for t, cols in DICTIONARY.items()
          for c, d, u, desc in cols]
    pd.DataFrame(dd).to_csv(os.path.join(out, "data_dictionary.csv"), index=False)
    stats_df.to_csv(os.path.join(out, "dataset_statistics.csv"), index=False)
    tot_rows = int(stats_df["sensor_rows"].sum())
    with open(os.path.join(out, "DATASHEET.md"), "w") as f:
        f.write(f"""# SABSS-Sim v{VERSION} - Simulated multi-modal dataset for pipeline-bunkering surveillance

## Purpose
Open, fully labelled, reproducible simulation data for developing and benchmarking detection and
agent-coordination methods against illegal petroleum-pipeline bunkering (hot/cold tapping, leaks, spills)
in Nigeria's maritime backwaters. Produced for the Smart Agent-Based Surveillance System (SABSS) research.

## Contents
- {len(SCENARIOS)} scenarios, {runs_total} independent runs, {tot_rows:,} sensor-node readings (1 reading/node/minute).
- Tables: sensor_readings, vessel_tracks, events, runs, nodes (see data_dictionary.csv).
- scenario_catalog.* documents every simulated scenario (this is the Chapter 3 scenario register).
- pipelines_used.geojson: the pipeline segments used (geometry (c) OpenStreetMap contributors, ODbL).
- dataset_statistics.csv: rows, events and label prevalence per scenario; manifest.json: checksums.

## How it was generated
Generator: build_dataset.py v{VERSION} (included). Command: `{' '.join(sys.argv)}`. Global seed {args.seed};
each run is seeded by (seed, scenario index, run index) so every run is exactly reproducible.
Pipeline geometry source: {src}. Sensor nodes are placed on the real line geometry; vessel positions and
event locations are geo-referenced (WGS84). Generation time {elapsed:.0f} s on {platform.platform()}.

## Physical and behavioural models (summary)
- Pressure: P(x,t) = P0 - k*x - alpha*t + diurnal cycle + common-mode AR(1) noise + sensor noise. Leaks add a
  decaying-to-sustained drop attenuated with distance (length {LAMBDA_P} km); hot taps add a sustained drop
  after the drilling phase; lawful valve/pump transients add benign steps.
- Acoustics: a 128-sample waveform window at {FS} Hz is synthesised per node-minute (flow hum, vessel-engine
  tone 25-45 Hz, drilling tone 50 Hz + harmonic, low-frequency disturbance, broadband noise that grows with rain);
  only spectral features are stored.
- Hydrocarbons: log-normal background; spills advect downstream with delay = distance/current speed.
- Vessels: cargo transits, fishing (mean-reverting random walk), licensed barges (legal look-alikes),
  loitering tappers, hit-and-run boats and mother ships; AIS on/off modelled as a Markov chain.
- Weather: random storms raise noise, turbidity and background hydrocarbons.

## Labels
Node-level labels mark the physical footprint of each event (radii: pressure {R_P} km, acoustic {R_A} km,
chemical {R_C} km downstream with concentration > {CONC_DETECT} ppm, environmental {R_E} km). Sensor faults and
lawful operational transients are never labelled as threats. Event-level truth is in `events`.

## Known limitations (please read)
1. All data are synthetic; physical constants and behaviours are modelling assumptions, not calibrated to
   measured Niger Delta incidents. Results on this data show method feasibility, not field performance.
2. Pipeline attributes (operator, substance) come from incomplete OpenStreetMap tags.
3. Acoustic features derive from simulated waveforms; raw waveforms are not distributed.
4. Cold-tap pressure behaviour is simplified (no pressure signature).
5. Overlapping events: `event_id` shows the most recently started active event at a node-tick.

## Intended uses
Benchmarking anomaly/threat detectors, sensor-fusion and multi-agent response strategies, robustness studies
(noise, faults, night, weather), scalability studies, teaching.

## Loading example
```python
import pandas as pd
s = pd.read_csv("data/sensor_readings/S03.csv.gz", parse_dates=["timestamp"])
ev = pd.read_csv("data/events/S03.csv.gz")
runs = pd.read_csv("data/runs/S03.csv.gz")
```

## License and citation
Synthetic data: licence to be confirmed by the authors/institution (CC BY 4.0 is recommended for open release).
Geometry in pipelines_used.geojson is derived from OpenStreetMap and must be attributed:
(c) OpenStreetMap contributors, ODbL. Suggested citation: <Author(s)>, "SABSS-Sim: a simulated multi-modal
dataset for pipeline bunkering surveillance", v{VERSION}, <year>, <DOI after deposit, e.g. Zenodo>.
""")
    return


def export_geojson(out, used):
    feats = []
    for c in used.values():
        feats.append(dict(type="Feature", properties=dict(pipeline_id=c.pid, length_km=round(c.length_km, 3), **c.meta),
                          geometry=json.loads(json.dumps(c.geom_ll.__geo_interface__))))
    with open(os.path.join(out, "pipelines_used.geojson"), "w") as f:
        json.dump(dict(type="FeatureCollection", features=feats,
                       attribution="(c) OpenStreetMap contributors, ODbL"), f)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ═════════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="SABSS-Sim dataset generator")
    ap.add_argument("--scale", choices=["small", "medium", "large"], default="medium")
    ap.add_argument("--runs", type=int, default=None, help="runs per scenario (overrides --scale)")
    ap.add_argument("--ticks", type=int, default=1440, help="minutes per run (>= 600)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shapefile", default="../data/gis_maps/pipeline_corridor.shp")
    ap.add_argument("--out", default="../dataset/sabss_sim_v1")
    ap.add_argument("--ais-interval", type=int, default=5, help="vessel report interval in minutes")
    ap.add_argument("--scenarios", default="all", help="comma list, e.g. S01,S03")
    ap.add_argument("--parquet", action="store_true", help="also write Parquet (needs pyarrow)")
    ap.add_argument("--zip", action="store_true", help="bundle the dataset into a ZIP")
    ap.add_argument("--list-scenarios", action="store_true")
    args = ap.parse_args()

    if args.list_scenarios:
        for r in scenario_catalog_rows():
            print(f"{r['scenario_id']}  {r['name']:<52} [{r['category']}]")
        return
    if args.ticks < 600:
        ap.error("--ticks must be >= 600")
    runs_per = args.runs or dict(small=3, medium=20, large=100)[args.scale]
    chosen = SCENARIOS if args.scenarios == "all" else [s for s in SCENARIOS if s["id"] in args.scenarios.split(",")]
    os.makedirs(args.out, exist_ok=True)

    corridors, src = load_corridors(args.shapefile)
    print(f"Pipeline source: {src}  ({len(corridors)} usable segments >= 3 km)")
    sink, stats, used, run_tables = Sink(args.out, args.parquet), [], {}, 0
    t_start = time.time()
    for sc in chosen:
        n_runs = max(1, int(round(runs_per * sc["runs_factor"])))
        t0 = time.time()
        for r in range(n_runs):
            res = simulate_run(sc, SC_INDEX[sc["id"]], r, corridors, args)
            for tname, key in (("sensor_readings", "sensors"), ("vessel_tracks", "tracks"), ("events", "events"),
                               ("runs", "runs"), ("nodes", "nodes")):
                sink.write(tname, sc["id"], res[key])
            stats.append(res["stats"])
            used[res["cor"].pid] = res["cor"]
            run_tables += 1
        print(f"  {sc['id']} {sc['name'][:46]:<46} {n_runs:>4} runs  {time.time() - t0:6.1f}s")
    sink.close()
    elapsed = time.time() - t_start

    sdf = pd.DataFrame(stats)
    agg = sdf.drop(columns=["run_id"]).groupby("scenario_id").sum(numeric_only=True)
    agg.insert(0, "runs", sdf.groupby("scenario_id").size())
    agg["pct_ticks_any_anomaly"] = (100 * agg["pos_any"] / agg["sensor_rows"]).round(2)
    agg = agg.reset_index()
    export_geojson(args.out, used)
    write_docs(args.out, args, agg, run_tables, src, elapsed, None)

    files = {}
    for root, _, fs in os.walk(args.out):
        for fn in sorted(fs):
            p = os.path.join(root, fn)
            if fn != "manifest.json":
                files[os.path.relpath(p, args.out)] = dict(bytes=os.path.getsize(p), sha256=sha256(p))
    manifest = dict(dataset="SABSS-Sim", version=VERSION, generated=datetime.now().isoformat(timespec="seconds"),
                    command=" ".join(sys.argv), seed=args.seed, ticks=args.ticks, runs_per_scenario=runs_per,
                    scenarios=[s["id"] for s in chosen], total_runs=run_tables,
                    total_sensor_rows=int(agg["sensor_rows"].sum()), total_track_rows=int(agg["track_rows"].sum()),
                    total_events=int(agg["n_events"].sum()), python=sys.version.split()[0], numpy=np.__version__,
                    pandas=pd.__version__, platform=platform.platform(), pipeline_source=src, files=files)
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    shutil_copy = os.path.abspath(__file__)
    try:
        import shutil
        shutil.copy(shutil_copy, os.path.join(args.out, "build_dataset.py"))
    except Exception:
        pass

    print("\n" + "=" * 72)
    print(f"DONE in {elapsed:.0f}s  ->  {os.path.abspath(args.out)}")
    print(f"runs: {run_tables}   sensor rows: {manifest['total_sensor_rows']:,}   "
          f"vessel track rows: {manifest['total_track_rows']:,}   events: {manifest['total_events']}")
    print(agg[["scenario_id", "runs", "sensor_rows", "n_events", "pct_ticks_any_anomaly"]].to_string(index=False))
    if args.zip:
        zp = args.out.rstrip("/\\") + ".zip"
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
            for root, _, fs in os.walk(args.out):
                for fn in fs:
                    p = os.path.join(root, fn)
                    z.write(p, os.path.join(os.path.basename(args.out), os.path.relpath(p, args.out)))
        print(f"ZIP bundle: {os.path.abspath(zp)}  ({os.path.getsize(zp) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
