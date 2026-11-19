# SABSS-Sim v1.0.0 - Simulated multi-modal dataset for pipeline-bunkering surveillance

## Purpose
Open, fully labelled, reproducible simulation data for developing and benchmarking detection and
agent-coordination methods against illegal petroleum-pipeline bunkering (hot/cold tapping, leaks, spills)
in Nigeria's maritime backwaters. Produced for the Smart Agent-Based Surveillance System (SABSS) research.

## Contents
- 13 scenarios, 250 independent runs, 4,406,400 sensor-node readings (1 reading/node/minute).
- Tables: sensor_readings, vessel_tracks, events, runs, nodes (see data_dictionary.csv).
- scenario_catalog.* documents every simulated scenario (this is the Chapter 3 scenario register).
- pipelines_used.geojson: the pipeline segments used (geometry (c) OpenStreetMap contributors, ODbL).
- dataset_statistics.csv: rows, events and label prevalence per scenario; manifest.json: checksums.

## How it was generated
Generator: build_dataset.py v1.0.0 (included). Command: `build_dataset.py --scale medium --zip --parquet`. Global seed 42;
each run is seeded by (seed, scenario index, run index) so every run is exactly reproducible.
Pipeline geometry source: OpenStreetMap shapefile: pipeline_corridor.shp. Sensor nodes are placed on the real line geometry; vessel positions and
event locations are geo-referenced (WGS84). Generation time 141 s on Linux-7.0.0-43-generic-x86_64-with-glibc2.43.

## Physical and behavioural models (summary)
- Pressure: P(x,t) = P0 - k*x - alpha*t + diurnal cycle + common-mode AR(1) noise + sensor noise. Leaks add a
  decaying-to-sustained drop attenuated with distance (length 8.0 km); hot taps add a sustained drop
  after the drilling phase; lawful valve/pump transients add benign steps.
- Acoustics: a 128-sample waveform window at 500 Hz is synthesised per node-minute (flow hum, vessel-engine
  tone 25-45 Hz, drilling tone 50 Hz + harmonic, low-frequency disturbance, broadband noise that grows with rain);
  only spectral features are stored.
- Hydrocarbons: log-normal background; spills advect downstream with delay = distance/current speed.
- Vessels: cargo transits, fishing (mean-reverting random walk), licensed barges (legal look-alikes),
  loitering tappers, hit-and-run boats and mother ships; AIS on/off modelled as a Markov chain.
- Weather: random storms raise noise, turbidity and background hydrocarbons.

## Labels
Node-level labels mark the physical footprint of each event (radii: pressure 6.0 km, acoustic 1.5 km,
chemical 4.0 km downstream with concentration > 0.3 ppm, environmental 3.0 km). Sensor faults and
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
dataset for pipeline bunkering surveillance", v1.0.0, <year>, <DOI after deposit, e.g. Zenodo>.
