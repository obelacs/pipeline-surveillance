# SABSS-Sim scenario catalogue

## S01 - Baseline normal operations
*Category:* Baseline

**Objective.** Measure the false alarm rate (FAR) of every detector when nothing is wrong.

**Description.** Pipeline operating normally. Legal traffic only (cargo transits, fishing boats, an occasional licensed bunker barge). Normal diurnal pressure cycle and tidal turbidity.

- Threat events: none
- Sensor nodes: 10 (spacing 1.5 km)
- Noise multiplier: 1.0; rain probability: 0.25; sensor fault probability: 0.0
- Lawful traffic: cargo_transit 2-5; fishing 3-8; licensed_barge 0-1
- Operational transients: 0-0

## S02 - Single pipeline leak
*Category:* Pipeline integrity

**Objective.** Evaluate pressure-drop detection accuracy, localisation and response time.

**Description.** One accidental or corrosion-induced leak at a random point of the monitored section. Sudden pressure drop that partially recovers to a sustained lower level until repaired.

- Threat events: leak x1-1
- Sensor nodes: 10 (spacing 1.5 km)
- Noise multiplier: 1.0; rain probability: 0.25; sensor fault probability: 0.0
- Lawful traffic: cargo_transit 2-5; fishing 3-8; licensed_barge 0-1
- Operational transients: 0-0

## S03 - Hot-tap illegal bunkering (full attack chain)
*Category:* Illegal bunkering

**Objective.** Evaluate multi-sensor fusion: vessel behaviour + drilling acoustics + pressure drop.

**Description.** A small boat (Cotonou-type) approaches, loiters at the pipeline and drills a live line. Drilling acoustics first, then product diversion causes a sustained pressure drop. Sediment disturbance raises local turbidity. AIS is frequently switched off.

- Threat events: hot_tap x1-1 [loitering_tapper]
- Sensor nodes: 10 (spacing 1.5 km)
- Noise multiplier: 1.0; rain probability: 0.25; sensor fault probability: 0.0
- Lawful traffic: cargo_transit 2-5; fishing 3-8; licensed_barge 0-1
- Operational transients: 0-0

## S04 - Cold-tap illegal bunkering
*Category:* Illegal bunkering

**Objective.** Test detection when the pressure signature is absent (acoustic + vessel only).

**Description.** Tapping of a depressurised line (during shut-in). Drilling acoustics and a loitering vessel are present but there is no pressure anomaly - a hard case for pressure-based systems.

- Threat events: cold_tap x1-1 [loitering_tapper]
- Sensor nodes: 10 (spacing 1.5 km)
- Noise multiplier: 1.0; rain probability: 0.25; sensor fault probability: 0.0
- Lawful traffic: cargo_transit 2-5; fishing 3-8; licensed_barge 0-1
- Operational transients: 0-0

## S05 - Oil spill from a vandalised pipeline
*Category:* Environmental

**Objective.** Evaluate spill detection, plume tracking and downstream alert delay.

**Description.** A rupture releases crude. Pressure drops (coupled leak) and a hydrocarbon plume advects downstream with the tidal/river current, arriving at downstream sensors with a delay.

- Threat events: spill x1-1
- Sensor nodes: 10 (spacing 1.5 km)
- Noise multiplier: 1.0; rain probability: 0.25; sensor fault probability: 0.0
- Lawful traffic: cargo_transit 2-5; fishing 3-8; licensed_barge 0-1
- Operational transients: 0-0

## S06 - Simultaneous multi-site attacks
*Category:* Illegal bunkering

**Objective.** Stress-test agent coordination and resource allocation (drones, patrols).

**Description.** A hot tap, a cold tap, a leak and a spill start within about an hour of each other at different points of a long monitored section, competing for limited response assets.

- Threat events: hot_tap x1-1 [loitering_tapper]; cold_tap x1-1 [loitering_tapper]; leak x1-1; spill x1-1
- Sensor nodes: 20 (spacing 1.5 km)
- Noise multiplier: 1.0; rain probability: 0.25; sensor fault probability: 0.0
- Lawful traffic: cargo_transit 2-5; fishing 3-8; licensed_barge 0-1
- Operational transients: 0-0

## S07 - Night-time covert hit-and-run tapping
*Category:* Illegal bunkering

**Objective.** Evaluate detection of short, covert attacks under reduced visibility.

**Description.** Fast boats arrive in the dark, tap for 30-90 minutes and leave. AIS is usually off. Models the situation where optical confirmation by UAV is degraded at night.

- Threat events: hot_tap x1-1 [hit_and_run]
- Sensor nodes: 10 (spacing 1.5 km)
- Noise multiplier: 1.0; rain probability: 0.25; sensor fault probability: 0.0
- Lawful traffic: cargo_transit 2-5; fishing 3-8; licensed_barge 0-1
- Operational transients: 0-0

## S08 - Adverse weather and high sea state
*Category:* Environmental stress

**Objective.** Measure robustness to noise: storms, rain and rough water raise sensor noise.

**Description.** Heavy rain/storm for much of the run, doubled sensor noise, high turbidity. Each run contains a hot tap or a leak with probability 0.5 (otherwise it is a pure false-alarm test under bad weather).

- Threat events: hot_tap x1-1 (p=0.5) [loitering_tapper]; leak x1-1 (p=0.5)
- Sensor nodes: 10 (spacing 1.5 km)
- Noise multiplier: 2.0; rain probability: 1.0; sensor fault probability: 0.0
- Lawful traffic: cargo_transit 2-5; fishing 3-8; licensed_barge 0-1
- Operational transients: 0-0

## S09 - Sensor degradation and faults
*Category:* Robustness

**Objective.** Test fault tolerance: dropouts, stuck sensors, drift and spikes.

**Description.** 30% of sensor nodes develop a fault (communication dropout, stuck value, calibration drift or spikes) while a hot tap or leak occurs with probability 0.6. Faults are recorded in sensor_ok / fault_type but are NOT labelled as threats.

- Threat events: hot_tap x1-1 (p=0.6) [loitering_tapper]; leak x1-1 (p=0.6)
- Sensor nodes: 10 (spacing 1.5 km)
- Noise multiplier: 1.0; rain probability: 0.25; sensor fault probability: 0.3
- Lawful traffic: cargo_transit 2-5; fishing 3-8; licensed_barge 0-1
- Operational transients: 0-0

## S10 - Legitimate operations (hard negatives)
*Category:* Baseline

**Objective.** Quantify false alarms caused by lawful look-alike activity.

**Description.** Licensed bunker barges work close to the pipeline for hours (slow, loitering, engine noise) and operators perform valve/pump actions that cause pressure transients. No threat exists, so every alert is a false alarm.

- Threat events: none
- Sensor nodes: 10 (spacing 1.5 km)
- Noise multiplier: 1.0; rain probability: 0.25; sensor fault probability: 0.0
- Lawful traffic: cargo_transit 2-4; fishing 3-6; licensed_barge 3-5
- Operational transients: 2-4

## S11 - Mother-ship supported multi-tap operation
*Category:* Illegal bunkering

**Objective.** Evaluate detection of organised, sequential tapping supported by a mother ship.

**Description.** An anchored mother ship collects product while skiffs shuttle between it and 2-3 tap points along the line, one after another during the night.

- Threat events: hot_tap x2-3 [mother_ship]
- Sensor nodes: 10 (spacing 1.5 km)
- Noise multiplier: 1.0; rain probability: 0.25; sensor fault probability: 0.0
- Lawful traffic: cargo_transit 2-5; fishing 3-8; licensed_barge 0-1
- Operational transients: 0-0

## S12 - Scalability stress test
*Category:* Scalability

**Objective.** Measure how latency/accuracy scale with the number of agents (sensor nodes and vessels).

**Description.** The same hot-tap/leak situation is simulated with 5, 10, 20, 50 and 100 sensor nodes and 3-5x denser vessel traffic (node spacing 0.5 km, shorter if the segment is short).

- Threat events: hot_tap x1-1 [loitering_tapper]; leak x1-1 (p=0.5)
- Sensor nodes: 5/10/20/50/100 (spacing 0.5 km)
- Noise multiplier: 1.0; rain probability: 0.25; sensor fault probability: 0.0
- Lawful traffic: cargo_transit 5-15; fishing 10-40; licensed_barge 1-4
- Operational transients: 0-0

## S13 - Non-malicious environmental disturbance
*Category:* Environmental

**Objective.** Test discrimination of natural/industrial disturbance (dredging, anchor dragging) from attacks.

**Description.** Sediment disturbance from dredging or anchor dragging raises turbidity and low-frequency vibration near the pipeline without any tapping. Frequent rain adds natural turbidity.

- Threat events: env_disturbance x1-2
- Sensor nodes: 10 (spacing 1.5 km)
- Noise multiplier: 1.0; rain probability: 0.6; sensor fault probability: 0.0
- Lawful traffic: cargo_transit 2-5; fishing 3-8; licensed_barge 1-3
- Operational transients: 0-0

