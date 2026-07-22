# Topological Structural-Health Monitor (TDA-based Damage Detector)

A demonstrative project applying **persistent homology** (algebraic topology)
to detect structural damage signatures in vibration sensor data — inspired
by real published research applying TDA to the **Z24 Bridge dataset**, a
real bridge in Switzerland that was deliberately damaged and monitored
before being demolished.
Live Demo:https://topo-shm.streamlit.app

This is a **working, runnable demo**: simulated but physically-motivated
sensor data, real persistence diagram computation (via `ripser`), and a
real (if simple) anomaly detector, with quantitative results — not just a
concept sketch.

## What it actually demonstrates

Running `main.py` simulates two accelerometer signals:

- **Healthy**: a single dominant vibration mode, plus slow temperature
  drift (a real confound in structural monitoring) and sensor noise.
- **Damaged**: the same signal, with a **breathing crack** nonlinearity
  added partway through — a well-documented real damage mechanism where
  a crack opens and closes with the vibration cycle, distorting the
  structure's dynamics asymmetrically.

Each signal is turned into a point cloud via **Takens' time-delay
embedding**, and **persistent homology** (H1 — loops) is computed on
sliding windows. The result:

- The healthy signal traces a **clean, simple loop** in embedded space.
- The damaged signal produces a **fragmented, higher-entropy shape**.
- A simple threshold (mean + 2×std of the *healthy* signal's own
  features — no damaged data used to set it) correctly flags damage in
  **~91% of post-damage windows** in this simulation, with a low false
  positive rate on the healthy period.

Output plots:
- `attractor_shapes.png` — visual comparison of the reconstructed
  healthy vs. damaged attractor shapes
- `topological_damage_signature.png` — topological features over time,
  with the true damage onset marked, showing the clear jump at onset

## Why persistent homology, not just raw signal thresholds?

Traditional structural health monitoring (natural frequency tracking,
cointegration methods) can be fooled by environmental/operational
variation — temperature, traffic load — producing false alarms.
Topological features are more robust to these confounds because they
capture the **global shape** of the system's dynamics rather than
pointwise values: a slow drift that keeps the same qualitative loop
shape doesn't register as damage, but a mechanism (like a breathing
crack) that genuinely distorts the dynamics does — which is exactly
what this demo shows.

## Honest scope: what this is, and isn't

This **is**: a correct, working implementation of the TDA pipeline
(embedding → persistence → features → detection), validated against a
known, simulated ground truth, with real quantitative results.

This **is not**: a deployed structural monitoring system, a collapse
predictor, or a validated tool for real infrastructure. The data here
is simulated. Real deployment requires real sensor data and real
engineering validation (see below).

## How this could become a real-world tool in the future

**The real problem:** unplanned structural failure (bridges, buildings,
tunnels) is catastrophic and expensive. By the time damage shows up in
simple threshold-based alerts, or becomes visible, it may already be
advanced. Early, environment-robust damage detection is a genuine,
active problem in civil engineering — this project explores one
promising mathematical approach to it.

**What would be needed to make it real:**

1. **Real sensor data** — replace `data_sim.py` with real accelerometer
   or strain-gauge streams. The public **Z24 Bridge dataset** (a real
   monitored bridge with documented, real induced damage scenarios) is
   the natural next step, and is what real published TDA-for-SHM
   research has validated against.
2. **Graded, validated ground truth** — real damage scenarios of varying
   severity, to measure not just detection but *lead time* before
   failure and false-positive rate over long deployments.
3. **Streaming pipeline engineering** — persistent homology is too slow
   for microsecond-scale problems, but structural monitoring operates on
   minutes-to-hours timescales, so a windowed, periodically-recomputed
   pipeline (as built here) is realistic — it would need to be
   engineered for continuous, unattended operation.
4. **Multi-sensor fusion** — a real structure has many sensors; combining
   topological features across sensor locations would improve both
   detection confidence and damage *localization*, which this
   single-channel demo doesn't attempt.
5. **Baseline comparison on real data** — benchmarking against
   established SHM methods (e.g. frequency-domain cointegration) on real
   data, to honestly establish whether TDA adds value over existing
   methods in practice, not just in simulation.
6. **Engineering and regulatory validation** — any real deployment on
   live infrastructure requires civil engineering sign-off, redundancy,
   and safety certification. This project is a research/software
   prototype, not a certified monitoring system.

## Project structure

```
topo-bridge-shm/
├── README.md
├── requirements.txt
├── data_sim.py         # simulates healthy vs. damaged bridge vibration signals
├── tda_pipeline.py     # Takens embedding + persistent homology feature extraction
├── main.py             # script version: produces static plots and a detection report
└── streamlit_app.py    # interactive web app version (recommended for showcasing)
```

## Running the script version locally

```bash
git clone <your-repo-url>
cd topo-bridge-shm
pip install -r requirements.txt
python main.py
```

This simulates a healthy and a damaged bridge signal, runs the TDA
pipeline on sliding windows of each, saves two comparison plots, and
prints a detection report (true/false positives, recall) evaluated
against the known simulated damage onset.

## Running the interactive web app locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

This opens a browser at `http://localhost:8501` with live sliders for
damage onset, damage severity, noise level, and analysis window size —
the attractor shape plots, feature time series, and detection metrics
update in real time as you move them.

## Deploying the web app publicly (free)

1. Push this project to a GitHub repository (see steps below).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in
   with GitHub.
3. Click **"New app"**, select this repository, and set the main file
   path to `streamlit_app.py`.
4. Click **Deploy**. Streamlit Community Cloud installs
   `requirements.txt` automatically and gives you a public URL
   (e.g. `https://<your-app-name>.streamlit.app`) you can share with
   anyone — no server management needed.

### Pushing to GitHub

```bash
cd topo-bridge-shm
git init
git add .
git commit -m "Initial commit: topological structural health monitor"
git branch -M main
git remote add origin https://github.com/<your-username>/topo-bridge-shm.git
git push -u origin main
```

## References

- Gowdridge, T., Dervilis, N., Worden, K. *On the application of
  topological data analysis: a Z24 Bridge case study.* (2022)
- Gowdridge, T., Dervilis, N., Worden, K. *On topological data analysis
  for structural dynamics: an introduction to persistent homology.* (2022)
