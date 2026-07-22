"""
Interactive web demo for the Topological Structural-Health Monitor.

Run locally:
    streamlit run streamlit_app.py

Deploy free:
    Push this repo to GitHub, then deploy at https://share.streamlit.io
    (Streamlit Community Cloud) pointing at streamlit_app.py.
"""
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from data_sim import generate_signal
from tda_pipeline import analyze_window, takens_embedding

# ----------------------------------------------------------------------
# Page setup & styling
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Topological Structural-Health Monitor",
    page_icon="🌉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main { background-color: #0f1116; }
        h1, h2, h3 { font-family: 'Helvetica Neue', sans-serif; }
        .disclaimer-badge {
            display: inline-block;
            background: #2b2f3a;
            color: #f0b429;
            border: 1px solid #f0b429;
            border-radius: 999px;
            padding: 4px 14px;
            font-size: 0.8rem;
            margin-bottom: 1rem;
        }
        .metric-card {
            background: #1a1d27;
            border-radius: 12px;
            padding: 1rem 1.2rem;
            border: 1px solid #2b2f3a;
        }
        div[data-testid="stMetric"] {
            background: #1a1d27;
            border-radius: 12px;
            padding: 0.8rem 1rem;
            border: 1px solid #2b2f3a;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🌉 Topological Structural-Health Monitor")
st.markdown(
    '<span class="disclaimer-badge">Research demo — simulated data, not a certified monitoring tool</span>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    This demo detects **structural damage signatures** in vibration sensor data
    using **persistent homology** (algebraic topology) instead of simple
    frequency thresholds. A structure's vibration signal is reconstructed as a
    geometric shape via *time-delay embedding* — a healthy structure traces a
    clean loop, while a damaged one produces a fragmented, distorted shape.
    Move the controls on the left and watch the topology change in real time.
    """
)

# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------
st.sidebar.header("⚙️ Simulation controls")

duration = st.sidebar.slider("Signal duration (s)", 30, 120, 60, step=10)
fs = 100
damage_start = st.sidebar.slider("Damage onset time (s)", 5, duration - 5, duration // 2)
damage_severity = st.sidebar.slider("Damage severity", 0.0, 1.5, 0.6, step=0.1)
noise_level = st.sidebar.slider("Sensor noise level", 0.01, 0.15, 0.03, step=0.01)
seed = st.sidebar.number_input("Random seed", value=1, step=1)

st.sidebar.header("🔬 Analysis controls")
window_size = st.sidebar.slider("Window size (samples)", 200, 800, 500, step=50)
step = st.sidebar.slider("Window step (samples)", 100, 500, 250, step=50)

run_button = st.sidebar.button("▶️ Run simulation", use_container_width=True)


# ----------------------------------------------------------------------
# Simulation core (cached so slider tweaks that don't affect the sim
# don't force a full recompute)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def simulate(duration, fs, damage_start, damage_severity, noise_level, seed):
    t = np.arange(0, duration, 1 / fs)
    rng = np.random.default_rng(seed)
    mode = np.sin(2 * np.pi * 2.0 * t)
    temp_drift = 0.15 * np.sin(2 * np.pi * (1 / duration) * t)
    noise = noise_level * rng.standard_normal(len(t))
    healthy = mode + temp_drift + noise

    damage_mask = (t >= damage_start).astype(float)
    progress = np.clip((t - damage_start) / max(1e-6, (duration - damage_start)), 0, 1)
    breathing_crack = damage_severity * progress * (mode ** 2) * np.sign(mode) * damage_mask
    extra_noise = (noise_level * 3) * progress * rng.standard_normal(len(t)) * damage_mask
    damaged = healthy + breathing_crack + extra_noise

    return t, healthy, damaged


@st.cache_data(show_spinner=False)
def compute_features(signal, window_size, step, fs):
    feats, centers = [], []
    for start in range(0, len(signal) - window_size, step):
        window = signal[start:start + window_size]
        f, _, _ = analyze_window(window)
        feats.append(f)
        centers.append((start + window_size / 2) / fs)
    return feats, centers


# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------
t, healthy, damaged = simulate(duration, fs, damage_start, damage_severity, noise_level, seed)

with st.spinner("Computing persistent homology across sliding windows..."):
    feats_healthy, centers_h = compute_features(healthy, window_size, step, fs)
    feats_damaged, centers_d = compute_features(damaged, window_size, step, fs)

persist_h = np.array([f["h1_total_persistence"] for f in feats_healthy])
persist_d = np.array([f["h1_total_persistence"] for f in feats_damaged])
entropy_h = np.array([f["h1_entropy"] for f in feats_healthy])
entropy_d = np.array([f["h1_entropy"] for f in feats_damaged])

threshold = persist_h.mean() + 2 * persist_h.std()
centers_d_arr = np.array(centers_d)
flagged = persist_d > threshold
true_damage = centers_d_arr >= damage_start

tp = int(np.sum(flagged & true_damage))
fn = int(np.sum(~flagged & true_damage))
fp = int(np.sum(flagged & ~true_damage))
recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

# ----------------------------------------------------------------------
# Metrics row
# ----------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Detection recall", f"{recall:.0%}" if recall == recall else "n/a")
col2.metric("True positives", tp)
col3.metric("False positives", fp)
col4.metric("Missed (false negatives)", fn)

st.divider()

# ----------------------------------------------------------------------
# Raw signal plot
# ----------------------------------------------------------------------
st.subheader("📈 Simulated sensor signal")
fig_signal = go.Figure()
fig_signal.add_trace(go.Scatter(x=t, y=healthy, name="Healthy (reference)",
                                 line=dict(color="#4dabf7", width=1)))
fig_signal.add_trace(go.Scatter(x=t, y=damaged, name="Damaged (monitored structure)",
                                 line=dict(color="#ff8787", width=1)))
fig_signal.add_vline(x=damage_start, line_dash="dash", line_color="#f0b429",
                      annotation_text="damage onset", annotation_position="top")
fig_signal.update_layout(
    template="plotly_dark", height=320, margin=dict(l=10, r=10, t=30, b=10),
    xaxis_title="Time (s)", yaxis_title="Acceleration (a.u.)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig_signal, use_container_width=True)

# ----------------------------------------------------------------------
# Attractor shapes (3D)
# ----------------------------------------------------------------------
st.subheader("🔵 Reconstructed attractor shape (Takens embedding)")
colA, colB = st.columns(2)

pre_idx = int(max(0, damage_start - 5) * fs)
post_idx = int(min(duration - 5, damage_start + (duration - damage_start) * 0.7) * fs)

cloud_healthy = takens_embedding(healthy[pre_idx:pre_idx + window_size], dim=3, delay=5)
cloud_damaged = takens_embedding(damaged[post_idx:post_idx + window_size], dim=3, delay=5)

with colA:
    fig_h = go.Figure(data=[go.Scatter3d(
        x=cloud_healthy[:, 0], y=cloud_healthy[:, 1], z=cloud_healthy[:, 2],
        mode="lines", line=dict(color="#51cf66", width=3),
    )])
    fig_h.update_layout(template="plotly_dark", height=380,
                         margin=dict(l=0, r=0, t=30, b=0),
                         title="Before damage: clean, simple loop",
                         scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False))
    st.plotly_chart(fig_h, use_container_width=True)

with colB:
    fig_d = go.Figure(data=[go.Scatter3d(
        x=cloud_damaged[:, 0], y=cloud_damaged[:, 1], z=cloud_damaged[:, 2],
        mode="lines", line=dict(color="#ff6b6b", width=3),
    )])
    fig_d.update_layout(template="plotly_dark", height=380,
                         margin=dict(l=0, r=0, t=30, b=0),
                         title="After damage: fragmented, distorted shape",
                         scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False))
    st.plotly_chart(fig_d, use_container_width=True)

# ----------------------------------------------------------------------
# Topological feature time series
# ----------------------------------------------------------------------
st.subheader("📊 Topological damage signature over time")

fig_feat = go.Figure()
fig_feat.add_trace(go.Scatter(x=centers_h, y=persist_h, name="Healthy — total persistence",
                               line=dict(color="#4dabf7"), mode="lines+markers"))
fig_feat.add_trace(go.Scatter(x=centers_d, y=persist_d, name="Damaged — total persistence",
                               line=dict(color="#ff8787"), mode="lines+markers"))
fig_feat.add_hline(y=threshold, line_dash="dot", line_color="#f0b429",
                    annotation_text="detection threshold (healthy mean + 2σ)")
fig_feat.add_vline(x=damage_start, line_dash="dash", line_color="#f0b429",
                    annotation_text="true damage onset")
fig_feat.update_layout(
    template="plotly_dark", height=380, margin=dict(l=10, r=10, t=30, b=10),
    xaxis_title="Time (s)", yaxis_title="H1 total persistence",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
)
st.plotly_chart(fig_feat, use_container_width=True)

# ----------------------------------------------------------------------
# Explanation
# ----------------------------------------------------------------------
with st.expander("ℹ️ How this works, and what it would take to make it real"):
    st.markdown(
        """
**Pipeline:** sensor signal → time-delay (Takens) embedding into a point
cloud → persistent homology (`ripser`) → numeric topological features
(total persistence, entropy of H1 loops) → threshold-based anomaly flag.

**Why topology instead of raw frequency thresholds:** temperature and
operational load changes shift a structure's vibration frequencies too,
which causes false alarms in naive threshold methods. Topological
features are more robust to this because they capture the *global shape*
of the signal's dynamics rather than pointwise values — this mirrors real
published research applying persistent homology to the Z24 Bridge dataset
(a real bridge, real induced damage, monitored before demolition).

**Honest scope:** this demo uses simulated data. Making it real would
require: real accelerometer data (e.g. the Z24 Bridge dataset), validated
graded damage scenarios, multi-sensor fusion, benchmarking against
established SHM methods, and formal engineering/regulatory validation
before any real deployment. See the project README for the full roadmap.
        """
    )

st.caption("Built with persistent homology (ripser), Takens embedding, and Streamlit.")
