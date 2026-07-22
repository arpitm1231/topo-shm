"""
Runs the full demo: generates a healthy bridge signal and a damaged
bridge signal, extracts topological features from sliding windows of
each, and shows that damage produces a clearly different topological
signature than temperature drift alone.

Produces two plots:
  - attractor_shapes.png       : the reconstructed point-cloud shapes
                                   (healthy vs damaged) side by side
  - topological_damage_signature.png : topological features over time,
                                   with the true damage onset marked
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3D projection)

from data_sim import generate_signal
from tda_pipeline import analyze_window, takens_embedding

WINDOW_SIZE = 500   # samples per analysis window (5s at fs=100Hz)
STEP = 250          # slide step (50% overlap)
FS = 100
DURATION = 60
DAMAGE_START = 30


def sliding_window_features(signal, window_size=WINDOW_SIZE, step=STEP):
    feats = []
    centers_s = []
    for start in range(0, len(signal) - window_size, step):
        window = signal[start:start + window_size]
        f, _, _ = analyze_window(window)
        feats.append(f)
        centers_s.append((start + window_size / 2) / FS)
    return feats, centers_s


def plot_attractor_shapes(healthy, damaged):
    cloud_h = takens_embedding(healthy[2000:2500], dim=3, delay=5)
    cloud_d = takens_embedding(damaged[4500:5000], dim=3, delay=5)  # window after damage onset

    fig = plt.figure(figsize=(11, 5))

    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot(cloud_h[:, 0], cloud_h[:, 1], cloud_h[:, 2], lw=0.8, color="#2b8a3e")
    ax1.set_title("Healthy window\n(clean, simple attractor loop)")

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot(cloud_d[:, 0], cloud_d[:, 1], cloud_d[:, 2], lw=0.8, color="#c92a2a")
    ax2.set_title("Damaged window\n(fragmented, distorted shape)")

    plt.tight_layout()
    plt.savefig("attractor_shapes.png", dpi=150)
    print("Saved plot to attractor_shapes.png")


def plot_feature_timeseries(feats_healthy, centers_h, feats_damaged, centers_d):
    entropy_h = [f["h1_entropy"] for f in feats_healthy]
    entropy_d = [f["h1_entropy"] for f in feats_damaged]
    persist_h = [f["h1_total_persistence"] for f in feats_healthy]
    persist_d = [f["h1_total_persistence"] for f in feats_damaged]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].plot(centers_h, entropy_h, label="Healthy (temp drift only)", marker="o")
    axes[0].plot(centers_d, entropy_d, label="Damaged signal", marker="o")
    axes[0].axvline(x=DAMAGE_START, color="red", linestyle="--", alpha=0.6, label="true damage onset")
    axes[0].set_title("H1 Persistent Entropy Over Time")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Entropy")
    axes[0].legend()

    axes[1].plot(centers_h, persist_h, label="Healthy (temp drift only)", marker="o")
    axes[1].plot(centers_d, persist_d, label="Damaged signal", marker="o")
    axes[1].axvline(x=DAMAGE_START, color="red", linestyle="--", alpha=0.6, label="true damage onset")
    axes[1].set_title("H1 Total Persistence Over Time")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Total persistence")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("topological_damage_signature.png", dpi=150)
    print("Saved plot to topological_damage_signature.png")


def simple_detection_report(feats_damaged, centers_d, feats_healthy):
    """
    A minimal, honest evaluation: use the healthy signal's own feature
    distribution to set a threshold, then check how well that threshold
    flags windows in the damaged signal that occur after the true
    damage onset (ground truth, since we simulated it).
    """
    healthy_persist = np.array([f["h1_total_persistence"] for f in feats_healthy])
    threshold = healthy_persist.mean() + 2 * healthy_persist.std()

    damaged_persist = np.array([f["h1_total_persistence"] for f in feats_damaged])
    flagged = damaged_persist > threshold
    true_damage = np.array(centers_d) >= DAMAGE_START

    tp = np.sum(flagged & true_damage)
    fn = np.sum(~flagged & true_damage)
    fp = np.sum(flagged & ~true_damage)
    tn = np.sum(~flagged & ~true_damage)

    print("\n--- Simple anomaly-detection report (threshold = healthy mean + 2*std) ---")
    print(f"Threshold on H1 total persistence: {threshold:.4f}")
    print(f"True positives (correctly flagged damage windows):  {tp}")
    print(f"False negatives (missed damage windows):            {fn}")
    print(f"False positives (healthy-period windows flagged):   {fp}")
    print(f"True negatives (correctly quiet healthy windows):   {tn}")
    if tp + fn > 0:
        print(f"Recall on damage windows: {tp / (tp + fn):.2%}")


def main():
    print("Simulating healthy bridge signal (temperature drift only)...")
    _, healthy = generate_signal(duration_s=DURATION, fs=FS, damage_start=None, seed=1)

    print(f"Simulating bridge signal with damage starting at t={DAMAGE_START}s...")
    _, damaged = generate_signal(duration_s=DURATION, fs=FS, damage_start=DAMAGE_START, seed=1)

    print("Extracting topological features (healthy)...")
    feats_healthy, centers_h = sliding_window_features(healthy)

    print("Extracting topological features (damaged)...")
    feats_damaged, centers_d = sliding_window_features(damaged)

    plot_attractor_shapes(healthy, damaged)
    plot_feature_timeseries(feats_healthy, centers_h, feats_damaged, centers_d)
    simple_detection_report(feats_damaged, centers_d, feats_healthy)

    print("\nDone. Open attractor_shapes.png and topological_damage_signature.png to see the results.")


if __name__ == "__main__":
    main()
