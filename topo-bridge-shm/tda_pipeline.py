"""
Core TDA pipeline: time-delay embedding + persistent homology feature
extraction. Uses ripser for real persistence diagram computation
(not a toy/approximate implementation).
"""
import numpy as np
from ripser import ripser
from persim.persistent_entropy import persistent_entropy


def takens_embedding(signal, dim=3, delay=5):
    """
    Time-delay (Takens) embedding: turns a 1D signal into a point cloud
    in `dim`-dimensional space. This reconstructs the shape of the
    underlying dynamical attractor from a single sensor channel.
    """
    n = len(signal) - (dim - 1) * delay
    if n <= 0:
        raise ValueError("Signal too short for the given dim/delay")
    return np.array([signal[i:i + n] for i in range(0, dim * delay, delay)]).T


def compute_persistence(point_cloud, maxdim=1):
    """
    Computes persistence diagrams via ripser.
    Returns a list: [H0 diagram (components), H1 diagram (loops)].
    """
    result = ripser(point_cloud, maxdim=maxdim)
    return result["dgms"]


def extract_features(dgms):
    """
    Turns persistence diagrams into a small numeric feature vector
    describing the H1 (loop) topology of the signal's attractor:

    - h1_total_persistence: sum of lifetimes of loops (overall "loopiness")
    - h1_max_persistence: strongest/longest-lived loop
    - h1_entropy: persistent entropy (how disordered the topology is —
      a clean single loop has low entropy, a fragmented noisy shape
      has high entropy)
    - h1_count_significant: number of loops with above-median lifetime
    """
    h1 = dgms[1]
    h1 = h1[np.isfinite(h1[:, 1])]  # drop infinite bars, if any

    if len(h1) == 0:
        return {
            "h1_total_persistence": 0.0,
            "h1_max_persistence": 0.0,
            "h1_entropy": 0.0,
            "h1_count_significant": 0,
        }

    lifetimes = h1[:, 1] - h1[:, 0]
    entropy = persistent_entropy([h1])[0]
    significant = int(np.sum(lifetimes > np.median(lifetimes)))

    return {
        "h1_total_persistence": float(np.sum(lifetimes)),
        "h1_max_persistence": float(np.max(lifetimes)),
        "h1_entropy": float(entropy),
        "h1_count_significant": significant,
    }


def analyze_window(signal_window, dim=3, delay=5):
    """
    Full pipeline for one signal window: embed -> persistence -> features.
    Returns (features_dict, persistence_diagrams, point_cloud).
    """
    cloud = takens_embedding(signal_window, dim=dim, delay=delay)
    dgms = compute_persistence(cloud)
    features = extract_features(dgms)
    return features, dgms, cloud
