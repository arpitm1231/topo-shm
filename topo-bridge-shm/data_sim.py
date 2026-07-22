"""
Simulates bridge accelerometer data.

This stands in for real sensor data (e.g. the Z24 Bridge dataset) so the
pipeline runs immediately with no external downloads. Swap in real CSV
data later by replacing generate_signal() with a loader for real sensor
logs (see README for details on the Z24 Bridge dataset).
"""
import numpy as np


def generate_signal(duration_s=60, fs=100, damage_start=None, seed=None):
    """
    Simulates one accelerometer channel on a bridge.

    duration_s: length of signal in seconds
    fs: sampling frequency (Hz)
    damage_start: time (s) at which a damage-like distortion begins,
                  or None for a fully healthy signal
    seed: RNG seed for reproducibility

    Returns (t, signal) as numpy arrays.

    Signal model:
    - Healthy: a single dominant vibration mode plus slow temperature
      drift (a real confound that should NOT be mistaken for damage)
      and small sensor noise. Reconstructed via time-delay embedding,
      this traces a clean, simple loop.
    - Damaged: models a "breathing crack" — a well-documented nonlinear
      structural damage mechanism where a crack opens and closes with
      the vibration cycle, changing the structure's effective stiffness
      asymmetrically. This is simulated as a nonlinear, sign-dependent
      distortion of the base mode, growing in severity after
      damage_start. It fragments the clean loop into a distorted,
      higher-complexity shape.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration_s, 1 / fs)

    mode = np.sin(2 * np.pi * 2.0 * t)  # dominant vibration mode

    # Slow temperature drift — should NOT look like damage.
    temp_drift = 0.15 * np.sin(2 * np.pi * (1 / duration_s) * t)

    noise = 0.03 * rng.standard_normal(len(t))

    signal = mode + temp_drift + noise

    if damage_start is not None:
        damage_mask = (t >= damage_start).astype(float)
        progress = np.clip((t - damage_start) / max(1e-6, (duration_s - damage_start)), 0, 1)

        # Breathing-crack nonlinearity: asymmetric distortion tied to the
        # sign of the vibration cycle, growing in severity over time.
        breathing_crack = 0.6 * progress * (mode ** 2) * np.sign(mode) * damage_mask
        extra_noise = 0.1 * progress * rng.standard_normal(len(t)) * damage_mask

        signal = signal + breathing_crack + extra_noise

    return t, signal


if __name__ == "__main__":
    t, healthy = generate_signal(damage_start=None, seed=1)
    t, damaged = generate_signal(damage_start=30, seed=1)
    print("Healthy signal sample:", healthy[:5])
    print("Damaged signal sample:", damaged[:5])
