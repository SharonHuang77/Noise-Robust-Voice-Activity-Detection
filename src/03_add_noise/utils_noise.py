from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

EPS = 1e-12

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64) + EPS))

def speech_mask_from_intervals(num_samples: int, speech_intervals: List[Dict]) -> np.ndarray:
    """
    speech_intervals: [{"start": int, "end": int}, ...]
    Returns boolean mask length num_samples.
    """
    m = np.zeros(num_samples, dtype=bool)
    for seg in speech_intervals:
        s = int(seg["start"])
        e = int(seg["end"])
        s = max(0, min(num_samples, s))
        e = max(0, min(num_samples, e))
        if e > s:
            m[s:e] = True
    return m

def crop_or_tile_to_length(x: np.ndarray, target_len: int, rng: np.random.Generator) -> np.ndarray:
    """
    Old helper: returns only the cropped/tiled result.
    Kept for backwards compatibility.
    """
    if x.size >= target_len:
        start = int(rng.integers(0, x.size - target_len + 1))
        return x[start:start + target_len]
    reps = int(np.ceil(target_len / x.size))
    xt = np.tile(x, reps)
    return xt[:target_len]

def crop_or_tile_with_decision(x: np.ndarray, target_len: int, rng: np.random.Generator):
    """
    Returns (y, decision_dict)
    decision_dict is sufficient to replay the operation.
    """
    if x.size >= target_len:
        start = int(rng.integers(0, x.size - target_len + 1))
        y = x[start:start + target_len]
        return y, {"mode": "crop", "start": start, "orig_len": int(x.size), "target_len": int(target_len)}
    reps = int(np.ceil(target_len / x.size))
    xt = np.tile(x, reps)
    y = xt[:target_len]
    return y, {"mode": "tile", "reps": reps, "orig_len": int(x.size), "target_len": int(target_len)}

def apply_peak(x: np.ndarray, peak: float = 0.99) -> np.ndarray:
    """
    Old helper: scales waveform down if needed.
    Kept for backwards compatibility (Step 3 now stores explicit clip_gain instead).
    """
    if x.size == 0:
        return x
    p = float(np.max(np.abs(x)))
    if p > peak and p > 0:
        return (x * (peak / p)).astype(np.float32, copy=False)
    return x.astype(np.float32, copy=False)

def compute_snr_db(s: np.ndarray, n: np.ndarray) -> float:
    """
    SNR = 20 log10(rms(s) / rms(n))
    """
    rs = rms(s)
    rn = rms(n)
    return float(20.0 * np.log10((rs + EPS) / (rn + EPS)))