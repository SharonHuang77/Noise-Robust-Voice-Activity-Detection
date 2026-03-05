from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np

from utils_noise import rms, crop_or_tile_with_decision, compute_snr_db

try:
    import soundfile as sf
    import librosa
except Exception as e:
    raise ImportError("Step 3 requires soundfile + librosa: pip install soundfile librosa") from e


@dataclass(frozen=True)
class NoiseParams:
    sr: int = 16000
    p_noise: float = 0.5
    p_music: float = 0.3
    p_babble: float = 0.2
    snr_buckets: Tuple[float, ...] = (-5, 0, 5, 10, 20)
    max_peak: float = 0.99

    # babble
    babble_k_min: int = 3
    babble_k_max: int = 8
    babble_chunk_min_s: float = 0.5
    babble_chunk_max_s: float = 2.0
    babble_divide_by_k: bool = False


def load_audio_mono_resample(path: Path, sr: int) -> np.ndarray:
    """
    Deterministic loader: read with soundfile, convert to mono, resample with librosa if needed.
    """
    y, file_sr = sf.read(str(path), always_2d=False)
    y = y.astype(np.float32, copy=False)

    if y.ndim == 2:
        y = np.mean(y, axis=1).astype(np.float32, copy=False)

    if int(file_sr) != int(sr):
        y = librosa.resample(y, orig_sr=int(file_sr), target_sr=int(sr)).astype(np.float32, copy=False)

    return y


def list_wavs(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*.wav") if p.is_file()])


@dataclass
class MusanIndex:
    noise_files: List[Path]
    music_files: List[Path]
    speech_files: List[Path]


def build_musan_index(musan_root: Path) -> MusanIndex:
    noise_files = list_wavs(musan_root / "noise")
    music_files = list_wavs(musan_root / "music")
    speech_files = list_wavs(musan_root / "speech")

    if not noise_files or not music_files or not speech_files:
        raise ValueError("MUSAN missing/empty. Expect musan_root/{noise,music,speech}/*.wav")

    return MusanIndex(noise_files=noise_files, music_files=music_files, speech_files=speech_files)


def sample_noise_type(params: NoiseParams, rng: np.random.Generator) -> str:
    types = ["noise", "music", "babble"]
    w = np.array([params.p_noise, params.p_music, params.p_babble], dtype=np.float64)
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError("Invalid noise type probabilities")
    w = w / w.sum()
    return str(rng.choice(types, p=w))


def sample_snr_db(params: NoiseParams, rng: np.random.Generator) -> float:
    return float(rng.choice(np.array(params.snr_buckets, dtype=np.float32)))


def scale_noise_to_snr(
    x_clean: np.ndarray,
    n_raw: np.ndarray,
    speech_mask: np.ndarray,
    snr_db: float,
) -> Tuple[np.ndarray, float, float, float]:
    """
    Returns:
      n_scaled, scale_a, rms_speech, rms_noise_over_speech_before_scale
    """
    xs = x_clean[speech_mask]
    ns = n_raw[speech_mask]
    if xs.size == 0:
        raise ValueError("speech_mask is empty; cannot set speech-active SNR.")

    rms_s = rms(xs)
    rms_n = rms(ns)

    rms_n_target = rms_s / (10.0 ** (snr_db / 20.0))
    a = rms_n_target / (rms_n + 1e-12)

    n_scaled = (a * n_raw).astype(np.float32, copy=False)
    return n_scaled, float(a), float(rms_s), float(rms_n)


def _make_noise_or_music_with_replay(
    files: List[Path],
    target_len: int,
    params: NoiseParams,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Returns (n_raw, replay_dict) where replay_dict is sufficient to rebuild n_raw exactly.
    """
    path = files[int(rng.integers(0, len(files)))]
    raw = load_audio_mono_resample(path, sr=params.sr)
    n_raw, crop_tile = crop_or_tile_with_decision(raw, target_len, rng)

    replay = {
        "path": str(path),
        "crop_tile": crop_tile,
    }
    return n_raw.astype(np.float32, copy=False), replay


def _make_babble_with_replay(
    files: List[Path],
    target_len: int,
    params: NoiseParams,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Returns (n_raw, replay_dict) where replay_dict fully describes each babble component.
    """
    K = int(rng.integers(params.babble_k_min, params.babble_k_max + 1))
    out = np.zeros(target_len, dtype=np.float32)
    components: List[Dict[str, Any]] = []

    for _ in range(K):
        path = files[int(rng.integers(0, len(files)))]
        y = load_audio_mono_resample(path, sr=params.sr)

        chunk_len = int(rng.uniform(params.babble_chunk_min_s, params.babble_chunk_max_s) * params.sr)
        chunk_len = max(1, min(chunk_len, y.size))
        chunk_start = int(rng.integers(0, y.size - chunk_len + 1))

        offset = int(rng.integers(0, target_len))

        chunk = y[chunk_start:chunk_start + chunk_len].astype(np.float32, copy=False)
        end = min(target_len, offset + chunk.size)
        out[offset:end] += chunk[:end - offset]

        components.append({
            "path": str(path),
            "chunk_start": int(chunk_start),
            "chunk_len": int(chunk_len),
            "offset": int(offset),
        })

    if params.babble_divide_by_k and K > 0:
        out /= float(K)

    replay = {
        "K": int(K),
        "divide_by_k": bool(params.babble_divide_by_k),
        "components": components,
        "chunk_min_s": float(params.babble_chunk_min_s),
        "chunk_max_s": float(params.babble_chunk_max_s),
        "k_min": int(params.babble_k_min),
        "k_max": int(params.babble_k_max),
    }
    return out, replay


def add_noise_example(
    x_clean: np.ndarray,
    speech_mask: np.ndarray,
    musan: MusanIndex,
    params: NoiseParams,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Returns x_noisy and metadata sufficient to fully replay x_noisy (given x_clean).
    """
    noise_type = sample_noise_type(params, rng)
    snr_db = sample_snr_db(params, rng)

    target_len = int(x_clean.size)

    # 1) Build n_raw + replay description
    if noise_type == "noise":
        n_raw, source = _make_noise_or_music_with_replay(musan.noise_files, target_len, params, rng)
        source_key = "noise_source"
    elif noise_type == "music":
        n_raw, source = _make_noise_or_music_with_replay(musan.music_files, target_len, params, rng)
        source_key = "music_source"
    else:
        n_raw, source = _make_babble_with_replay(musan.speech_files, target_len, params, rng)
        source_key = "babble_source"

    # 2) Scale to target SNR using speech-active RMS
    n_scaled, a, rms_s, rms_n = scale_noise_to_snr(x_clean, n_raw, speech_mask, snr_db)

    # 3) Mix
    x_mix = (x_clean + n_scaled).astype(np.float32, copy=False)

    # 4) Clipping guard with explicit gain (store gain so replay is exact)
    peak_before = float(np.max(np.abs(x_mix))) if x_mix.size else 0.0
    clip_gain = 1.0
    if peak_before > params.max_peak and peak_before > 0:
        clip_gain = float(params.max_peak / peak_before)
        x_noisy = (x_mix * clip_gain).astype(np.float32, copy=False)
    else:
        x_noisy = x_mix

    # 5) Actual SNR (for sanity/debug)
    snr_actual = compute_snr_db(x_clean[speech_mask], n_scaled[speech_mask])

    meta: Dict[str, Any] = {
        "noise_type": noise_type,
        "snr_db_target": float(snr_db),
        "snr_db_actual": float(snr_actual),

        # Replay-critical: store decisions + scale + clip gain
        source_key: source,      # exact noise generation recipe
        "scale_a": float(a),     # n_scaled = scale_a * n_raw
        "clip_gain": float(clip_gain),  # x_noisy = (x_clean + n_scaled) * clip_gain

        # Helpful debugging
        "rms_speech": float(rms_s),
        "rms_noise_speechregion_before_scale": float(rms_n),
        "peak_before_clipfix": float(peak_before),
        "max_peak": float(params.max_peak),
    }

    return x_noisy, meta