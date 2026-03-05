from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import json
from typing import Dict, List

from noise_engine import build_musan_index, NoiseParams, add_noise_example
from utils_noise import speech_mask_from_intervals

# -----------------------------
# JSONL helpers
# -----------------------------

def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# -----------------------------
# Main
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 3: Add MUSAN noise to clean sequences with controlled SNR.")
    p.add_argument("--split", choices=["train", "dev", "test"], required=True)
    p.add_argument("--generated_dir", required=True, type=str, help="e.g. data/generated/train")
    p.add_argument("--musan_root", required=True, type=str, help="e.g. data/raw/musan")
    p.add_argument("--seed", type=int, default=1337)

    # noise params
    p.add_argument("--p_noise", type=float, default=0.5)
    p.add_argument("--p_music", type=float, default=0.3)
    p.add_argument("--p_babble", type=float, default=0.2)
    p.add_argument("--snr_buckets", nargs="+", type=float, default=[-5, 0, 5, 10, 20])
    p.add_argument("--max_peak", type=float, default=0.99)

    # babble
    p.add_argument("--babble_k_min", type=int, default=3)
    p.add_argument("--babble_k_max", type=int, default=8)
    p.add_argument("--babble_chunk_min_s", type=float, default=0.5)
    p.add_argument("--babble_chunk_max_s", type=float, default=2.0)
    p.add_argument("--babble_divide_by_k", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    gen_dir = Path(args.generated_dir).expanduser().resolve()
    manifest_in = gen_dir / "manifests" / f"{args.split}_manifest.jsonl"
    if not manifest_in.exists():
        raise FileNotFoundError(f"Missing Step2 manifest: {manifest_in}")

    rows = read_jsonl(manifest_in)
    if not rows:
        raise ValueError(f"Empty manifest: {manifest_in}")

    musan_root = Path(args.musan_root).expanduser().resolve()
    musan = build_musan_index(musan_root)

    out_noisy = gen_dir / "noisy_audio"
    out_noisy.mkdir(parents=True, exist_ok=True)

    params = NoiseParams(
        sr=int(rows[0]["sr"]),
        p_noise=args.p_noise,
        p_music=args.p_music,
        p_babble=args.p_babble,
        snr_buckets=tuple(float(x) for x in args.snr_buckets),
        max_peak=float(args.max_peak),
        babble_k_min=args.babble_k_min,
        babble_k_max=args.babble_k_max,
        babble_chunk_min_s=float(args.babble_chunk_min_s),
        babble_chunk_max_s=float(args.babble_chunk_max_s),
        babble_divide_by_k=bool(args.babble_divide_by_k),
    )

    out_rows: List[Dict] = []

    for i, r in enumerate(rows):
        ex_id = r["ex_id"]
        sr = int(r["sr"])

        # per-example determinism (still good to keep)
        ex_seed = int(r.get("ex_seed", 0))
        ex_rng = np.random.default_rng((ex_seed ^ args.seed) & 0x7FFFFFFF)

        clean_path = gen_dir / r["clean_audio_path"]
        labels_path = gen_dir / r["labels_path"]

        x_clean = np.load(clean_path).astype(np.float32, copy=False)
        _y = np.load(labels_path)  # unchanged, but sanity load

        speech_mask = speech_mask_from_intervals(int(x_clean.size), r["speech_intervals"])

        x_noisy, noise_meta = add_noise_example(
            x_clean=x_clean,
            speech_mask=speech_mask,
            musan=musan,
            params=params,
            rng=ex_rng,
        )

        noisy_path = out_noisy / f"{ex_id}.npy"
        np.save(noisy_path, x_noisy.astype(np.float32, copy=False))

        out_rows.append({
            "ex_id": ex_id,
            "split": r.get("split", args.split),
            "sr": sr,

            "noisy_audio_path": str(noisy_path.relative_to(gen_dir)),
            "clean_audio_path": r["clean_audio_path"],
            "labels_path": r["labels_path"],
            "num_samples": r["num_samples"],

            # keep Step2 context for traceability
            "utterances": r.get("utterances"),
            "speech_intervals": r["speech_intervals"],
            "silences": r.get("silences"),
            "frame_params": r.get("frame_params"),
            "standardize": r.get("standardize"),
            "source": r.get("source"),

            # step3 additions (now fully replayable)
            "noise": noise_meta,

            "seed": args.seed,
            "ex_seed": ex_seed,
        })

        if (i + 1) % 200 == 0:
            print(f"[OK] Noised {i+1}/{len(rows)}")

    manifest_out = gen_dir / "manifests" / f"{args.split}_noisy_manifest.jsonl"
    write_jsonl(out_rows, manifest_out)

    print(f"\n[Done] Wrote noisy manifest: {manifest_out}")
    print(f"[Done] Noisy audio dir: {out_noisy}")


if __name__ == "__main__":
    main()