# scripts/03_add_noise_all.ps1
# Run from repo root:
#   .\scripts\03_add_noise_all.ps1
#
# This runs Step 3 (add MUSAN noise) for train/dev/test using the Step 2 outputs.

$ErrorActionPreference = "Stop"

# -------- paths (edit if your folders differ) --------
$MUSAN_ROOT = "data\raw\musan"
$NOISE_SCRIPT = "src\03_add_noise\add_musan_noise.py"
$OUT_BASE = "data\generated"

# -------- seed + params --------
$SEED = 1337

# Noise-type weights (must sum to > 0; script will normalize)
$P_NOISE  = 0.5
$P_MUSIC  = 0.3
$P_BABBLE = 0.2

# SNR buckets (space-separated)
$SNR_BUCKETS = @("-5", "0", "5", "10", "20")

# Babble controls
$BABBLE_DIVIDE_BY_K = $true
$BABBLE_K_MIN = 3
$BABBLE_K_MAX = 8
$BABBLE_CHUNK_MIN_S = 0.5
$BABBLE_CHUNK_MAX_S = 2.0

# Clipping guard
$MAX_PEAK = 0.99

$COMMON_ARGS = @(
  "--musan_root", $MUSAN_ROOT,
  "--seed", "$SEED",
  "--p_noise", "$P_NOISE",
  "--p_music", "$P_MUSIC",
  "--p_babble", "$P_BABBLE",
  "--max_peak", "$MAX_PEAK",
  "--babble_k_min", "$BABBLE_K_MIN",
  "--babble_k_max", "$BABBLE_K_MAX",
  "--babble_chunk_min_s", "$BABBLE_CHUNK_MIN_S",
  "--babble_chunk_max_s", "$BABBLE_CHUNK_MAX_S"
) + @("--snr_buckets") + $SNR_BUCKETS

if ($BABBLE_DIVIDE_BY_K) {
  $COMMON_ARGS += "--babble_divide_by_k"
}

function Run-Noise {
  param(
    [string]$Split
  )

  $GenDir = Join-Path $OUT_BASE $Split
  $ManifestIn  = Join-Path $GenDir "manifests\$Split`_manifest.jsonl"
  $ManifestOut = Join-Path $GenDir "manifests\$Split`_noisy_manifest.jsonl"

  Write-Host "===================================================="
  Write-Host "Adding MUSAN noise split=$Split"
  Write-Host "GenDir:      $GenDir"
  Write-Host "Manifest In: $ManifestIn"
  Write-Host "Manifest Out:$ManifestOut"
  Write-Host "===================================================="

  if (!(Test-Path $ManifestIn)) {
    throw "Step 2 manifest not found. Run Step 2 first: $ManifestIn"
  }

  # Optional: clean regenerate noisy outputs only
  $NoisyDir = Join-Path $GenDir "noisy_audio"
  if (Test-Path $NoisyDir) {
    Remove-Item -Recurse -Force $NoisyDir
  }

  python $NOISE_SCRIPT `
    --split $Split `
    --generated_dir $GenDir `
    @COMMON_ARGS

  # quick check
  if (!(Test-Path $ManifestOut)) {
    throw "Noisy manifest not found after noise generation: $ManifestOut"
  } else {
    Write-Host "[OK] Noisy manifest: $ManifestOut"
  }

  if (!(Test-Path $NoisyDir)) {
    throw "Noisy audio dir not found: $NoisyDir"
  } else {
    $Count = (Get-ChildItem -Path $NoisyDir -Filter "*.npy" | Measure-Object).Count
    Write-Host "[OK] Noisy audio files: $Count in $NoisyDir"
  }
}

# -------- run all splits --------
Run-Noise -Split "train"
Run-Noise -Split "dev"
Run-Noise -Split "test"

Write-Host "`nAll done."