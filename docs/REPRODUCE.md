# Reproducing the NoTime submission

One command regenerates the submitted file from this repository, the competition bundle and the external datasets
listed in section 2. Requirements and installation are in the [README](../README.md); the method is described in
[`NoTime.md`](NoTime.md).

## 0. The one command

```bash
MTAH_DATA_DIR=/path/to/bundle .venv/bin/python run.py     # download, convert, train, run the video steps, predict, decode, validate
```

| flag | what it skips |
|---|---|
| (none) | nothing: every missing raw dataset is downloaded, every missing npz converted |
| `--skip-download` | the download step (raw data already under `data_external/<source>/`) |
| `--from-cache` | download and conversion (`data_external/common/*.npz` present); cached feature views are reused |
| `--decode-only` | also training and the video steps: re-decodes the saved member signals (seconds) |
| `--dry-run` | everything: prints the steps and the exact commands |
| `--only STEP ...` | runs only the named steps |
| `--final FILE` | reads another recipe file instead of `FINAL.json` |
| `--out PATH` | writes the CSV elsewhere (default: `FINAL.json` `output`) |
| `--allow-drift` | on another GPU / driver: an intermediate vision or player file whose content hash differs from the build is logged as DRIFT and the run continues to a CSV; the final sha256 check still reports MATCH / DIFFERS (exit 1 on DIFFERS). Without it the run stops at the first differing intermediate file |

Steps (`run.py --dry-run` prints them with the exact commands). Steps that the recipe does not need log "skipped".

| # | step | what it does for this recipe |
|---|---|---|
| 1 | check | packages, CUDA GPU, provided-match files present |
| 2 | download | raw external data (`src/external/download_all.py`, `download_pff.py`) |
| 3 | convert | raw -> `data_external/common/<source>_<match>.npz` (converter table below) |
| 4 | inputs | content hash (arrays only) of the 102 external training matches and of the parsed provided tracking, against [`external_manifest.json`](external_manifest.json) |
| 5 | assets | shipped degradation assets present, feature hashes `fb555aa8d1` / `bf7cc90c6a` |
| 6, 7 | gamestate, corrector | skipped (this recipe has no game-state member and no corrector) |
| 8 | vision | the refined video ball `cache/vision/ball_video_refD3.npz`: ball detector (shipped weights) on the tactical video, then the shipped diffusion trajectory refiner; content-hash checked |
| 9 | players | both player sources, each content-hash checked. Source A: person detection on the tactical video (`.venv-vision` `src/vision/run_players.py detect`, shipped detector, NVDEC + fp16 YOLO) into `cache/vision/players_cands/`, then `python -m mtah.player_fix build` (team correction, CPU) -> `cache/player_fix/atd_tactical_T2team.npz`. Source B: our tracker on the same detections (`.venv-vision` `python -m vision.d4track --cands cache/vision/players_cands --work cache/player_fix/d4h`: re-ID embeddings on the GPU, pieces, learned linker; shipped weights, sha-checked), then mode OWN of `mtah.player_fix` -> `cache/player_fix/atd_tactical_OWN_d4h.npz` |
| 10 | oof | skipped (no stacked experts) |
| 11 | signals | `python -m mtah.final_signals` (member 1) and `python -m mtah.run_refD final` (member 2): each member is trained once on the 102 external matches (CPU LightGBM, CUDA BiGRU / transformer seeds, checkpointed per unit in `runs/ckpt/`) and then predicts the provided match twice, `--tracking <source A file> --tracking OWN=<source B file>` -> `runs/final/<member>.F3.signals.npz` and `<member>.F3.OWN.signals.npz` |
| 12 | decode | `python -m mtah.merge_final` with the recipe's `merge_sources` (per member and head 0.625 · source A + 0.375 · source B), then the member blend, the Viterbi decoder and rule step B2 (on the provided tracking, the Smart Tagging XML and the refined video ball, all hash-checked) |
| 13 | validate | `python evaluation/validate_submission.py <csv>` |
| 14 | sha256 | prints the hash and compares it with `FINAL.json` `expected.csv_sha256` (**exit code 1 if it differs**) |

An interrupted run resumes: every trained unit is saved in `runs/ckpt/` with the random-number state it leaves, and
reused; with or without reuse the CSV is bit-identical.

## 1. Hardware and runtime

Measured on an Intel i7-13700K (24 threads), 46 GB RAM, NVIDIA RTX 5080 16 GB,
from a fresh clone (creating the two environments takes about 15 min more).

| | wall time | peak RSS |
|---|---|---|
| `run.py`, all steps | 4 h 0 min | 20.7 GB |
| of which the players step (detection, correction, tracker) | 54 min | |

## 2. Data

### 2.1 Competition bundle (not redistributed)

Unpack it into `data/` (or set `MTAH_DATA_DIR`). The recipe reads:
- `data/tactical/BAR-ATH_tactical_atd.txt` and `BAR-ATH_tactical_atd.xml` (parsed once into `cache/atd_tactical.npz`;
  step 4 checks the parsed arrays against the build's hash);
- the tactical video `BAR-ATH_tactical.mp4` and `BAR-ATH_tactical_homography.csv` (steps 8 and 9; their products are
  content-hash checked);
- `BAR-ATH_tactical_smart_tagging.xml` (rule step, sha256-checked before decoding).

### 2.2 External datasets (training only)

All land under `data_external/` (git-ignored, never redistributed).

| source | used as | download | access | citation / licence |
|---|---|---|---|---|
| IDSSE / DFL open data, 7 Bundesliga matches | dead + team heads | figshare API, article 28196177, md5-checked | open | Bassek, Rein, Weber & Memmert (2025), Scientific Data 12:195, doi:10.1038/s41597-025-04505-y; CC BY 4.0 |
| PathCRF repository (DFL events synchronised to the tracking) | controlled-possession relabelling of IDSSE | `git clone https://github.com/hyunsungkim-ds/pathcrf` @ `1b7c6f8` | open | Kim et al. (2026), PathCRF, KDD; code MPL-2.0 |
| ELASTIC benchmark GT (3 matches) | check of the IDSSE relabelling only | `git clone https://github.com/hyunsungkim-ds/elastic` @ `e245224` | open | Kim et al., ELASTIC; CC BY 4.0 |
| Metrica Sports sample data, 3 games | team heads; rule-step timing | `git clone https://github.com/metrica-sports/sample-data` @ `e706dd5` | open | Metrica Sports sample data |
| SkillCorner Open Data, 20 A-League 2024/25 matches | team heads | raw.githubusercontent / Git LFS at `4340d27` | open | https://github.com/SkillCorner/opendata, MIT |
| SkillCorner Open Data, 2020 release, 9 matches | natural ball-gap pool only | raw.githubusercontent at `1b99d40` | open | same, MIT |
| PFF FC (Gradient Sports) FIFA World Cup 2022, 64 matches | team heads (largest source) | `src/external/download_pff.py` (the Google Drive folder PFF shares after the form) | **request form** at [blog.fc.pff.com](https://www.blog.fc.pff.com/blog/enhanced-2022-world-cup-dataset) | PFF FC / Gradient Sports |
| SoccerTrack v2, 10 university matches | dead + team heads (8 of 10 have provider ball-status labels) | Hugging Face `atomscott/soccertrack-v2`, tag `v1.2` | **Hugging Face account** if the dataset asks | arXiv:2508.01802; CC BY 4.0 |

The detectors, the ball refiner and our tracker's re-ID embedding and linker were trained on SoccerNet-GSR 2024 train
(GPL-3.0, dataset card), Roboflow Football-Player-Detection v3 (CC BY 4.0) and external ball tracks; their weights are
shipped (section 4), so these datasets are not downloaded by `run.py`.

ELASTIC is downloaded by default only because `idsse_ctrl_labels.py` reports the agreement of our IDSSE relabelling
with it; the CSV does not depend on it. `src/external/` also holds download scripts for datasets that were explored
and are not used by this recipe; `run.py` never calls them.

Converters (step 3, all with `python -I` because they read downloaded files): `src/external/convert_idsse.py`,
`idsse_ctrl_labels.py`, `convert_metrica.py`, `convert_skillcorner.py`, `convert_pff.py --jobs 4`,
`convert_stv2.py --jobs 3` (two SoccerTrack v2 matches have no provider label and are excluded). Label construction
per source: [`NoTime.md`](NoTime.md) section 3.

**Important for bit-exactness.** The degradation seed of a canonical-source match is 7 × its index in the sorted list
of idsse-ctrl / metrica / pff / skillcorner npz files, so `data_external/common` must contain exactly those 94 files
(step 4 checks the list). Other npz files there do not move the seeds.

## 3. What is deterministic, and how

| component | determinism |
|---|---|
| external conversion | pure numpy / pandas; npz files differ in zip timestamps, their arrays must not (step 4 compares array hashes) |
| degradation `ext_rand` | fixed seeds per match; the gap pool is shipped |
| LightGBM | CPU only (`MTAH_LGB_DEVICE=cpu` forced by `run.py`), `deterministic=True`, `force_row_wise=True`, `n_jobs=6`, `random_state=0`. The CUDA build of LightGBM is not reproducible and is never used. Keep `n_jobs=6`: LightGBM's determinism is per thread count |
| possession models: BiGRU and transformer seeds | CUDA with `torch.use_deterministic_algorithms(True)`, cuDNN deterministic and benchmark off, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, TF32 off, fixed seeds. There is no CPU path. Bit-identical reruns verified on the build GPU (RTX 5080, torch 2.11.0+cu128) |
| ball and person detection | fp16 YOLO with cuDNN benchmark off (no deterministic mode); bit-identical run to run and with a different batch composition on the build GPU |
| team correction (source A) | numpy / scipy / scikit-learn, k-means single-threaded. It is not invariant to detection noise (±1 px boxes change 0.03% of player-frames), so it relies on the bit-exact detection; its content hash is checked |
| our tracker (source B) | the re-ID CNN and the linker run without PyTorch's deterministic mode (default settings); the batch composition of the re-ID embeddings is fixed by the frame loop, the linker's sampler draws from a per-window generator seeded 0, sorts are stable. Two from-scratch rebuilds from `src/` were bit-identical on the build GPU; step 9 checks the content hash of the tracker output (`83ca0482...`) and of source B |
| fusion, member blend, decoder, rule step | float64, no randomness |

On another GPU model or driver the intermediate files may differ in low-order bits and therefore in a few frames; run `run.py --allow-drift` (section 0) to obtain a CSV anyway and compare it with the expected sha256.

## 4. Shipped assets

Trained on external data only and committed because their training is not bit-deterministic or runs a GPU-only chain
(`run.py` checks every sha256):

| asset | path | sha256 |
|---|---|---|
| degradation gap pool (integer gap lengths of external providers and of our ball detector on SoccerTrack v2 video) | `cache/degprof/ext_rand_gaps.npz` | feature hash `fb555aa8d1` |
| refiner error profile (held-out SoccerTrack v2 video) | `cache/degprof/refD3_ext_sequences.npz` | feature hash `bf7cc90c6a` |
| YOLO11s ball detector (SoccerNet-GSR + Roboflow) | `cache/vision/runs/ball/weights/best.pt` | `dd78e2bc...` |
| ball trajectory refiner (external ball tracks) | `cache/vision/refine/runs/all_v3tied/model_20000.pt` | `dceacd24...` |
| YOLO11s person detector (SoccerNet-GSR + Roboflow) | `cache/vision/runs/players_s/weights/best.pt` | `3f8be2bf...` |
| our re-ID CNN (SoccerNet-GSR train) | `cache/vision/track/k2_reid_h3.pt` | `9c7b240a...` |
| our learned linker (SoccerNet-GSR train, a-priori degradations) | `cache/vision/track/d4h_linker.pt` | `281103f9...` |

Training code of the two tracker weights (not run by `run.py`; for inspection and retraining): `tools/trackK2/gsr_pack.py`,
`gsr_eval.py`, `reid_crops.py`, `reid_train.py` (re-ID CNN) and `tools/trackK2/k2track.py`, `tools/trackD4/feats.py`, `build.py`,
`build2.py`, `build4.py`, `aug_emb.py`, `score.py`, `linker.py`, `match.py`, `gsr_compose.py` (linker), with the
sibling modules they import (e.g. `tools/trackK2/run_match.py`). The data paths at the top of these scripts are the
authors' storage folders: point them at your own before retraining. The detectors' and the refiner's training code is
in `src/vision/`.
