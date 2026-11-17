# Third-party notices

## Licences in this repository

- **The organisers' files** (`evaluation/`, `starter/`, `games.py`) keep their own MIT notice ([LICENSE](LICENSE)).
  It does not cover the match data, which is not in this repository, nor the shipped weights listed below.
- **Shipped detector weights**: `cache/vision/runs/ball/weights/best.pt` and
  `cache/vision/runs/players_s/weights/best.pt` were trained with Ultralytics YOLO11 from Ultralytics'
  COCO-pretrained `yolo11s.pt` and are distributed under the **GNU AGPL-3.0**
  (full text: [LICENSES/AGPL-3.0.txt](LICENSES/AGPL-3.0.txt)), as Ultralytics requires for models trained with its software.
  Their complete training code is in `src/vision/` (`prep_dataset.py`, `train.py`, `train_players.py`). When the code in `src/vision/` that imports `ultralytics` is
  distributed together with Ultralytics, the combination is subject to AGPL-3.0.
- **Shipped tracker weights**: `cache/vision/track/k2_reid_h3.pt` and `cache/vision/track/d4h_linker.pt` were
  trained only on SoccerNet-GSR 2024 train, whose dataset card declares GPL-3.0. We distribute these two files
  under the **GNU GPL-3.0** (full text: [LICENSES/GPL-3.0.txt](LICENSES/GPL-3.0.txt)). Their training code is in
  `tools/trackK2/` and `tools/trackD4/`.
- **Shipped ball refiner and derived statistics**: `cache/vision/refine/runs/all_v3tied/model_20000.pt`,
  `cache/degprof/*.npz` and the aggregate JSON statistics under `notes/avo/` and in `notes/trackG_metrica_timing.json` are derived from the external
  datasets below. They contain no tracking, event or video record of any dataset. The attributions below apply.

## External data (not redistributed; downloaded by `run.py` or used to train the shipped weights)

| dataset | licence / terms | used for | attribution |
|---|---|---|---|
| IDSSE / DFL open data, figshare 10.6084/m9.figshare.28196177 | CC BY 4.0 (with the authorization of the DFL) | possession models, ball refiner | Bassek, M., Rein, R., Weber, H. & Memmert, D. (2025). An integrated dataset of spatiotemporal and event data in elite soccer. *Scientific Data* 12, 195. doi:10.1038/s41597-025-04505-y |
| PathCRF repository, synced DFL events (github.com/hyunsungkim-ds/pathcrf) | MPL-2.0 repository; events derived from the CC BY 4.0 DFL data | IDSSE relabelling | Kim, H., Lee, K., Seo, S., Ko, S.-K., Yoon, J. & Park, C. (2026). PathCRF: Ball-Free Soccer Event Detection via Possession Path Inference from Player Trajectories. KDD 2026. doi:10.1145/3770855.3818152 |
| ELASTIC benchmark (github.com/hyunsungkim-ds/elastic) | data CC BY 4.0, code MPL-2.0 | cross-check of the IDSSE relabelling | Kim, H., Choi, H., Lee, K., Seo, S., Boomstra, T., Yoon, J. & Park, C. (2026). ELASTIC: Trajectory-Based Synchronization of Event and Tracking Data in Soccer. CIKM 2026. arXiv:2608.30227 |
| PFF FC (Gradient Sports) FIFA World Cup 2022 data | PFF FC's free-access terms (obtained through PFF's request form) | possession models, ball refiner, gap statistics | Data: PFF FC / Gradient Sports (@PFF_FC) |
| SkillCorner Open Data (github.com/SkillCorner/opendata) | MIT, Copyright (c) 2020 SkillCorner | possession models, gap statistics | Data: SkillCorner |
| Metrica Sports sample data (github.com/metrica-sports/sample-data) | Metrica Sports' README terms | possession models, ball refiner, set-piece timing | Data: Metrica Sports sample data |
| SoccerTrack v2 (huggingface.co/datasets/atomscott/soccertrack-v2) | CC BY 4.0 | possession models, ball refiner, gap statistics | Scott, A., Uchida, I., Kuroda, K., Kim, Y. & Fujii, K. (2025). SoccerTrack v2: A Full-Pitch Panoramic Video Dataset for Game State Reconstruction and Ball Action Spotting. arXiv:2508.01802 |
| SoccerNet Game State Reconstruction 2024 (huggingface.co/datasets/SoccerNet/SN-GSR-2024) | GPL-3.0 (dataset card) | detectors, re-ID CNN, linker (train); tracker benchmark (valid) | Somers, V. et al. (2024). SoccerNet Game State Reconstruction: End-to-End Athlete Tracking and Identification on a Minimap. CVPR Workshops (CVsports) 2024 |
| Football-Player-Detection v3 (universe.roboflow.com/augmented-startups/football-player-detection-kucab) | CC BY 4.0 | detectors | Augmented Startups (2022). Football-Player-Detection Dataset. Roboflow Universe |
| COCO (through Ultralytics' pretrained `yolo11s.pt`) | as released by Ultralytics / COCO consortium | detector initialisation | Lin, T.-Y. et al. (2014). Microsoft COCO: Common Objects in Context. ECCV |

## Software (installed from the lock files, not redistributed)

Ultralytics YOLO11 8.4.208 (AGPL-3.0; Jocher, G. & Qiu, J. (2024), Ultralytics YOLO11,
github.com/ultralytics/ultralytics), PyTorch and torchvision (BSD-3-Clause), LightGBM (MIT), NumPy, SciPy,
pandas, scikit-learn (BSD), kloppy (BSD-3-Clause), OpenCV (Apache-2.0), huggingface_hub (Apache-2.0),
gdown (MIT). `ffmpeg` is a system dependency (LGPL/GPL) and is not shipped.
