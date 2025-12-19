# Basketball Analysis System: End-to-End Build Guide

This guide walks through building and running the basketball video analysis pipeline contained in this repository. It mirrors the methodology used to train detectors, stitch together tracking and analytics components, and render the annotated outputs.

## Phase 1: Environment & Project Layout

- **Folder structure** (already present in this repo):
  - `input_videos/` for sample or custom footage.
  - `models/` for the trained weights (`player_detector.pt`, `ball_detector_model.pt`, `court_keypoint_detector.pt`).
  - Processing modules: `trackers/`, `team_assigner/`, `ball_aquisition/`, `pass_and_interception_detector/`, `court_keypoint_detector/`, `tactical_view_converter/`, `speed_and_distance_calculator/`, `drawers/`, `utils/`, plus `configs/` for defaults. The orchestration logic now lives in `pipeline_runner.py` so it can be imported directly by API services (e.g., RunPod) or the CLI.
- **Dependencies**: create a virtual environment and install `pip install -r requirements.txt`. Key packages include `ultralytics`, `opencv-python`, `supervision`, `pandas`, `numpy`, and `transformers` (for zero-shot jersey classification).
- **Inputs**: place your videos in `input_videos/`. The reference demo uses three clips—one each for distance/speed, passes, and interceptions.

## Phase 2: Train or Download Detection Models

> Training requires a GPU (Colab T4 or better is recommended). If you prefer not to train, download the provided weights linked in the README and place them in `models/`.

1. Open the notebooks in `training_notebooks/` on Colab and install the runtime dependencies:
   ```bash
   pip install roboflow ultralytics
   ```
2. **Player detector** (YOLO v5/11): download the Roboflow dataset referenced in the notebook, then run training (example command inside the notebook):
   ```bash
   yolo detect train model=yolov5l6u.pt data={dataset.location}/data.yaml epochs=100 imgsz=640 batch=8 plots=True
   ```
   Export `runs/detect/train/weights/best.pt` as `models/player_detector.pt`.
3. **Ball detector**: duplicate the player notebook, raise epochs (e.g., `epochs=250`), and save the resulting `best.pt` as `models/ball_detector_model.pt`.
4. **Court keypoint detector** (YOLOv8 pose): use the court keypoint dataset from Roboflow and train with a pose model, for example:
   ```bash
   yolo task=pose mode=train model=yolov8x-pose.pt data={dataset.location}/data.yaml epochs=500 imgsz=640 batch=16
   ```
   Save the resulting weights as `models/court_keypoint_detector.pt`.

## Phase 3: Tracking Players and Ball

- `trackers/player_tracker.py` loads `player_detector.pt`, batches inference, converts detections to Supervision format, and uses ByteTrack to maintain stable player IDs. Results are cached with `utils.stubs_utils` to avoid recomputation.
- `trackers/ball_tracker.py` runs the ball detector, keeps the highest-confidence ball per frame, prunes implausible jumps (`remove_wrong_detections`), and linearly interpolates gaps (`interpolate_ball_positions`).
- Stub files default to `stubs/` (configurable via `--stub_path` in `main.py`). Delete stubs to force fresh inference.

## Phase 4: Zero-Shot Team Assignment

- `team_assigner/team_assigner.py` uses the Fashion-CLIP model (`patrickjohncyh/fashion-clip`) to classify cropped player images as `"white shirt"` vs. `"dark blue shirt"` by default.
- Team IDs are cached per `player_id` and reset every 50 frames to prevent stale classifications. Results can also be stubbed for reuse.
- To change jersey prompts, adjust `team_1_class_name` and `team_2_class_name` when constructing `TeamAssigner`.

## Phase 5: Ball Possession, Passes, and Interceptions

- `ball_aquisition/ball_aquisition_detector.py` evaluates containment (IoU of ball within player box) and proximity to choose the ball holder. Possession is confirmed only after a minimum frame count to reduce flicker.
- `pass_and_interception_detector/pass_and_interception_detector.py` marks a **pass** when possession switches between players on the same team, and an **interception** when teams differ. Outputs drive the on-frame overlays.

## Phase 6: Court Keypoints and Tactical View

- `court_keypoint_detector/` runs the keypoint pose model, returning per-frame court landmarks. `validate_keypoints` drops geometrically inconsistent points.
- `tactical_view_converter/` builds a homography from detected keypoints to a 28m x 15m tactical map (`images/basketball_court.png`). Player foot positions are transformed to top-down coordinates for plotting and downstream distance calculations.

## Phase 7: Speed & Distance Metrics

- `speed_and_distance_calculator/` converts pixel movement on the tactical map into meters using the known court dimensions. It accumulates per-frame distance and estimates km/h speeds over a rolling window.

## Phase 8: Running the Pipeline

### Local or Server (including RunPod/API reuse)

1. Place your trained weights in `models/` (or use the downloadable defaults) and your input video in `input_videos/`.
2. Run via CLI:
   ```bash
   python main.py input_videos/video1.mp4 --output_video output_videos/output_result.avi --stub_path stubs
   ```
   - `--stub_path` controls where detection/tracking caches are stored.
   - `--output_video` sets the annotated output path (default configured in `configs/`).
3. Programmatic use (e.g., RunPod handler):
   ```python
   from pipeline_runner import run_analysis
   run_analysis("input_videos/video1.mp4", output_video_path="output_videos/output_result.avi", stub_path="stubs")
   ```

### Google Colab (GPU)

1. Upload or clone the repo in Colab and switch to a GPU runtime.
2. Bootstrap dependencies and download pretrained weights in a cell:
   ```bash
   !python colab_setup.py --install-deps --download-models --models-dir models
   ```
3. Upload your input video to `input_videos/` (or mount Drive) and run:
   ```bash
   !python main.py input_videos/video1.mp4 --output_video output_videos/output_result.avi --stub_path stubs
   ```
4. The pipeline performs detection, tracking, team assignment, possession/pass/interception detection, tactical-view projection, speed/distance estimation, and writes the fully annotated video to `output_videos/`.

## Troubleshooting Tips

- If outputs seem stale, clear the stub files in your stub directory to force fresh inference.
- The zero-shot team classifier relies on clear jersey color contrast; adjust the class prompts or lighting/contrast of the input if teams are misassigned.
- Homography requires at least four valid court keypoints. If tactical dots disappear, verify the keypoint detector weights and video resolution.
