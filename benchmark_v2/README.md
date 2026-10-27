Run a mapping service:

cd scenescape/mapping
git apply 0001-Run-it-on-CPU.patch
make mapping
docker compose --profile mapping up -d mapping

From the repository root, install the mapping client's dependencies and generate the scenes:

```sh
python3 -m pip install -r benchmark_v2/requirements.txt
python3 benchmark_v2/mapanything.py
```

For each `data/chute0*/cam2.avi`, the script sends the first frame to MapAnything
and writes these files beside the video:

- `scene.glb`: mesh aligned to a Z-up floor.
- `camera.json`: deterministic camera ID, aligned camera-to-world pose (XYZW quaternion),
  estimated intrinsics, zero assumed distortion, and original video resolution.
- `alignment.json`: shared mesh/camera transform, floor plane, and scale.
- `scene.raw.glb` and `mapping.json`: original reconstruction and camera estimates.

Camera calibration and metric scale are estimates, not measured ground truth.
Automatic floor alignment can select a wall instead of the floor. Review the scene
and camera together before interpreting fall-detection results.

To realign existing outputs without calling MapAnything again:

```sh
python3 benchmark_v2/mapanything.py --realign
```

Optional `--meters-per-unit` and `--floor-plane NX NY NZ D` overrides apply to every
selected chute. The floor plane is expressed in the raw GLB coordinate system,
with its normal pointing toward the camera. `--data-dir` selects the input root.

Mapping only generates files; registration and replay use the Phase 2 command below.

Run person detection on every video in `benchmark_v2/data`:

```sh
python3 benchmark_v2/detect_videos.py
```

Uses `models/yolo11s-pose/FP16/yolo11s-pose.xml` (override with `--model`) with embedded
OpenVINO preprocessing and labels. Only `person` detections are saved; all other
classes are removed, retaining empty frames. The JSON results are written under
`benchmark_v2/detections-yolo11s-pose`, preserving the input subdirectories. Previous
retail and non-pose detections remain untouched under `benchmark_v2/detections`
and `benchmark_v2/detections-yolo11s`.

## Inspect Detection Boxes

Overlay the saved person detections without running inference again (requires
FFmpeg and ffprobe, with the libwebp_anim encoder for WebP):

```sh
python3 benchmark_v2/overlay_detections.py
python3 benchmark_v2/overlay_detections.py --format mp4
```

Outputs are `benchmark_v2/overlays-yolo11s-pose/<chute>/cam2.webp` or `cam2.mp4`.
Green boxes appear only on matching source timestamps, never held across gaps.
WebP previews loop at 20 fps and can omit brief detections; MP4 preserves the
source frame rate for frame-by-frame inspection. Both preserve the original
resolution and playback speed, omit audio, and leave the inputs unchanged.
These are raw detector boxes, not SceneScape tracks or fall classifications.

## Replay Saved Detections (Phase 2)

Start the current Controller stack with the benchmark-owned override:

```sh
docker compose --project-directory "$PWD/scenescape" \
    -f scenescape/docker-compose.yml -f benchmark_v2/compose.override.yml \
    --profile controller up -d web scene analytics
```
# TODO remove information about thread settings
The override enables observed camera bounds on the topic used by the fall detector
without overriding math-library thread settings. After updating SceneScape sources, rebuild the
Manager image with `make -C scenescape manager`, then recreate `web` with the same
Compose files. The updated Manager supports resolution and distortion through REST;
no direct database workaround is used.

```sh
python3 benchmark_v2/run_fall_detection.py
python3 benchmark_v2/run_fall_detection.py --chute chute01
```

To replay twice as slowly and keep the original results:

```sh
python3 benchmark_v2/run_fall_detection.py --speed 0.5 --output benchmark_v2/results-half-speed
```

`--speed` scales both publication intervals and MQTT timestamps, so tracking and
fall-detector smoothing use the slowed timeline. Result `video_time_s` values still
refer to the original video, and `playback_speed` records the selected multiplier.
Video files and saved detection files are not modified.

The runner uses the existing fall-detection image and the running web container's
Docker network and mounted credentials. It authenticates through the Manager API,
registers/reuses `benchmark-v2-<chute>` scenes, and uses separate `v2-` camera IDs
so manually registered cameras are not overwritten. Scenes and cameras remain in
SceneScape after execution; temporary replay workers and detector processes exit.

Saved detections are replayed at their original PTS spacing, without video playback
or inference. Each detection gets a frame-local ID; SceneScape assigns persistent
person IDs. The runner resets the uploaded mesh transform to identity so it stays
aligned with `camera.json`.

Replay defaults to `detections-yolo11s-pose`. Results are saved as
`results-yolo11s-pose/<chute>.json`, with captured scene/fall messages and
detector logs under `results-yolo11s-pose/<chute>/`. Each transition into `fallen` contains the
person ID, UTC replay timestamp, approximate video time, and metrics snapshot.
`first_observation: true` marks people first observed already fallen. Zero events
does not validate calibration or prove that no fall occurred. A run with no observed
scene people is marked failed rather than reported as a successful zero-fall run.

## Evaluate Fall Detection

The source videos are sped up. Choose a slowdown factor X greater than 1, shared
by every video in a run. Replay with `--speed 1/X`; full-speed results cannot be
converted into slowed results after the fact because tracking and smoothing
depend on input timing. Keep different models and slowdown factors in separate
result/report directories.

Annotate each `data/<chute>/ground_truth.json` using the source file's seconds:

```json
{
    "reviewed": true,
    "falls": [{"start_s": 9, "end_s": 12}]
}
```

`start_s` is when falling starts; `end_s` is when the person stands up (or the
video ends). A person already fallen at the beginning has `start_s: 0`. Mark
`reviewed: true` only after reviewing the whole clip. An empty `falls` list means
no falls only when reviewed. Missing review flags/annotations are unreviewed.
Do not rescale labels or source files. Use the source video's time display when
labeling, even if you slow playback to inspect it.

For example, at X=4 (an example, not an assumed dataset speedup):

```sh
python3 benchmark_v2/run_fall_detection.py --detections benchmark_v2/detections-yolo11s-pose \
    --speed 0.25 --output benchmark_v2/results-yolo11s-pose-quarter-speed
python3 benchmark_v2/evaluate.py --results benchmark_v2/results-yolo11s-pose-quarter-speed \
    --slowdown-factor 4 --output benchmark_v2/reports/yolo11s-pose-quarter-speed
```

The runner now defaults to `detections-yolo11s-pose`; the example keeps the input
explicit for reproducibility. The saved `results-quarter-speed` runs contain no
fall alerts; the later `results-yolo11s-pose-quarter-speed` runs do contain alerts.
These are different experiments, not interchangeable result directories. Evaluating
saved outputs does not rerun or change the detector.

Evaluation is offline, requires only Python's standard library and `ffprobe`,
and does not start Docker or inference. Add `--chute chute01` to select one clip.
The selected slowdown must match the saved `playback_speed` in every scored run.

Matching uses source-video seconds, including the default `--tolerance-s 1` at
both ends of an episode. The runner's `video_time_s` is already mapped back to
source seconds: never scale it twice. The 9-12s example becomes 36-48s during
4x-slowed replay, but matching still uses 8-13 source seconds with tolerance.

- First matching alert per episode: TP. No matching alert: FN.
- Additional alerts: FP, separated into duplicates and outside-window alerts.
- Match alerts chronologically to the earliest eligible unmatched episode.
    Overlapping tolerance windows are flagged for review.
- Report precision, recall and F1 from total counts, not averaged clip scores.
    Undefined ratios are N/A. No FP/hour or frame accuracy.
- Approximate delays are shown in source seconds and slowed-replay seconds;
    replay delay equals source delay times X, not necessarily real-world delay.

Outputs are `report.md` and `per_video.csv`, with per-clip counts, metrics,
slowdown, timing units and status. The Markdown report lists missed intervals
and alert timestamps for manual inspection. Missing/failed runs and unreviewed
clips remain visible but are excluded from scored-subset metrics; incomplete
coverage is explicit. Exit status is 0 for complete coverage, 1 for incomplete
coverage, and 2 for invalid input. Invalid labels or mismatched speeds are errors.

Time-only evaluation currently assumes at most one fall at a time. Simultaneous
falls will need person labels and an association to detector tracks. It also
does not measure how long an incorrect fallen state persists after an alert.
Results measure the whole pipeline, including calibration and tracking, not
just the fall classifier. Add reviewed no-fall clips to assess nuisance alerts.

Run the focused tests without Docker:

```sh
python3 -m unittest discover -s benchmark_v2/tests
```

## Compare With The YOLO11 Fall Model

The standalone baseline runs `yolo11_code/best.pt` on the original videos and
reuses this evaluator, with confidence 0.4 and a 0.25-source-second fall-free gap.
It compares against the saved 24-video quarter-speed SceneScape experiment,
reporting individual coverage and paired metrics on common scored clips.
See [the runner instructions](yolo11_code/README.md).
