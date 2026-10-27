#!/usr/bin/env python3

import argparse
import json
import os
import subprocess
from pathlib import Path


DEFAULT_IMAGE = "docker.io/intel/dlstreamer-pipeline-server:2026.3.0-20260915-weekly-ubuntu24"
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / "models/yolo11s-pose/FP16/yolo11s-pose.xml"
VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".m4v", ".ts", ".webm"}


def main():
  parser = argparse.ArgumentParser(description="Run DL Streamer person detection on every video")
  parser.add_argument("--input", type=Path, default=Path(__file__).parent / "data")
  parser.add_argument("--output", type=Path, default=Path(__file__).parent / "detections-yolo11s-pose")
  parser.add_argument("--image", default=DEFAULT_IMAGE)
  parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
  args = parser.parse_args()

  input_dir = args.input.resolve()
  output_dir = args.output.resolve()
  model = args.model.resolve(strict=True)
  videos = sorted(
      path for path in input_dir.rglob("*")
      if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
  )
  if not videos:
    raise SystemExit(f"No videos found in {input_dir}")

  for video in videos:
    relative = video.relative_to(input_dir)
    output = output_dir / relative.with_suffix(".json")
    output.parent.mkdir(parents=True, exist_ok=True)
    container_input = Path("/data") / relative
    container_output = Path("/output") / output.relative_to(output_dir)

    pipeline = [
        "docker", "run", "--rm",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--entrypoint", "gst-launch-1.0",
        "-v", f"{input_dir}:/data:ro",
        "-v", f"{output_dir}:/output",
        "-v", f"{model.parent}:/model:ro",
        args.image, "-q",
        "filesrc", f"location={container_input}", "!",
        "decodebin", "!", "videoconvert", "!", "video/x-raw,format=BGR", "!",
        "gvadetect", f"model=/model/{model.name}", "device=CPU",
        "pre-process-backend=opencv", "!",
        "gvametaconvert", "add-tensor-data=true", "!",
        "gvametapublish", "method=file", "file-format=json",
        f"file-path={container_output}", "!", "fakesink", "sync=false",
    ]
    print(f"Detecting {relative} -> {output.relative_to(output_dir)}", flush=True)
    subprocess.run(pipeline, check=True)
    records = json.loads(output.read_text())
    for record in records:
      record["objects"] = [
          obj for obj in record.get("objects", [])
          if obj.get("detection", {}).get("label") == "person"
      ]
    output.write_text(json.dumps(records))

  print(f"Wrote {len(videos)} detection files to {output_dir}")


if __name__ == "__main__":
  main()