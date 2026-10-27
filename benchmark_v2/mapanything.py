#!/usr/bin/env python3

"""Generate aligned MapAnything scenes and cameras from each chute's first frame."""

import argparse
import base64
import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path

import requests
import trimesh

from alignment import align_geometry


ROOT = Path(__file__).resolve().parent


def extract_first_frame(video_path: Path, frame_path: Path) -> None:
  subprocess.run(
      [
          "ffmpeg", "-v", "error", "-i", str(video_path),
          "-frames:v", "1", "-y", str(frame_path),
      ],
      check=True,
  )


def generate_scene(session: requests.Session, api_url: str, video_path: Path,
                   timeout: int, camera_id: str) -> None:

  with tempfile.TemporaryDirectory() as temp_dir:
    frame_path = Path(temp_dir) / "frame.jpg"
    extract_first_frame(video_path, frame_path)

    with frame_path.open("rb") as frame:
      response = session.post(
          f"{api_url}/reconstruction",
          files={"images": (frame_path.name, frame, "image/jpeg")},
          data={"output_format": "glb", "mesh_type": "mesh", "camera_ids": camera_id},
          timeout=60,
      )
    response.raise_for_status()
    request_id = response.json()["request_id"]

  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    response = session.get(
        f"{api_url}/reconstruction/status/{request_id}", timeout=30)
    response.raise_for_status()
    status = response.json()

    if status["state"] == "complete":
      result = status["result"]
      glb_data = base64.b64decode(result.pop("glb_data"), validate=True)
      if not glb_data.startswith(b"glTF"):
        raise ValueError("Mapping service returned an invalid GLB file")
      (video_path.parent / "scene.raw.glb").write_bytes(glb_data)
      (video_path.parent / "mapping.json").write_text(
          json.dumps(result, indent=2, allow_nan=False) + "\n")
      return
    if status["state"] == "failed":
      raise RuntimeError(status.get("error", "Mapping reconstruction failed"))

    time.sleep(2)

  raise TimeoutError(f"Reconstruction timed out after {timeout} seconds")


def align_scene(video_path: Path, camera_id: str, meters_per_unit: float,
                floor_plane) -> None:
  directory = video_path.parent
  result = json.loads((directory / "mapping.json").read_text())
  poses = [pose for pose in result["camera_poses"] if pose["camera_id"] == camera_id]
  intrinsics = [item for item in result["intrinsics"] if item["camera_id"] == camera_id]
  if len(poses) != 1 or len(intrinsics) != 1:
    raise ValueError(f"Expected one pose and intrinsics for {camera_id}")
  probe = subprocess.run(
      ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
       "stream=width,height", "-of", "json", str(video_path)],
      check=True, capture_output=True, text=True)
  stream = json.loads(probe.stdout)["streams"][0]
  scene = trimesh.load_scene(directory / "scene.raw.glb", process=False)
  mesh, camera, alignment = align_geometry(
      scene, poses[0], intrinsics[0]["K"], stream["width"], stream["height"],
      meters_per_unit, floor_plane)
  camera.update(uid=camera_id, name=camera_id, transform_type="quaternion")
  mesh.export(directory / "scene.glb", file_type="glb")
  (directory / "camera.json").write_text(json.dumps(camera, indent=2, allow_nan=False) + "\n")
  (directory / "alignment.json").write_text(json.dumps(alignment, indent=2, allow_nan=False) + "\n")


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Generate aligned scene.glb and camera.json for each chute video")
  parser.add_argument(
      "--url", default="https://localhost:8444/v1",
      help="Mapping service API URL (default: %(default)s)")
  parser.add_argument(
      "--data-dir", type=Path, default=ROOT / "data",
      help="Directory containing chute folders (default: %(default)s)")
  parser.add_argument(
      "--timeout", type=int, default=900,
      help="Timeout per reconstruction in seconds (default: %(default)s)")
  parser.add_argument("--realign", action="store_true",
                      help="Reuse scene.raw.glb and mapping.json without calling MapAnything")
  parser.add_argument("--meters-per-unit", type=float, default=1.0)
  parser.add_argument("--floor-plane", nargs=4, type=float, metavar=("NX", "NY", "NZ", "D"))
  args = parser.parse_args()

  api_url = args.url.rstrip("/")
  session = requests.Session()
  session.verify = False

  if not args.realign:
    health_response = session.get(f"{api_url}/health", timeout=30)
    health_response.raise_for_status()
    health = health_response.json()
    if health.get("model") != "mapanything" or health.get("device") != "cpu":
      raise RuntimeError("Mapping service must run MapAnything in CPU mode")

  videos = sorted(args.data_dir.glob("chute0*/cam2.avi"))
  if not videos:
    raise FileNotFoundError(f"No chute0*/cam2.avi files found in {args.data_dir}")

  for video_path in videos:
    camera_id = "bench-" + hashlib.sha256(
        f"{video_path.parent.name}/{video_path.stem}".encode()).hexdigest()[:12]
    if not args.realign:
      print(f"Generating scene for {video_path.parent.name}...", flush=True)
      generate_scene(session, api_url, video_path, args.timeout, camera_id)
    align_scene(video_path, camera_id, args.meters_per_unit, args.floor_plane)
    print(f"Saved aligned scene.glb and camera.json in {video_path.parent}", flush=True)


if __name__ == "__main__":
  main()
