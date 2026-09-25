#!/usr/bin/env python3

import getpass
import json
import os
import shutil
import subprocess
import sys

import requests

SCENESCAPE_CERT_HOSTNAME = "web.scenescape.intel.com"

class SceneScapeHTTPSAdapter(requests.adapters.HTTPAdapter):
    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        pool_kwargs["assert_hostname"] = SCENESCAPE_CERT_HOSTNAME
        pool_kwargs["server_hostname"] = SCENESCAPE_CERT_HOSTNAME
        return super().init_poolmanager(
            connections, maxsize, block=block, **pool_kwargs)

session = requests.Session()
session.mount("https://localhost", SceneScapeHTTPSAdapter())


def prompt_for_path(label, default_path):
    value = input(f"Enter the path to your {label} [{default_path}]: ").strip()
    path = os.path.abspath(value or default_path)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Directory does not exist: {path}")
    return path


def read_env(env_path):
    values = {}
    if not os.path.isfile(env_path):
        return values
    with open(env_path, "r") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value
    return values


def compose_command(scenescape_path, override_path, env_path, *args):
    base_compose = os.path.join(
        scenescape_path, "sample_data", "compose",
        "docker-compose-dl-streamer-example.yml")
    return [
        "docker", "compose",
        "--project-directory", scenescape_path,
        "--env-file", env_path,
        "-f", base_compose,
        "-f", override_path,
        "--profile", "controller",
        *args,
    ]


def delete_app_cameras(app_path, api_key, ca_cert):
    with open(os.path.join(app_path, "dataset", "cameras.json"), "r") as camera_file:
        camera_data = json.load(camera_file)
    configured = camera_data.get("cameras", camera_data)
    configured_ids = {camera.get("uid") for camera in configured}

    api_url = "https://localhost:443/api/v1"
    headers = {"Authorization": f"Token {api_key}"}
    response = session.get(
        f"{api_url}/cameras", headers=headers, verify=ca_cert, timeout=10)
    response.raise_for_status()
    response_data = response.json()
    cameras = response_data.get("results", response_data) \
        if isinstance(response_data, dict) else response_data
    for camera in cameras:
        camera_id = camera.get("uid") or camera.get("id")
        if camera_id not in configured_ids:
            continue
        response = session.delete(
            f"{api_url}/camera/{camera_id}", headers=headers,
            verify=ca_cert, timeout=10)
        response.raise_for_status()
        print(f"Deleted camera: {camera_id}")


def main():
    app_path = prompt_for_path("fall detection app", os.getcwd())
    scenescape_path = prompt_for_path(
        "SceneScape install", os.path.expanduser("~/scenescape"))
    generated_dir = os.path.join(app_path, ".generated")
    env_path = os.path.join(generated_dir, "fall-detection.env")
    override_path = os.path.join(generated_dir, "docker-compose.override.yml")

    if os.path.isfile(env_path) and os.path.isfile(override_path):
        subprocess.run(
            compose_command(
                scenescape_path, override_path, env_path,
                "rm", "--stop", "--force",
                "falling-cams", "falling-video", "fall-detection", "node-red"),
            check=True,
        )
    else:
        print("Generated deployment files not found; no app containers were removed.")

    env_values = read_env(env_path)
    api_key = os.environ.get("SCENESCAPE_API_KEY") \
        or env_values.get("SCENESCAPE_API_KEY")
    remove_cameras = input("Delete the configured fall-detection cameras? [y/N]: ").strip().lower()
    if remove_cameras in ("y", "yes"):
        api_key = api_key or getpass.getpass("SceneScape API key: ")
        ca_cert = os.path.join(
            scenescape_path, "manager", "secrets", "certs", "scenescape-ca.pem")
        delete_app_cameras(app_path, api_key, ca_cert)

    remove_node_red = input("Delete app-owned Node-RED data? [y/N]: ").strip().lower()
    if remove_node_red in ("y", "yes"):
        node_red_path = os.path.join(app_path, "node_red_data")
        if os.path.isdir(node_red_path):
            shutil.rmtree(node_red_path)

    if os.path.isdir(generated_dir):
        shutil.rmtree(generated_dir)
    print("Uninstall complete. SceneScape files and shared volumes were left unchanged.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, requests.RequestException, subprocess.CalledProcessError) as error:
        print(f"Uninstall failed: {error}", file=sys.stderr)
        sys.exit(1)
