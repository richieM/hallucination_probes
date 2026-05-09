"""Boot/terminate RunPod pods for auth_projection experiments. Uses runpod SDK.

Subcommands:
    uv run python -m auth_projection.runpod_helper boot        --gpu_id "NVIDIA A40" --name v6-replay
    uv run python -m auth_projection.runpod_helper wait        --pod_id <id>
    uv run python -m auth_projection.runpod_helper ssh         --pod_id <id>
    uv run python -m auth_projection.runpod_helper kill        --pod_id <id>

`boot` blocks until the pod has a public SSH endpoint, then prints SSH host:port + pod_id
on stdout in a parseable form:

    POD_ID=<id>
    SSH_HOST=<ip>
    SSH_PORT=<port>

so the caller can `eval $(... boot ...)`.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import runpod
from dotenv import load_dotenv

load_dotenv()
assert os.environ.get("RUNPOD_API_KEY"), "RUNPOD_API_KEY is not set"
runpod.api_key = os.environ["RUNPOD_API_KEY"]

DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


def _public_tcp(pod: dict) -> tuple[str, int] | None:
    rt = pod.get("runtime") or {}
    for p in (rt.get("ports") or []):
        if p.get("privatePort") == 22 and p.get("type") == "tcp" and p.get("isIpPublic"):
            return p["ip"], int(p["publicPort"])
    return None


def boot(name: str, gpu_id: str, container_disk_gb: int, volume_gb: int,
         min_memory_gb: int, image: str, env: dict | None) -> dict:
    pod = runpod.create_pod(
        name=name,
        image_name=image,
        gpu_type_id=gpu_id,
        cloud_type="ALL",
        support_public_ip=True,
        start_ssh=True,
        gpu_count=1,
        volume_in_gb=volume_gb,
        container_disk_in_gb=container_disk_gb,
        min_vcpu_count=8,
        min_memory_in_gb=min_memory_gb,
        ports="22/tcp",
        volume_mount_path="/workspace",
        env=env or {},
    )
    return pod


def wait_for_ssh(pod_id: str, timeout_s: int = 600) -> tuple[str, int]:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        pod = runpod.get_pod(pod_id)
        endp = _public_tcp(pod)
        if endp:
            return endp
        time.sleep(5)
    raise TimeoutError(f"Pod {pod_id} did not get a public SSH endpoint within {timeout_s}s")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("boot")
    b.add_argument("--name", default="v6-replay")
    b.add_argument("--gpu_id", default="NVIDIA A40")
    b.add_argument("--container_disk_gb", type=int, default=80)
    b.add_argument("--volume_gb", type=int, default=0)
    b.add_argument("--min_memory_gb", type=int, default=32)
    b.add_argument("--image", default=DEFAULT_IMAGE)
    b.add_argument("--ssh_timeout_s", type=int, default=600)

    w = sub.add_parser("wait")
    w.add_argument("--pod_id", required=True)
    w.add_argument("--timeout_s", type=int, default=600)

    s = sub.add_parser("ssh")
    s.add_argument("--pod_id", required=True)

    k = sub.add_parser("kill")
    k.add_argument("--pod_id", required=True)

    args = ap.parse_args()

    if args.cmd == "boot":
        env = {}
        if hf := os.environ.get("HF_TOKEN"):
            env["HF_TOKEN"] = hf
        if a := os.environ.get("ANTHROPIC_API_KEY"):
            env["ANTHROPIC_API_KEY"] = a
        pod = boot(
            name=args.name, gpu_id=args.gpu_id,
            container_disk_gb=args.container_disk_gb, volume_gb=args.volume_gb,
            min_memory_gb=args.min_memory_gb, image=args.image, env=env,
        )
        pod_id = pod["id"]
        print(f"[runpod_helper] booted {pod_id}; waiting for SSH...", file=sys.stderr)
        ip, port = wait_for_ssh(pod_id, timeout_s=args.ssh_timeout_s)
        print(f"POD_ID={pod_id}")
        print(f"SSH_HOST={ip}")
        print(f"SSH_PORT={port}")
    elif args.cmd == "wait":
        ip, port = wait_for_ssh(args.pod_id, timeout_s=args.timeout_s)
        print(f"SSH_HOST={ip}")
        print(f"SSH_PORT={port}")
    elif args.cmd == "ssh":
        pod = runpod.get_pod(args.pod_id)
        endp = _public_tcp(pod)
        if not endp:
            print(f"no public SSH yet for {args.pod_id}", file=sys.stderr)
            sys.exit(1)
        ip, port = endp
        print(f"ssh -p {port} root@{ip}")
    elif args.cmd == "kill":
        runpod.terminate_pod(args.pod_id)
        print(f"terminated {args.pod_id}")


if __name__ == "__main__":
    main()
