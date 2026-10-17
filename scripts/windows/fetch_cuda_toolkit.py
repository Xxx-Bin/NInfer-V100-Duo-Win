"""Assemble a portable CUDA 12.8 toolkit from NVIDIA's redistributable archives.

The project targets sm_70 (V100), and CUDA 13 removed Volta offline compilation, so 12.8 is the
last usable toolkit. A normal toolkit install needs an elevated installer; this script instead
downloads the four components the build actually consumes (nvcc, cudart, CCCL headers and NVTX)
and merges them into ``_toolchain/cuda-12.8`` in the repository, which ``scripts/windows/
configure.cmd`` then passes to CMake as ``CUDAToolkit_ROOT``.

Usage:
    python scripts/windows/fetch_cuda_toolkit.py [--force]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
import zipfile

BASE = "https://developer.download.nvidia.com/compute/cuda/redist"
COMPONENTS = [
    "cuda_nvcc/windows-x86_64/cuda_nvcc-windows-x86_64-12.8.61-archive.zip",
    "cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-12.8.57-archive.zip",
    "cuda_cccl/windows-x86_64/cuda_cccl-windows-x86_64-12.8.55-archive.zip",
    "cuda_nvtx/windows-x86_64/cuda_nvtx-windows-x86_64-12.8.55-archive.zip",
]
# Only these top-level directories are merged from the secondary packages.
MERGED_DIRS = {
    "cuda_cudart": ("include", "lib"),
    "cuda_cccl": ("include",),
    "cuda_nvtx": ("include",),
}
REQUIRED = [
    "bin/nvcc.exe",
    "nvvm/libdevice/libdevice.10.bc",
    "include/cuda_runtime.h",
    "include/cuda.h",
    "lib/x64/cudart.lib",
    "lib/x64/cudart_static.lib",
]


def download(url: str, destination: str) -> None:
    if os.path.exists(destination):
        print(f"have {os.path.basename(destination)}")
        return
    partial = destination + ".part"
    print(f"fetch {url}")
    with urllib.request.urlopen(url, timeout=120) as response, open(partial, "wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / 1e6:.1f}/{total / 1e6:.1f} MB", end="", flush=True)
    print()
    os.replace(partial, destination)


def merge(archive: str, root: str, subdirs: tuple[str, ...] | None) -> None:
    with zipfile.ZipFile(archive) as bundle:
        prefix = bundle.namelist()[0].split("/")[0]
        for entry in bundle.namelist():
            relative = entry[len(prefix) + 1 :]
            if not relative:
                continue
            if subdirs is not None and relative.split("/")[0] not in subdirs:
                continue
            target = os.path.join(root, relative)
            if entry.endswith("/"):
                os.makedirs(target, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with bundle.open(entry) as source, open(target, "wb") as destination:
                shutil.copyfileobj(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="re-extract over an existing tree")
    args = parser.parse_args()

    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    cache = os.path.join(repo, "_toolchain", "downloads")
    root = os.path.join(repo, "_toolchain", "cuda-12.8")
    os.makedirs(cache, exist_ok=True)
    if args.force:
        shutil.rmtree(root, ignore_errors=True)
    os.makedirs(root, exist_ok=True)

    for component in COMPONENTS:
        name = os.path.basename(component)
        archive = os.path.join(cache, name)
        download(f"{BASE}/{component}", archive)
        package = name.split("-windows")[0]
        merge(archive, root, MERGED_DIRS.get(package))

    missing = [path for path in REQUIRED if not os.path.exists(os.path.join(root, path))]
    if missing:
        print("toolkit incomplete, missing:", ", ".join(missing), file=sys.stderr)
        return 1
    print(f"CUDA 12.8 toolkit ready at {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
