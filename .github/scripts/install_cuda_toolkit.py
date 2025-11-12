#!/usr/bin/env python3
# pyre-strict

"""
Download and extract CUDA components.

Automatically fetches component metadata from redistrib JSON files
based on the specified CUDA version and platform.
"""

import argparse
import json
import logging
import platform
import shutil
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Final


logger = logging.getLogger(__name__)

DEFAULT_BASE_URL: Final[str] = (
    "https://developer.download.nvidia.com/compute/cuda/redist"
)
DEFAULT_CUDA_VERSION: Final[str] = "13.0.2"
COMPONENTS_TO_DOWNLOAD: Final[list[str]] = ["cuda_cudart", "cuda_nvcc", "cuda_crt", "libnvvm", "cuda_cccl"]


def detect_platform() -> str:
    """Detect the current platform and return the CUDA platform identifier."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "linux":
        if machine in ("x86_64", "amd64"):
            return "linux-x86_64"
        elif machine in ("ppc64le", "ppc64"):
            return "linux-ppc64le"
        elif machine in ("aarch64", "arm64"):
            return "linux-sbsa"
    elif system == "windows":
        if machine in ("x86_64", "amd64"):
            return "windows-x86_64"

    raise RuntimeError(f"Unsupported platform: {system} {machine}")


def fetch_redistrib_metadata(base_url: str, cuda_version: str) -> dict[str, Any]:
    """Fetch the redistrib metadata JSON for the specified CUDA version."""
    metadata_url = f"{base_url}/redistrib_{cuda_version}.json"
    logger.info("Fetching metadata from %s", metadata_url)

    try:
        with urllib.request.urlopen(metadata_url, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"Failed to fetch metadata from {metadata_url}: HTTP {e.code} {e.reason}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to connect to {metadata_url}: {e.reason}") from e


def get_component_path(
    metadata: dict[str, Any],
    component_name: str,
    platform_name: str,
) -> str:
    """Extract the relative path for a component from the metadata."""
    if component_name not in metadata:
        raise ValueError(f"Component {component_name} not found in metadata")

    component = metadata[component_name]
    if platform_name not in component:
        available_platforms = [
            key
            for key in component.keys()
            if not key.startswith(("name", "license", "version"))
        ]
        raise ValueError(
            f"Platform {platform_name} not found for component {component_name}. "
            f"Available platforms: {', '.join(available_platforms)}"
        )

    return component[platform_name]["relative_path"]


def download_file(url: str, destination: Path) -> None:
    """Download a file from a URL to a local path."""
    logger.info("Downloading %s ...", url)

    try:
        with urllib.request.urlopen(url, timeout=300) as response:
            with open(destination, "wb") as f:
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)
    except urllib.error.HTTPError as e:
        raise RuntimeError(
            f"Failed to download from {url}: HTTP {e.code} {e.reason}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Failed to connect to {url}: {e.reason}") from e


def extract_archive(archive_path: Path, extract_to: Path) -> None:
    """Extract an archive (zip or tar.xz) to a directory."""
    logger.info("Extracting %s to %s ...", archive_path, extract_to)
    kwargs = {}

    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(extract_to, **kwargs)
    elif archive_path.name.endswith(".tar.xz"):
        with tarfile.open(archive_path, "r:xz") as tar_ref:
            tar_ref.extractall(extract_to, **kwargs)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")


def copy_directory_contents(src: Path, dst: Path) -> None:
    """Copy all contents from src directory to dst directory."""
    logger.info("Copying contents from %s to %s ...", src, dst)
    for item in src.iterdir():
        dst_item = dst / item.name
        if item.is_dir():
            shutil.copytree(item, dst_item, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst_item)


def process_component_from_path(
    relative_path: str,
    base_url: str,
    base_dir: Path,
    cleanup: bool,
) -> None:
    """Download, extract, and install a CUDA component from its relative path."""
    archive_name = relative_path.split("/")[-1]
    archive_path = Path(archive_name)

    download_url = f"{base_url}/{relative_path}"
    download_file(download_url, archive_path)

    extract_archive(archive_path, base_dir)

    if cleanup:
        archive_path.unlink()
        logger.info("Removed temporary file %s", archive_path)

    component_dir_name = archive_name.replace(".tar.xz", "").replace(".zip", "")
    component_dir = base_dir / component_dir_name
    if component_dir.exists():
        copy_directory_contents(component_dir, base_dir)
    else:
        logger.warning(
            "Component directory %s not found after extraction", component_dir
        )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download and install CUDA components. "
        "This script automatically fetches component metadata from NVIDIA's redistrib "
        "JSON files and downloads the specified CUDA components for your platform.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download CUDA 12.3.2 for current platform
  python download_cuda.py --cuda-version 12.3.2

  # Download to custom directory
  python download_cuda.py -c 12.4.1 -d /opt/cuda

  # Download specific components
  python download_cuda.py -c 12.3.2 -C cuda_cudart -C cuda_nvcc -C libcublas

  # Download for specific platform
  python download_cuda.py -c 12.3.2 -p linux-x86_64
        """,
    )

    parser.add_argument(
        "--cuda-version",
        "-c",
        type=str,
        default=DEFAULT_CUDA_VERSION,
        help="CUDA version to download (e.g., '12.3.2', '12.4.1', '13.0.1'). "
        f"Default: {DEFAULT_CUDA_VERSION}",
    )

    parser.add_argument(
        "--base-dir",
        "-d",
        type=Path,
        help="Base directory where CUDA components will be installed. "
        "Defaults to a version-specific directory.",
    )

    parser.add_argument(
        "--base-url",
        "-u",
        type=str,
        default=DEFAULT_BASE_URL,
        help=f"Base URL for downloading CUDA components. Default: {DEFAULT_BASE_URL}",
    )

    parser.add_argument(
        "--platform",
        "-p",
        type=str,
        help="Target platform (e.g., 'windows-x86_64', 'linux-x86_64'). "
        "Auto-detected if not specified.",
    )

    parser.add_argument(
        "--component",
        "-C",
        dest="components",
        action="append",
        type=str,
        help="Specific component(s) to download. Can be specified multiple times. "
        "Defaults to cuda_cudart and cuda_nvcc.",
    )

    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Keep downloaded archive files after extraction.",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging output.",
    )

    return parser.parse_args()


def main() -> None:
    """Download and install CUDA components."""
    args = parse_arguments()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    cuda_version = args.cuda_version
    base_dir = args.base_dir
    if base_dir is None:
        base_dir = Path(
            f"/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v{cuda_version}"
        )

    platform_name = args.platform if args.platform else detect_platform()
    logger.info("Target platform: %s", platform_name)

    components_to_download = (
        args.components if args.components else COMPONENTS_TO_DOWNLOAD
    )
    logger.info("Components to download: %s", ", ".join(components_to_download))

    metadata = fetch_redistrib_metadata(args.base_url, cuda_version)
    logger.info(
        "Successfully fetched metadata for CUDA %s (release: %s)",
        cuda_version,
        metadata.get("release_label", "unknown"),
    )

    base_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Installing CUDA %s to %s", cuda_version, base_dir)

    cleanup = not args.no_cleanup

    for component_name in components_to_download:
        try:
            relative_path = get_component_path(metadata, component_name, platform_name)
            logger.info("Processing component: %s", component_name)
            process_component_from_path(relative_path, args.base_url, base_dir, cleanup)
        except ValueError as e:
            logger.error("Failed to process component %s: %s", component_name, e)
            raise

    logger.info("Installation complete!")


if __name__ == "__main__":
    main()
