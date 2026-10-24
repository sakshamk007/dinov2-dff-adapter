#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from pathlib import Path


def download_file(url, output_dir, filename=None):
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use a user-specified filename or let gdown determine it.
    output_path = output_dir / filename if filename else None

    command = [
        sys.executable,
        "-m",
        "gdown",
        "--continue",
        url,
    ]

    if output_path:
        command.extend(["-O", str(output_path)])
    else:
        command.extend(["--output", str(output_dir) + os.sep])

    print(f"Google Drive URL: {url}")
    print(f"Download directory: {output_dir}")
    print("Starting download...")
    print("Keep the SSH session alive, or use tmux/screen.")
    print("-" * 60)

    try:
        subprocess.run(command, check=True)

        print("-" * 60)
        print("Download completed successfully.")

        if output_path and output_path.exists():
            print(f"Saved file: {output_path}")
            print(f"File size: {output_path.stat().st_size / (1024**3):.2f} GB")
        else:
            print(f"Files downloaded to: {output_dir}")

    except subprocess.CalledProcessError as exc:
        print(f"Download failed with exit code {exc.returncode}.")
        print("Check the Google Drive link, permissions, and available disk space.")
        sys.exit(exc.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="Download a file from Google Drive."
    )

    parser.add_argument(
        "url",
        help="Google Drive file or supported download URL"
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        help="Directory where the file will be saved (default: current directory)"
    )

    parser.add_argument(
        "-f",
        "--filename",
        default=None,
        help="Optional output filename, e.g. dataset.zip"
    )

    args = parser.parse_args()

    download_file(args.url, args.output_dir, args.filename)


if __name__ == "__main__":
    main()
