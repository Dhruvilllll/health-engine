"""
Deploy the current Health Engine app and model to a Hugging Face Space.

This script stages a minimal Space payload locally and uploads it to the target
Space repo using the Hugging Face Hub API.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, get_token


DEFAULT_SPACE_ID = "dhruvilmalvania/health-engine"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGING_DIR = PROJECT_ROOT / ".hf_space_build"
TEMPLATE_DIR = PROJECT_ROOT / "deployment" / "hf_space"

COPY_PATHS = [
    "app.py",
    "config",
    "src",
    "app",
]

DATA_PATHS = [
    "data/processed/physiological_cleaned.csv",
    "data/processed/physiological_merged.csv",
    "data/raw/physiological/physiological_cycles.csv",
]

MODEL_PATHS = [
    "models/trained/recovery_random_forest.pkl",
]

DELETE_PATTERNS = [
    "README.md",
    "requirements.txt",
    "app.py",
    "config/*",
    "src/*",
    "app/*",
    "data/*",
    "models/*",
]

IGNORED_NAMES = shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy Health Engine to a Hugging Face Space.")
    parser.add_argument("--repo-id", default=DEFAULT_SPACE_ID, help="Target Hugging Face Space id.")
    parser.add_argument(
        "--commit-message",
        default="Update Health Engine dashboard and model",
        help="Commit message for the Space update.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. Defaults to HF_TOKEN or local login cache.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the staging folder without uploading it.",
    )
    return parser.parse_args()


def reset_staging_dir() -> None:
    if STAGING_DIR.exists():
        shutil.rmtree(STAGING_DIR)
    STAGING_DIR.mkdir(parents=True, exist_ok=True)


def copy_path(relative_path: str) -> None:
    source = PROJECT_ROOT / relative_path
    destination = STAGING_DIR / relative_path

    if not source.exists():
        raise FileNotFoundError(f"Required path is missing: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True, ignore=IGNORED_NAMES)
    else:
        shutil.copy2(source, destination)


def write_space_templates() -> None:
    shutil.copy2(TEMPLATE_DIR / "README.md", STAGING_DIR / "README.md")
    shutil.copy2(TEMPLATE_DIR / "requirements.txt", STAGING_DIR / "requirements.txt")


def build_staging_bundle() -> None:
    reset_staging_dir()
    write_space_templates()

    for relative_path in COPY_PATHS + DATA_PATHS + MODEL_PATHS:
        copy_path(relative_path)


def resolve_token(explicit_token: str | None) -> str | None:
    if explicit_token:
        return explicit_token

    env_token = os.getenv("HF_TOKEN")
    if env_token:
        return env_token

    return get_token()


def staged_file_manifest() -> list[str]:
    return sorted(
        str(path.relative_to(STAGING_DIR)).replace("\\", "/")
        for path in STAGING_DIR.rglob("*")
        if path.is_file()
    )


def upload_bundle(repo_id: str, token: str, commit_message: str) -> None:
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="space",
        folder_path=STAGING_DIR,
        commit_message=commit_message,
        delete_patterns=DELETE_PATTERNS,
    )


def main() -> None:
    args = parse_args()
    build_staging_bundle()

    manifest = staged_file_manifest()
    print("Prepared Space bundle:")
    for file_path in manifest:
        print(f" - {file_path}")

    if args.dry_run:
        print("\nDry run complete. No files were uploaded.")
        return

    token = resolve_token(args.token)
    if not token:
        raise SystemExit(
            "No Hugging Face token found. Set HF_TOKEN or run `huggingface-cli login`, then rerun."
        )

    upload_bundle(args.repo_id, token, args.commit_message)
    print(f"\nSpace updated: https://huggingface.co/spaces/{args.repo_id}")


if __name__ == "__main__":
    main()
