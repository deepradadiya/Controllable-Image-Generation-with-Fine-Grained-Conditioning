"""
Set Up a HuggingFace Space for ControlNet Demo Deployment

This script automates the creation and configuration of a HuggingFace Space
for hosting the ControlNet image generation demo. It handles:
- Creating the Space repository on HuggingFace
- Copying necessary source files from the project
- Pushing the initial deployment

Usage:
    python scripts/setup_space.py \
        --space-id username/controlnet-demo \
        --hardware gpu-basic

Requirements Addressed:
- 8.1: HuggingFace Space compatible web interface
- 10.5: Load pre-trained models from HuggingFace Hub
"""

import argparse
import logging
import shutil
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Project root relative to this script
PROJECT_ROOT = Path(__file__).parent.parent

# Files and directories to include in the Space deployment
SPACE_FILES = [
    # Entry point
    "deployment/app.py",
    "deployment/requirements.txt",
    "deployment/README.md",
    # Source code needed for inference
    "src/__init__.py",
    "src/app/__init__.py",
    "src/app/gradio_app.py",
    "src/app/controls.py",
    "src/app/model_manager.py",
    "src/inference/__init__.py",
    "src/inference/pipeline.py",
    "src/inference/controls.py",
    "src/inference/model_loader.py",
    "src/models/__init__.py",
    "src/models/controlnet.py",
    "src/models/config.py",
    "src/models/unet_wrapper.py",
    "src/data/__init__.py",
    "src/data/extract_depth.py",
    "src/data/extract_pose.py",
    "src/data/extract_edges.py",
    "src/utils/__init__.py",
    "src/utils/memory_utils.py",
    "src/deployment/__init__.py",
    "src/deployment/health_check.py",
    # Configuration
    "configs/__init__.py",
    "configs/base_config.py",
]


def collect_space_files(project_root: Path) -> List[Path]:
    """
    Collect all files needed for the Space deployment.

    Args:
        project_root: Root directory of the project.

    Returns:
        List of Path objects for files that exist and should be deployed.
    """
    existing_files = []
    missing_files = []

    for file_path in SPACE_FILES:
        full_path = project_root / file_path
        if full_path.exists():
            existing_files.append(file_path)
        else:
            missing_files.append(file_path)

    if missing_files:
        logger.warning(
            f"The following files are missing and will be skipped: "
            f"{', '.join(missing_files)}"
        )

    return existing_files


def create_space(
    space_id: str,
    hardware: str = "gpu-basic",
    private: bool = False,
    token: Optional[str] = None,
) -> str:
    """
    Create a new HuggingFace Space repository.

    Args:
        space_id: Space identifier (e.g., 'username/space-name').
        hardware: Hardware tier for the Space (cpu-basic, gpu-basic, etc.).
        private: Whether to create a private Space.
        token: HuggingFace API token.

    Returns:
        URL of the created Space.
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        logger.error(
            "huggingface_hub is required. Install with: pip install huggingface-hub"
        )
        sys.exit(1)

    api = HfApi(token=token)

    try:
        repo_url = create_repo(
            repo_id=space_id,
            repo_type="space",
            space_sdk="gradio",
            space_hardware=hardware,
            private=private,
            exist_ok=True,
            token=token,
        )
        logger.info(f"Space created/verified: {repo_url}")
        return str(repo_url)
    except Exception as e:
        logger.error(f"Failed to create Space: {e}")
        raise


def prepare_deployment_directory(
    project_root: Path,
    files_to_deploy: List[str],
) -> Path:
    """
    Prepare a temporary directory with the Space deployment structure.

    The deployment directory is structured so that:
    - deployment/app.py becomes the root app.py
    - deployment/requirements.txt becomes the root requirements.txt
    - deployment/README.md becomes the root README.md
    - All other files maintain their relative paths

    Args:
        project_root: Root directory of the project.
        files_to_deploy: List of relative file paths to include.

    Returns:
        Path to the temporary deployment directory.
    """
    deploy_dir = Path(tempfile.mkdtemp(prefix="controlnet_space_"))
    logger.info(f"Preparing deployment in: {deploy_dir}")

    for file_path in files_to_deploy:
        src = project_root / file_path
        if not src.exists():
            continue

        # Remap deployment/ files to root of the Space
        if file_path.startswith("deployment/"):
            relative_name = file_path.replace("deployment/", "", 1)
            dst = deploy_dir / relative_name
        else:
            dst = deploy_dir / file_path

        # Create parent directories
        dst.parent.mkdir(parents=True, exist_ok=True)

        # Copy the file
        shutil.copy2(src, dst)
        logger.debug(f"Copied: {file_path} -> {dst.relative_to(deploy_dir)}")

    logger.info(f"Prepared {len(files_to_deploy)} files for deployment")
    return deploy_dir


def push_to_space(
    deploy_dir: Path,
    space_id: str,
    commit_message: str = "Initial Space deployment",
    token: Optional[str] = None,
) -> None:
    """
    Push the prepared deployment directory to the HuggingFace Space.

    Args:
        deploy_dir: Path to the prepared deployment directory.
        space_id: HuggingFace Space identifier.
        commit_message: Commit message for the push.
        token: HuggingFace API token.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.error(
            "huggingface_hub is required. Install with: pip install huggingface-hub"
        )
        sys.exit(1)

    api = HfApi(token=token)

    try:
        api.upload_folder(
            folder_path=str(deploy_dir),
            repo_id=space_id,
            repo_type="space",
            commit_message=commit_message,
            token=token,
        )
        logger.info(f"Successfully pushed to Space: https://huggingface.co/spaces/{space_id}")
    except Exception as e:
        logger.error(f"Failed to push to Space: {e}")
        raise


def setup_space(
    space_id: str,
    hardware: str = "gpu-basic",
    private: bool = False,
    token: Optional[str] = None,
    commit_message: str = "Initial ControlNet Space deployment",
) -> str:
    """
    Complete Space setup: create repo, prepare files, and push deployment.

    Args:
        space_id: HuggingFace Space identifier.
        hardware: Hardware tier for the Space.
        private: Whether to create a private Space.
        token: HuggingFace API token.
        commit_message: Commit message for the initial push.

    Returns:
        URL of the deployed Space.
    """
    logger.info(f"Setting up HuggingFace Space: {space_id}")

    # Step 1: Create the Space repository
    space_url = create_space(
        space_id=space_id,
        hardware=hardware,
        private=private,
        token=token,
    )

    # Step 2: Collect and prepare deployment files
    files_to_deploy = collect_space_files(PROJECT_ROOT)
    if not files_to_deploy:
        logger.error("No files found to deploy. Check project structure.")
        sys.exit(1)

    deploy_dir = prepare_deployment_directory(PROJECT_ROOT, files_to_deploy)

    # Step 3: Push to the Space
    try:
        push_to_space(
            deploy_dir=deploy_dir,
            space_id=space_id,
            commit_message=commit_message,
            token=token,
        )
    finally:
        # Clean up temporary directory
        shutil.rmtree(deploy_dir, ignore_errors=True)
        logger.debug("Cleaned up temporary deployment directory")

    return f"https://huggingface.co/spaces/{space_id}"


def main():
    """Parse arguments and execute Space setup."""
    parser = argparse.ArgumentParser(
        description="Set up a HuggingFace Space for ControlNet demo deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create a public Space with GPU
    python scripts/setup_space.py \\
        --space-id myuser/controlnet-demo \\
        --hardware gpu-basic

    # Create a private Space with CPU (for testing)
    python scripts/setup_space.py \\
        --space-id myuser/controlnet-test \\
        --hardware cpu-basic \\
        --private

    # Use a specific token
    python scripts/setup_space.py \\
        --space-id myuser/controlnet-demo \\
        --token hf_xxxxxxxxxxxxx
        """,
    )

    parser.add_argument(
        "--space-id",
        type=str,
        required=True,
        help="HuggingFace Space ID (e.g., 'username/space-name')",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        choices=["cpu-basic", "cpu-upgrade", "gpu-basic", "gpu-upgrade"],
        default="gpu-basic",
        help="Hardware tier for the Space (default: gpu-basic)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private Space (default: public)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (uses cached token if not provided)",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Initial ControlNet Space deployment",
        help="Commit message for the initial push",
    )

    args = parser.parse_args()

    try:
        url = setup_space(
            space_id=args.space_id,
            hardware=args.hardware,
            private=args.private,
            token=args.token,
            commit_message=args.commit_message,
        )
        logger.info(f"Space setup complete: {url}")
    except Exception as e:
        logger.error(f"Space setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
