"""
Deploy Trained ControlNet Models to HuggingFace Hub

This script uploads trained ControlNet model weights and configuration
to the HuggingFace Hub, making them available for inference and sharing.

Features:
- Upload model weights (safetensors/pytorch format) and config files
- Create or update a model card with training details
- Support for private and public repositories
- Automatic model card generation with metadata

Usage:
    python scripts/deploy_to_hub.py \
        --model-path models/trained/controlnet-depth \
        --repo-id username/controlnet-depth-sd15 \
        --condition-type depth \
        --private

Requirements Addressed:
- 10.4: Model versioning to track different training runs
- 10.5: Load pre-trained models from HuggingFace Hub
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


MODEL_CARD_TEMPLATE = """---
license: apache-2.0
library_name: diffusers
tags:
  - controlnet
  - stable-diffusion
  - image-generation
  - {condition_type}
pipeline_tag: image-to-image
base_model: runwayml/stable-diffusion-v1-5
---

# ControlNet - {condition_type_title} Conditioning

This is a ControlNet model trained for {condition_type} conditioning with Stable Diffusion 1.5.

## Model Description

This ControlNet adapter enables spatial conditioning of image generation using {condition_type} maps.
The model follows the architecture from "Adding Conditional Control to Text-to-Image Diffusion Models"
(Zhang et al., 2023) with zero convolution initialization for stable training.

## Training Details

- **Base Model**: Stable Diffusion 1.5
- **Condition Type**: {condition_type_title}
- **Training Dataset**: COCO 2017 subset
- **Training Date**: {training_date}
- **Architecture**: ControlNet with multi-resolution feature outputs

## Usage

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

controlnet = ControlNetModel.from_pretrained("{repo_id}")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
)

# Generate with condition map
image = pipe(
    prompt="your prompt here",
    image=condition_map,
    num_inference_steps=30,
).images[0]
```

## Limitations

- Trained on COCO 2017 subset; may not generalize to all domains
- Optimized for 512x512 resolution
- Best results with clear, well-defined condition maps
"""


def create_model_card(
    repo_id: str,
    condition_type: str,
    model_path: Path,
    extra_metadata: Optional[dict] = None,
) -> str:
    """
    Generate a model card for the uploaded ControlNet model.

    Args:
        repo_id: HuggingFace repository ID (e.g., 'username/model-name').
        condition_type: Type of conditioning (depth, pose, edge).
        model_path: Local path to the model directory.
        extra_metadata: Optional additional metadata to include.

    Returns:
        Formatted model card string.
    """
    condition_titles = {
        "depth": "Depth Map",
        "pose": "Pose Skeleton",
        "edge": "Canny Edge",
    }
    condition_type_title = condition_titles.get(condition_type, condition_type.title())

    card = MODEL_CARD_TEMPLATE.format(
        condition_type=condition_type,
        condition_type_title=condition_type_title,
        repo_id=repo_id,
        training_date=datetime.now().strftime("%Y-%m-%d"),
    )

    # Append extra metadata if provided
    if extra_metadata:
        card += "\n## Additional Metadata\n\n"
        card += "```json\n"
        card += json.dumps(extra_metadata, indent=2)
        card += "\n```\n"

    return card


def upload_model_to_hub(
    model_path: str,
    repo_id: str,
    condition_type: str = "depth",
    private: bool = False,
    commit_message: Optional[str] = None,
    token: Optional[str] = None,
) -> str:
    """
    Upload a trained ControlNet model to HuggingFace Hub.

    Args:
        model_path: Local path to the trained model directory.
        repo_id: Target HuggingFace repository ID.
        condition_type: Type of conditioning the model was trained for.
        private: Whether to create a private repository.
        commit_message: Custom commit message for the upload.
        token: HuggingFace API token (uses cached token if not provided).

    Returns:
        URL of the uploaded model repository.

    Raises:
        FileNotFoundError: If model_path does not exist.
        ValueError: If model_path contains no model files.
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        logger.error(
            "huggingface_hub is required. Install with: pip install huggingface-hub"
        )
        sys.exit(1)

    model_dir = Path(model_path)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    # Verify model files exist
    model_files = list(model_dir.glob("*.safetensors")) + list(
        model_dir.glob("*.bin")
    )
    config_files = list(model_dir.glob("*.json"))

    if not model_files:
        raise ValueError(
            f"No model weight files (.safetensors or .bin) found in {model_path}"
        )

    logger.info(f"Found {len(model_files)} model file(s) and {len(config_files)} config file(s)")

    # Create or get the repository
    api = HfApi(token=token)

    try:
        repo_url = create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
            token=token,
        )
        logger.info(f"Repository ready: {repo_url}")
    except Exception as e:
        logger.error(f"Failed to create/access repository: {e}")
        raise

    # Generate and write model card
    model_card_content = create_model_card(
        repo_id=repo_id,
        condition_type=condition_type,
        model_path=model_dir,
    )
    model_card_path = model_dir / "README.md"
    model_card_path.write_text(model_card_content)
    logger.info("Generated model card (README.md)")

    # Upload all files in the model directory
    if commit_message is None:
        commit_message = (
            f"Upload ControlNet {condition_type} model - "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')}"
        )

    try:
        api.upload_folder(
            folder_path=str(model_dir),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
            token=token,
        )
        logger.info(f"Successfully uploaded model to: https://huggingface.co/{repo_id}")
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise

    return f"https://huggingface.co/{repo_id}"


def main():
    """Parse arguments and execute model upload."""
    parser = argparse.ArgumentParser(
        description="Upload trained ControlNet models to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Upload depth model to public repo
    python scripts/deploy_to_hub.py \\
        --model-path models/trained/controlnet-depth \\
        --repo-id myuser/controlnet-depth-sd15 \\
        --condition-type depth

    # Upload pose model to private repo
    python scripts/deploy_to_hub.py \\
        --model-path models/trained/controlnet-pose \\
        --repo-id myuser/controlnet-pose-sd15 \\
        --condition-type pose \\
        --private

    # Upload with custom commit message
    python scripts/deploy_to_hub.py \\
        --model-path models/trained/controlnet-edge \\
        --repo-id myuser/controlnet-edge-sd15 \\
        --condition-type edge \\
        --commit-message "v2.0 - improved edge conditioning"
        """,
    )

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model directory containing weights and config",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repository ID (e.g., 'username/model-name')",
    )
    parser.add_argument(
        "--condition-type",
        type=str,
        choices=["depth", "pose", "edge"],
        default="depth",
        help="Type of conditioning the model was trained for (default: depth)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository (default: public)",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default=None,
        help="Custom commit message for the upload",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (uses cached token if not provided)",
    )

    args = parser.parse_args()

    try:
        url = upload_model_to_hub(
            model_path=args.model_path,
            repo_id=args.repo_id,
            condition_type=args.condition_type,
            private=args.private,
            commit_message=args.commit_message,
            token=args.token,
        )
        logger.info(f"Deployment complete: {url}")
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
