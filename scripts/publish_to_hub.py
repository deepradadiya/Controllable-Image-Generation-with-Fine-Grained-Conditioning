"""
Publish Trained ControlNet Models to HuggingFace Hub

This script provides a comprehensive publishing pipeline that:
- Pushes all three ControlNet adapters (depth, pose, edge) to individual repos
- Creates a combined multi-condition repository
- Deploys a HuggingFace Space with the Gradio demo
- Generates rich model cards with metrics and visual grids

Authentication Flow:
    1. --token CLI argument
    2. HF_TOKEN environment variable
    3. Cached huggingface-cli login token
    4. Exit with error if none available

Usage:
    python scripts/publish_to_hub.py \\
        --model-dir models/trained \\
        --metrics-dir evaluation/results \\
        --visual-grid-dir evaluation/grids \\
        --deploy-space \\
        --token $HF_TOKEN

Requirements Addressed: 7.4, 7.6
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path for imports
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

# Import PublishConfig directly from the models module to avoid
# triggering src/app/__init__.py which requires gradio
import importlib.util

_models_spec = importlib.util.spec_from_file_location(
    "src.app.models", _project_root / "src" / "app" / "models.py"
)
_models_module = importlib.util.module_from_spec(_models_spec)
_models_spec.loader.exec_module(_models_module)
PublishConfig = _models_module.PublishConfig
ModelCardMetadata = _models_module.ModelCardMetadata

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def resolve_token(cli_token: Optional[str] = None) -> str:
    """Resolve HuggingFace authentication token using fallback chain.

    Authentication priority:
        1. Token provided via --token CLI argument
        2. HF_TOKEN environment variable
        3. Cached token from huggingface-cli login
        4. Raises SystemExit with error message

    Args:
        cli_token: Token provided via CLI argument, or None.

    Returns:
        A valid HuggingFace API token string.

    Raises:
        SystemExit: If no valid token can be resolved from any source.
    """
    # 1. CLI token argument
    if cli_token:
        logger.info("Using token from --token CLI argument")
        return cli_token

    # 2. HF_TOKEN environment variable
    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        logger.info("Using token from HF_TOKEN environment variable")
        return env_token

    # 3. Cached huggingface-cli login token
    try:
        from huggingface_hub import HfFolder

        cached_token = HfFolder.get_token()
        if cached_token:
            logger.info("Using cached token from huggingface-cli login")
            return cached_token
    except ImportError:
        logger.warning("huggingface_hub not installed, cannot check cached token")
    except Exception as e:
        logger.warning(f"Failed to retrieve cached token: {e}")

    # 4. No token available — exit with error
    logger.error(
        "Authentication failed: No valid HuggingFace token found.\n"
        "Please provide a token using one of the following methods:\n"
        "  1. --token <your_token> CLI argument\n"
        "  2. Set the HF_TOKEN environment variable\n"
        "  3. Run 'huggingface-cli login' to cache your token"
    )
    sys.exit(1)


def create_publish_config(args: argparse.Namespace) -> PublishConfig:
    """Create a PublishConfig instance from parsed CLI arguments.

    Args:
        args: Parsed argparse namespace with CLI arguments.

    Returns:
        A configured PublishConfig instance ready for the publishing pipeline.
    """
    token = resolve_token(args.token)

    config = PublishConfig(
        model_dir=Path(args.model_dir),
        metrics_dir=Path(args.metrics_dir),
        visual_grid_dir=Path(args.visual_grid_dir),
        deploy_space=args.deploy_space,
        token=token,
    )

    return config


class AdapterPublisher:
    """Handles uploading individual adapter weights to HuggingFace Hub.

    Publishes each ControlNet adapter (depth, pose, edge) to its own
    HuggingFace repository with safetensors weights and config.json.

    Target repositories:
        - deepradadiya/controlnet-sd15-depth
        - deepradadiya/controlnet-sd15-pose
        - deepradadiya/controlnet-sd15-edge

    Requirements Addressed: 7.1, 7.2, 7.3, 7.5, 7.7
    """

    def __init__(self, config: PublishConfig) -> None:
        """Initialize AdapterPublisher with publishing configuration.

        Args:
            config: PublishConfig instance with repo details and token.
        """
        self.config = config
        from huggingface_hub import HfApi

        self.api = HfApi(token=config.token)

    def publish_adapter(self, condition_type: str) -> str:
        """Upload safetensors weights and config.json to an individual repo.

        Creates the repository if it doesn't exist, uploads the adapter's
        weight file (model.safetensors) and configuration file (config.json),
        then verifies the upload was successful.

        Args:
            condition_type: The condition type to publish ('depth', 'pose', or 'edge').

        Returns:
            The repository URL on success (e.g.,
            'https://huggingface.co/deepradadiya/controlnet-sd15-depth').

        Raises:
            FileNotFoundError: If the weight file or config file is missing locally.
            RuntimeError: If the upload or verification fails.
        """
        repo_id = self.config.get_adapter_repo_id(condition_type)
        weight_path = self.config.get_adapter_weight_path(condition_type)
        config_path = self.config.get_adapter_config_path(condition_type)

        # Check for missing weight files with clear error messages
        missing_files: List[str] = []
        if not weight_path.exists():
            missing_files.append(str(weight_path))
        if not config_path.exists():
            missing_files.append(str(config_path))

        if missing_files:
            raise FileNotFoundError(
                f"Cannot publish '{condition_type}' adapter: "
                f"missing files: {missing_files}"
            )

        logger.info(f"Publishing '{condition_type}' adapter to {repo_id}...")

        # Create repository if it doesn't exist
        self._create_repo_if_needed(repo_id)

        # Upload weight file (model.safetensors)
        logger.info(f"Uploading {weight_path.name} to {repo_id}...")
        self.api.upload_file(
            path_or_fileobj=str(weight_path),
            path_in_repo="model.safetensors",
            repo_id=repo_id,
            repo_type="model",
        )

        # Upload config file (config.json)
        logger.info(f"Uploading {config_path.name} to {repo_id}...")
        self.api.upload_file(
            path_or_fileobj=str(config_path),
            path_in_repo="config.json",
            repo_id=repo_id,
            repo_type="model",
        )

        # Verify upload
        repo_url = f"https://huggingface.co/{repo_id}"
        if self.verify_upload(repo_id):
            logger.info(
                f"Successfully published '{condition_type}' adapter. "
                f"Repository URL: {repo_url}"
            )
        else:
            logger.warning(
                f"Upload verification failed for '{condition_type}' adapter at {repo_id}. "
                f"The safetensors file may not be present in the repository."
            )

        return repo_url

    def _create_repo_if_needed(self, repo_id: str) -> None:
        """Create a public HuggingFace repository if it doesn't already exist.

        Uses exist_ok=True so that existing repositories are not modified.

        Args:
            repo_id: Full repository ID (e.g., 'deepradadiya/controlnet-sd15-depth').
        """
        logger.info(f"Ensuring repository exists: {repo_id}")
        self.api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            exist_ok=True,
            private=False,
        )

    def verify_upload(self, repo_id: str) -> bool:
        """Verify that the safetensors weight file exists in the repository.

        Lists the repository files and checks for the presence of
        'model.safetensors'.

        Args:
            repo_id: Full repository ID to verify.

        Returns:
            True if model.safetensors is found in the repo, False otherwise.
        """
        try:
            repo_files = self.api.list_repo_files(repo_id=repo_id, repo_type="model")
            if "model.safetensors" in repo_files:
                return True
            else:
                logger.warning(
                    f"Verification failed: 'model.safetensors' not found in {repo_id}. "
                    f"Files present: {repo_files}"
                )
                return False
        except Exception as e:
            logger.warning(f"Verification failed for {repo_id}: {e}")
            return False


def build_cli_parser() -> argparse.ArgumentParser:
    """Build the argparse CLI parser for the publishing script.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        description="Publish trained ControlNet adapters to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Publish all adapters with metrics
    python scripts/publish_to_hub.py \\
        --model-dir models/trained \\
        --metrics-dir evaluation/results \\
        --visual-grid-dir evaluation/grids

    # Publish and deploy Space
    python scripts/publish_to_hub.py \\
        --model-dir models/trained \\
        --metrics-dir evaluation/results \\
        --visual-grid-dir evaluation/grids \\
        --deploy-space

    # Publish with explicit token
    python scripts/publish_to_hub.py \\
        --model-dir models/trained \\
        --metrics-dir evaluation/results \\
        --visual-grid-dir evaluation/grids \\
        --token hf_xxxxxxxxxxxxx
        """,
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/trained",
        help="Path to directory containing trained model weights (default: models/trained)",
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        default="evaluation/results",
        help="Path to directory containing evaluation metrics (default: evaluation/results)",
    )
    parser.add_argument(
        "--visual-grid-dir",
        type=str,
        default="evaluation/grids",
        help="Path to directory containing visual grid images (default: evaluation/grids)",
    )
    parser.add_argument(
        "--deploy-space",
        action="store_true",
        default=False,
        help="Deploy the Gradio demo as a HuggingFace Space",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (falls back to HF_TOKEN env var or cached login)",
    )

    return parser


class CombinedPipelinePublisher:
    """Publishes the multi-condition combined repository to HuggingFace Hub.

    Organizes all three ControlNet adapter weights (depth, pose, edge) into
    a single repository with subdirectories and a JSON configuration file
    mapping condition types to their weight paths.

    Requirements Addressed: 8.1, 8.2, 8.3, 8.4
    """

    def __init__(self, config: PublishConfig) -> None:
        """Initialize the combined pipeline publisher.

        Args:
            config: Publishing configuration with paths and credentials.
        """
        self.config = config
        self._api: Optional["HfApi"] = None

    @property
    def api(self) -> "HfApi":
        """Lazy-initialize the HuggingFace API client."""
        if self._api is None:
            from huggingface_hub import HfApi

            self._api = HfApi(token=self.config.token)
        return self._api

    def publish_combined(self) -> str:
        """Upload all 3 adapters to the combined multi-condition repository.

        Creates the repository if it does not exist, organizes weights into
        subdirectories, generates a config.json, and uploads everything.

        Returns:
            The URL of the published combined repository.

        Raises:
            SystemExit: If any adapter weight files are missing locally.
        """
        repo_id = self.config.get_combined_repo_id()
        logger.info(f"Publishing combined pipeline to {repo_id}...")

        # Check for missing adapter weights — abort if any are absent
        missing = self.config.get_missing_adapters()
        if missing:
            missing_paths = [
                str(self.config.get_adapter_weight_path(ct)) for ct in missing
            ]
            logger.error(
                f"Cannot publish combined pipeline. "
                f"Missing adapter weight files: {missing_paths}"
            )
            sys.exit(1)

        # Create repository if it does not exist
        self.api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
        logger.info(f"Repository {repo_id} ready.")

        # Organize weights into a temporary directory structure
        upload_dir = self._organize_weights()

        # Build and write config.json
        config_json = self._build_config_json()
        config_path = upload_dir / "config.json"
        import json

        config_path.write_text(json.dumps(config_json, indent=2))
        logger.info("Generated config.json for combined pipeline.")

        # Upload the organized directory to the repository
        self.api.upload_folder(
            folder_path=str(upload_dir),
            repo_id=repo_id,
            repo_type="model",
        )

        repo_url = f"https://huggingface.co/{repo_id}"
        logger.info(f"Combined pipeline published successfully: {repo_url}")
        return repo_url

    def _organize_weights(self) -> Path:
        """Organize adapter weights into subdirectories for upload.

        Creates a temporary directory with the structure:
            depth/model.safetensors
            depth/config.json (if exists)
            pose/model.safetensors
            pose/config.json (if exists)
            edge/model.safetensors
            edge/config.json (if exists)

        Returns:
            Path to the temporary directory containing organized weights.
        """
        import shutil
        import tempfile

        upload_dir = Path(tempfile.mkdtemp(prefix="controlnet_combined_"))

        for condition_type in self.config.condition_types:
            subdir = upload_dir / condition_type
            subdir.mkdir(parents=True, exist_ok=True)

            # Copy weight file
            weight_src = self.config.get_adapter_weight_path(condition_type)
            shutil.copy2(str(weight_src), str(subdir / "model.safetensors"))

            # Copy config file if it exists
            config_src = self.config.get_adapter_config_path(condition_type)
            if config_src.exists():
                shutil.copy2(str(config_src), str(subdir / "config.json"))

        logger.info(
            f"Organized weights into subdirectories: "
            f"{[ct for ct in self.config.condition_types]}"
        )
        return upload_dir

    def _build_config_json(self) -> dict:
        """Generate JSON configuration mapping condition types to weight paths.

        Returns:
            Dictionary with base model info and condition type mappings:
            {
                "base_model": "runwayml/stable-diffusion-v1-5",
                "condition_types": {
                    "depth": {
                        "weight_path": "depth/model.safetensors",
                        "config_path": "depth/config.json"
                    },
                    ...
                }
            }
        """
        config_json: dict = {
            "base_model": self.config.base_model_id,
            "condition_types": {},
        }

        for condition_type in self.config.condition_types:
            config_json["condition_types"][condition_type] = {
                "weight_path": f"{condition_type}/model.safetensors",
                "config_path": f"{condition_type}/config.json",
            }

        return config_json


class SpaceDeployer:
    """Deploys the Gradio app as a HuggingFace Space.

    Creates or updates the HuggingFace Space 'deepradadiya/controlnet-demo'
    with the Gradio application code and requirements. Configures the Space
    with SDK=gradio, hardware=t4, and public visibility.

    Requirements Addressed: 9.1, 9.2, 9.3, 9.4
    """

    def __init__(self, config: PublishConfig) -> None:
        """Initialize the Space deployer.

        Args:
            config: Publishing configuration with credentials and repo details.
        """
        self.config = config
        self._api: Optional["HfApi"] = None

    @property
    def api(self) -> "HfApi":
        """Lazy-initialize the HuggingFace API client."""
        if self._api is None:
            from huggingface_hub import HfApi

            self._api = HfApi(token=self.config.token)
        return self._api

    def deploy_space(self) -> str:
        """Create or update the HuggingFace Space with the Gradio demo.

        Creates the Space repository if it doesn't exist, configures it with
        SDK=gradio, hardware=t4, and public visibility, then uploads the
        application files (app.py and requirements.txt).

        Returns:
            The URL of the deployed Space
            (e.g., 'https://huggingface.co/spaces/deepradadiya/controlnet-demo').

        Raises:
            SystemExit: If deployment fails due to authentication, network,
                or quota errors.
        """
        repo_id = self.config.get_space_repo_id()
        logger.info(f"Deploying Gradio Space to {repo_id}...")

        try:
            # Create or update the Space repository with metadata
            metadata = self._configure_space_metadata()
            self.api.create_repo(
                repo_id=repo_id,
                repo_type="space",
                space_sdk=metadata["sdk"],
                space_hardware=metadata["hardware"],
                exist_ok=True,
                private=False,
            )
            logger.info(
                f"Space repository {repo_id} ready "
                f"(sdk={metadata['sdk']}, hardware={metadata['hardware']})."
            )

            # Prepare and upload Space files
            space_files = self._prepare_space_files()
            for file_path in space_files:
                path_in_repo = file_path.name
                logger.info(f"Uploading {path_in_repo} to Space {repo_id}...")
                self.api.upload_file(
                    path_or_fileobj=str(file_path),
                    path_in_repo=path_in_repo,
                    repo_id=repo_id,
                    repo_type="space",
                )

            space_url = f"https://huggingface.co/spaces/{repo_id}"
            logger.info(f"Space deployed successfully: {space_url}")
            return space_url

        except Exception as e:
            error_msg = str(e)
            # Determine failure category for user-friendly messaging
            if "401" in error_msg or "403" in error_msg or "auth" in error_msg.lower():
                reason = f"Authentication error: {error_msg}"
            elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
                reason = f"Quota limitation: {error_msg}"
            elif (
                "timeout" in error_msg.lower()
                or "connection" in error_msg.lower()
                or "network" in error_msg.lower()
            ):
                reason = f"Network error: {error_msg}"
            else:
                reason = f"Deployment failed: {error_msg}"

            logger.error(
                f"Failed to deploy Space '{repo_id}'. {reason}"
            )
            sys.exit(1)

    def _prepare_space_files(self) -> List[Path]:
        """Gather the files needed for the HuggingFace Space deployment.

        Collects the Gradio application entry point (deployment/app.py) and
        the Space-specific requirements.txt (deployment/requirements.txt).

        Returns:
            List of Path objects pointing to the files to upload.

        Raises:
            SystemExit: If required Space files are not found locally.
        """
        # The Space entry point and requirements are in the deployment/ directory
        deployment_dir = _project_root / "deployment"
        app_file = deployment_dir / "app.py"
        requirements_file = deployment_dir / "requirements.txt"

        missing_files: List[str] = []
        if not app_file.exists():
            missing_files.append(str(app_file))
        if not requirements_file.exists():
            missing_files.append(str(requirements_file))

        if missing_files:
            logger.error(
                f"Cannot deploy Space: missing required files: {missing_files}"
            )
            sys.exit(1)

        logger.info(
            f"Prepared Space files: {app_file.name}, {requirements_file.name}"
        )
        return [app_file, requirements_file]

    def _configure_space_metadata(self) -> dict:
        """Configure the Space metadata settings.

        Returns:
            Dictionary with Space configuration:
            {
                "sdk": "gradio",
                "hardware": "t4",
                "visibility": "public"
            }
        """
        return {
            "sdk": "gradio",
            "hardware": "t4",
            "visibility": "public",
        }


class ModelCardGenerator:
    """Generates HuggingFace model cards with YAML front matter.

    Produces valid HuggingFace README.md content for each ControlNet adapter,
    including YAML front matter, training details, metrics, usage snippets,
    and visual grid references.

    Requirements Addressed: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6
    """

    def __init__(self, config: PublishConfig) -> None:
        """Initialize the model card generator.

        Args:
            config: Publishing configuration with base model and repo details.
        """
        self.config = config

    def generate_card(
        self,
        condition_type: str,
        metrics: Optional[dict] = None,
        visual_grid_path: Optional[str] = None,
    ) -> str:
        """Generate a complete HuggingFace model card README.md.

        Produces a valid HuggingFace README.md with YAML front matter,
        model description, training details, metrics, usage snippet,
        and visual grid.

        Args:
            condition_type: The condition type ('depth', 'pose', or 'edge').
            metrics: Optional dictionary with 'fid_score' and/or
                'alignment_score' keys. None or missing keys result in
                "not yet computed" notes.
            visual_grid_path: Optional path to the visual grid image file.
                None results in placeholder text.

        Returns:
            Complete model card content as a string, ready to be written
            as README.md in the HuggingFace repository.
        """
        if metrics is None:
            metrics = {}

        repo_id = self.config.get_adapter_repo_id(condition_type)

        # Build ModelCardMetadata for structured access
        metadata = ModelCardMetadata(
            condition_type=condition_type,
            repo_id=repo_id,
            base_model=self.config.base_model_id,
            fid_score=metrics.get("fid_score"),
            alignment_score=metrics.get("alignment_score"),
            visual_grid_path=visual_grid_path,
        )

        sections = [
            self._build_yaml_frontmatter(metadata),
            self._build_title_section(metadata),
            self._build_description_section(metadata),
            self._build_training_section(metadata),
            self._build_metrics_section(metadata),
            self._build_usage_snippet(repo_id),
            self._build_visual_grid_section(metadata),
        ]

        return "\n".join(sections)

    def _build_yaml_frontmatter(self, metadata: "ModelCardMetadata") -> str:
        """Build YAML front matter for the model card.

        Includes license, tags, and base_model fields as required by
        HuggingFace model card format.

        Args:
            metadata: ModelCardMetadata instance with card details.

        Returns:
            YAML front matter string enclosed in '---' delimiters.
        """
        tags_yaml = "\n".join(f"- {tag}" for tag in metadata.tags)
        return (
            "---\n"
            f"license: {metadata.license}\n"
            f"base_model: {metadata.base_model}\n"
            "tags:\n"
            f"{tags_yaml}\n"
            f"library_name: diffusers\n"
            f"pipeline_tag: image-to-image\n"
            "---\n"
        )

    def _build_title_section(self, metadata: "ModelCardMetadata") -> str:
        """Build the title section of the model card.

        Args:
            metadata: ModelCardMetadata instance with card details.

        Returns:
            Markdown title section string.
        """
        condition_display = {
            "depth": "Depth Map",
            "pose": "Pose Skeleton",
            "edge": "Edge Map",
        }
        display_name = condition_display.get(
            metadata.condition_type, metadata.condition_type
        )
        return (
            f"\n# ControlNet SD1.5 — {display_name} Conditioning\n"
        )

    def _build_description_section(self, metadata: "ModelCardMetadata") -> str:
        """Build the model description section.

        Args:
            metadata: ModelCardMetadata instance with card details.

        Returns:
            Markdown description section string.
        """
        return (
            f"\nA ControlNet adapter for {metadata.condition_type} conditioning, "
            f"fine-tuned on top of [{metadata.base_model}]"
            f"(https://huggingface.co/{metadata.base_model}).\n"
            f"\nThis model accepts a {metadata.condition_type} condition map "
            f"as spatial guidance to control image generation.\n"
        )

    def _build_training_section(self, metadata: "ModelCardMetadata") -> str:
        """Build the training details section.

        Args:
            metadata: ModelCardMetadata instance with card details.

        Returns:
            Markdown training details section string.
        """
        lines = [
            "\n## Training Details\n",
            f"- **Base Model:** {metadata.base_model}",
            f"- **Condition Type:** {metadata.condition_type}",
            f"- **Dataset:** {metadata.dataset}",
            f"- **Hardware:** {metadata.hardware}",
        ]
        if metadata.training_steps is not None:
            lines.append(f"- **Training Steps:** {metadata.training_steps}")
        return "\n".join(lines) + "\n"

    def _build_metrics_section(self, metadata: "ModelCardMetadata") -> str:
        """Build the metrics section with FID and alignment scores.

        If a metric is unavailable (None), includes a "not yet computed"
        note instead.

        Args:
            metadata: ModelCardMetadata instance with card details.

        Returns:
            Markdown metrics section string.
        """
        lines = ["\n## Evaluation Metrics\n"]

        if metadata.fid_score is not None:
            lines.append(
                f"- **FID Score:** {metadata.fid_score:.2f} (lower is better)"
            )
        else:
            lines.append("- **FID Score:** not yet computed")

        if metadata.alignment_score is not None:
            metric_name = metadata.alignment_metric_name
            lines.append(
                f"- **Alignment Score ({metric_name}):** "
                f"{metadata.alignment_score:.4f} (higher is better)"
            )
        else:
            metric_name = metadata.alignment_metric_name
            lines.append(
                f"- **Alignment Score ({metric_name}):** not yet computed"
            )

        return "\n".join(lines) + "\n"

    def _build_usage_snippet(self, repo_id: str) -> str:
        """Build a code usage snippet demonstrating how to load the adapter.

        The snippet is at most 5 lines of code as required.

        Args:
            repo_id: Full HuggingFace repository ID
                (e.g., 'deepradadiya/controlnet-sd15-depth').

        Returns:
            Markdown section with fenced code block containing the snippet.
        """
        return (
            "\n## Usage\n"
            "\n```python\n"
            "from diffusers import ControlNetModel, StableDiffusionControlNetPipeline\n"
            f"controlnet = ControlNetModel.from_pretrained(\"{repo_id}\")\n"
            "pipe = StableDiffusionControlNetPipeline.from_pretrained(\n"
            "    \"runwayml/stable-diffusion-v1-5\", controlnet=controlnet\n"
            ")\n"
            "```\n"
        )

    def _build_visual_grid_section(self, metadata: "ModelCardMetadata") -> str:
        """Build the visual grid section with example generations.

        If the visual grid image is unavailable, includes placeholder text.

        Args:
            metadata: ModelCardMetadata instance with card details.

        Returns:
            Markdown visual grid section string.
        """
        lines = ["\n## Example Generations\n"]

        if metadata.has_visual_grid:
            lines.append(
                f"![Visual Grid]({metadata.visual_grid_path})\n"
            )
        else:
            lines.append(
                "*Example generation grid is not yet available. "
                "Visual examples will be added after evaluation is complete.*\n"
            )

        return "\n".join(lines)


def _load_metrics(config: PublishConfig) -> dict:
    """Load evaluation metrics from the metrics directory.

    Looks for a metrics.json file in the metrics directory. If not found,
    returns an empty dictionary so the pipeline can continue without metrics.

    Args:
        config: PublishConfig with metrics_dir path.

    Returns:
        Dictionary with condition type keys mapping to dicts containing
        'fid_score' and 'alignment_score'. Empty dict if unavailable.
    """
    import json

    metrics_file = config.metrics_dir / "metrics.json"
    if not metrics_file.exists():
        logger.warning(
            f"Metrics file not found at {metrics_file}. "
            "Continuing without metrics."
        )
        return {}

    try:
        metrics = json.loads(metrics_file.read_text())
        logger.info(f"Loaded metrics from {metrics_file}")
        return metrics
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load metrics from {metrics_file}: {e}")
        return {}


def _get_visual_grid_path(config: PublishConfig, condition_type: str) -> Optional[str]:
    """Get the visual grid image path for a condition type.

    Looks for a PNG file in the visual grid directory named by condition type.

    Args:
        config: PublishConfig with visual_grid_dir path.
        condition_type: The condition type ('depth', 'pose', or 'edge').

    Returns:
        Path string to the visual grid image, or None if not found.
    """
    grid_path = config.visual_grid_dir / f"{condition_type}_grid.png"
    if grid_path.exists():
        return str(grid_path)
    return None


def main() -> None:
    """CLI entry point for the publishing pipeline.

    Orchestrates the full publishing flow in sequential order:
    1. Parse CLI args and create PublishConfig
    2. Authenticate with HuggingFace Hub (via resolve_token in create_publish_config)
    3. Publish individual adapters (depth, pose, edge) with model cards
    4. Publish combined multi-condition pipeline
    5. Deploy Space (if --deploy-space flag is set)
    6. Generate README.md using ReadmeBuilder and write to repository root

    Requirements Addressed: 7.1, 7.2, 7.3, 8.1, 9.1, 10.1
    """
    parser = build_cli_parser()
    args = parser.parse_args()

    logger.info("Starting ControlNet publishing pipeline...")

    # Step 1: Create configuration from CLI args (includes token resolution)
    config = create_publish_config(args)

    logger.info(f"Model directory: {config.model_dir}")
    logger.info(f"Metrics directory: {config.metrics_dir}")
    logger.info(f"Visual grid directory: {config.visual_grid_dir}")
    logger.info(f"Deploy Space: {config.deploy_space}")
    logger.info("Authentication successful")

    # Load metrics for model cards and README
    metrics = _load_metrics(config)

    # Step 2: Publish individual adapters with model cards
    logger.info("=" * 60)
    logger.info("STEP 1: Publishing individual adapters...")
    logger.info("=" * 60)

    adapter_publisher = AdapterPublisher(config)
    model_card_generator = ModelCardGenerator(config)
    published_urls: List[str] = []

    for condition_type in config.condition_types:
        try:
            # Generate model card for this adapter
            condition_metrics = metrics.get(condition_type, {})
            visual_grid_path = _get_visual_grid_path(config, condition_type)
            model_card_content = model_card_generator.generate_card(
                condition_type=condition_type,
                metrics=condition_metrics,
                visual_grid_path=visual_grid_path,
            )

            # Upload model card as README.md to the adapter repo
            repo_id = config.get_adapter_repo_id(condition_type)
            adapter_publisher._create_repo_if_needed(repo_id)
            adapter_publisher.api.upload_file(
                path_or_fileobj=model_card_content.encode("utf-8"),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
            )
            logger.info(f"Uploaded model card to {repo_id}")

            # Publish adapter weights and config
            url = adapter_publisher.publish_adapter(condition_type)
            published_urls.append(url)
        except FileNotFoundError as e:
            logger.error(f"Skipping '{condition_type}' adapter: {e}")
            continue
        except Exception as e:
            logger.error(f"Failed to publish '{condition_type}' adapter: {e}")
            continue

    logger.info(f"Published {len(published_urls)} adapter(s) successfully.")

    # Step 3: Publish combined multi-condition pipeline
    logger.info("=" * 60)
    logger.info("STEP 2: Publishing combined pipeline...")
    logger.info("=" * 60)

    combined_publisher = CombinedPipelinePublisher(config)
    combined_url = combined_publisher.publish_combined()
    logger.info(f"Combined pipeline URL: {combined_url}")

    # Step 4: Deploy Space (if --deploy-space flag is set)
    space_url: Optional[str] = None
    if config.deploy_space:
        logger.info("=" * 60)
        logger.info("STEP 3: Deploying HuggingFace Space...")
        logger.info("=" * 60)

        space_deployer = SpaceDeployer(config)
        space_url = space_deployer.deploy_space()
        logger.info(f"Space URL: {space_url}")
    else:
        logger.info("Skipping Space deployment (--deploy-space not set)")

    # Step 5: Generate README.md and write to repository root
    logger.info("=" * 60)
    logger.info("STEP 4: Generating README.md...")
    logger.info("=" * 60)

    # Import ReadmeBuilder from the sibling module
    _readme_builder_spec = importlib.util.spec_from_file_location(
        "readme_builder", _project_root / "scripts" / "readme_builder.py"
    )
    _readme_builder_module = importlib.util.module_from_spec(_readme_builder_spec)
    _readme_builder_spec.loader.exec_module(_readme_builder_module)
    ReadmeBuilder = _readme_builder_module.ReadmeBuilder

    readme_builder = ReadmeBuilder()
    readme_content = readme_builder.build(metrics=metrics, config=config)

    readme_path = _project_root / "README.md"
    readme_path.write_text(readme_content, encoding="utf-8")
    logger.info(f"README.md written to {readme_path}")

    # Summary
    logger.info("=" * 60)
    logger.info("Publishing pipeline complete!")
    logger.info("=" * 60)
    logger.info(f"Adapters published: {len(published_urls)}")
    for url in published_urls:
        logger.info(f"  - {url}")
    logger.info(f"Combined pipeline: {combined_url}")
    if space_url:
        logger.info(f"Space: {space_url}")
    logger.info(f"README: {readme_path}")


if __name__ == "__main__":
    main()
