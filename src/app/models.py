"""
Enhanced Data Models for Demo, Publish & README Pipeline

This module defines dataclasses for the enhanced Gradio demo application
and the HuggingFace publishing pipeline. It includes:

- EnhancedGenerationParams: Parameters for the 3-panel Gradio app generation
- PresetExample: Preset example combinations for the demo
- MultiConditionResult: Results from multi-condition comparison
- PublishConfig: Configuration for the publishing pipeline
- ModelCardMetadata: Metadata for generating HuggingFace model cards

Requirements satisfied: 3.1, 5.1, 8.3
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image as PILImage


# ============================================================================
# Constants
# ============================================================================

# Guidance scale constraints for the enhanced app (narrower than controls.py)
ENHANCED_GUIDANCE_SCALE_MIN = 1.0
ENHANCED_GUIDANCE_SCALE_MAX = 15.0
ENHANCED_GUIDANCE_SCALE_STEP = 0.5
ENHANCED_GUIDANCE_SCALE_DEFAULT = 7.5

# Preset prompt max length
PRESET_PROMPT_MAX_LENGTH = 200

# Valid condition types
VALID_CONDITION_TYPES = ["depth", "pose", "edge"]

# Display names for condition types (used in Gradio dropdown)
CONDITION_TYPE_DISPLAY_NAMES = {
    "depth": "Depth Map",
    "pose": "Pose Skeleton",
    "edge": "Edge Map",
}

# Reverse mapping: display name -> internal name
CONDITION_TYPE_FROM_DISPLAY = {v: k for k, v in CONDITION_TYPE_DISPLAY_NAMES.items()}


# ============================================================================
# Validation Helpers
# ============================================================================


def validate_guidance_scale(value: float) -> float:
    """Validate and return guidance scale value within allowed range.

    Args:
        value: The guidance scale value to validate.

    Returns:
        The validated guidance scale value (unchanged if valid).

    Raises:
        ValueError: If value is outside the range [1.0, 15.0].
    """
    if not (ENHANCED_GUIDANCE_SCALE_MIN <= value <= ENHANCED_GUIDANCE_SCALE_MAX):
        raise ValueError(
            f"guidance_scale must be between {ENHANCED_GUIDANCE_SCALE_MIN} "
            f"and {ENHANCED_GUIDANCE_SCALE_MAX}, got {value}"
        )
    return value


def validate_preset_prompt(prompt: str) -> str:
    """Validate preset prompt length.

    Args:
        prompt: The text prompt to validate.

    Returns:
        The validated prompt (unchanged if valid).

    Raises:
        ValueError: If prompt exceeds 200 characters.
    """
    if len(prompt) > PRESET_PROMPT_MAX_LENGTH:
        raise ValueError(
            f"Preset prompt must be at most {PRESET_PROMPT_MAX_LENGTH} characters, "
            f"got {len(prompt)} characters"
        )
    return prompt


def validate_condition_type(condition_type: str) -> str:
    """Validate condition type is one of the supported types.

    Accepts both internal names ('depth', 'pose', 'edge') and
    display names ('Depth Map', 'Pose Skeleton', 'Edge Map').

    Args:
        condition_type: The condition type to validate.

    Returns:
        The validated condition type (unchanged if valid).

    Raises:
        ValueError: If condition_type is not a valid option.
    """
    valid_options = VALID_CONDITION_TYPES + list(CONDITION_TYPE_DISPLAY_NAMES.values())
    if condition_type not in valid_options:
        raise ValueError(
            f"condition_type must be one of {valid_options}, got '{condition_type}'"
        )
    return condition_type


# ============================================================================
# Enhanced Gradio App Data Models
# ============================================================================


@dataclass
class EnhancedGenerationParams:
    """Parameters for the enhanced Gradio app generation with 3-panel display.

    This extends the concept from GenerationParameters in controls.py with
    a narrower guidance_scale range (1.0-15.0 vs 1.0-20.0) as specified
    in the design document.

    Attributes:
        prompt: Text prompt for image generation.
        condition_type: Conditioning method ('depth', 'pose', or 'edge').
        guidance_scale: Classifier-free guidance parameter (1.0-15.0, step 0.5).
        num_inference_steps: Number of DDIM sampling steps.
        conditioning_strength: ControlNet conditioning strength.
        seed: Random seed for reproducibility (None for random).
        width: Output image width in pixels.
        height: Output image height in pixels.
    """

    prompt: str
    condition_type: str  # "depth" | "pose" | "edge"
    guidance_scale: float = ENHANCED_GUIDANCE_SCALE_DEFAULT
    num_inference_steps: int = 30
    conditioning_strength: float = 1.0
    seed: Optional[int] = None
    width: int = 512
    height: int = 512

    def __post_init__(self) -> None:
        """Validate parameters after initialization."""
        validate_guidance_scale(self.guidance_scale)
        validate_condition_type(self.condition_type)

    def to_dict(self) -> Dict[str, object]:
        """Convert parameters to dictionary for pipeline integration."""
        return {
            "prompt": self.prompt,
            "condition_type": self.condition_type,
            "guidance_scale": self.guidance_scale,
            "num_inference_steps": self.num_inference_steps,
            "conditioning_scale": self.conditioning_strength,
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class PresetExample:
    """A preset example combination for the demo.

    Each preset provides a source image, condition type, and text prompt
    that users can click to quickly populate the demo inputs.

    Attributes:
        source_image_path: Path to the example source image file.
        condition_type: Display name of the condition type
            ('Depth Map', 'Pose Skeleton', or 'Edge Map').
        prompt: Text prompt for generation (max 200 characters).
    """

    source_image_path: str
    condition_type: str  # "Depth Map" | "Pose Skeleton" | "Edge Map"
    prompt: str

    def __post_init__(self) -> None:
        """Validate preset fields after initialization."""
        validate_preset_prompt(self.prompt)
        validate_condition_type(self.condition_type)

    @property
    def internal_condition_type(self) -> str:
        """Get the internal condition type name (depth/pose/edge)."""
        return CONDITION_TYPE_FROM_DISPLAY.get(self.condition_type, self.condition_type)


@dataclass
class MultiConditionResult:
    """Result from running all 3 condition types on one input.

    Used by the 'Generate with all 3 conditions' button to store
    results from each condition type, including partial failures.

    Attributes:
        condition_maps: Mapping of condition type to extracted condition map image.
        generated_images: Mapping of condition type to generated output image.
        errors: Mapping of condition type to error message (for failed types).
        status: Overall status message describing the result.
    """

    condition_maps: Dict[str, "PILImage.Image"] = field(default_factory=dict)
    generated_images: Dict[str, "PILImage.Image"] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)
    status: str = ""

    @property
    def successful_types(self) -> List[str]:
        """Return list of condition types that completed successfully."""
        return [t for t in VALID_CONDITION_TYPES if t not in self.errors]

    @property
    def failed_types(self) -> List[str]:
        """Return list of condition types that failed."""
        return [t for t in VALID_CONDITION_TYPES if t in self.errors]

    @property
    def all_succeeded(self) -> bool:
        """Check if all condition types completed successfully."""
        return len(self.errors) == 0

    @property
    def has_any_result(self) -> bool:
        """Check if at least one condition type produced a result."""
        return len(self.generated_images) > 0


# ============================================================================
# Publishing Pipeline Data Models
# ============================================================================


@dataclass
class PublishConfig:
    """Configuration for the HuggingFace publishing pipeline.

    Controls which repositories to publish to, where to find local
    model weights and metrics, and whether to deploy a Space.

    Attributes:
        hf_username: HuggingFace username for repository ownership.
        base_model_id: Base model identifier (Stable Diffusion version).
        condition_types: List of condition types to publish.
        repo_prefix: Prefix for individual adapter repository names.
        combined_repo_name: Name for the combined multi-condition repository.
        space_repo_name: Name for the HuggingFace Space.
        model_dir: Local path to trained model weights.
        metrics_dir: Local path to evaluation metrics results.
        visual_grid_dir: Local path to generated visual grid images.
        deploy_space: Whether to deploy the Gradio Space.
        token: HuggingFace API token (None to use env var or cached login).
    """

    hf_username: str = "deepradadiya"
    base_model_id: str = "runwayml/stable-diffusion-v1-5"
    condition_types: List[str] = field(
        default_factory=lambda: ["depth", "pose", "edge"]
    )
    repo_prefix: str = "controlnet-sd15"
    combined_repo_name: str = "controlnet-sd15-multi"
    space_repo_name: str = "controlnet-demo"
    model_dir: Path = field(default_factory=lambda: Path("models/trained"))
    metrics_dir: Path = field(default_factory=lambda: Path("evaluation/results"))
    visual_grid_dir: Path = field(default_factory=lambda: Path("evaluation/grids"))
    deploy_space: bool = False
    token: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        for ct in self.condition_types:
            if ct not in VALID_CONDITION_TYPES:
                raise ValueError(
                    f"Invalid condition type '{ct}'. "
                    f"Must be one of {VALID_CONDITION_TYPES}"
                )

    def get_adapter_repo_id(self, condition_type: str) -> str:
        """Get the full repository ID for an individual adapter.

        Args:
            condition_type: The condition type ('depth', 'pose', or 'edge').

        Returns:
            Full repo ID like 'deepradadiya/controlnet-sd15-depth'.
        """
        return f"{self.hf_username}/{self.repo_prefix}-{condition_type}"

    def get_combined_repo_id(self) -> str:
        """Get the full repository ID for the combined pipeline.

        Returns:
            Full repo ID like 'deepradadiya/controlnet-sd15-multi'.
        """
        return f"{self.hf_username}/{self.combined_repo_name}"

    def get_space_repo_id(self) -> str:
        """Get the full repository ID for the HuggingFace Space.

        Returns:
            Full repo ID like 'deepradadiya/controlnet-demo'.
        """
        return f"{self.hf_username}/{self.space_repo_name}"

    def get_adapter_weight_path(self, condition_type: str) -> Path:
        """Get the local path to an adapter's weight file.

        Args:
            condition_type: The condition type ('depth', 'pose', or 'edge').

        Returns:
            Path to the safetensors weight file.
        """
        return self.model_dir / condition_type / "model.safetensors"

    def get_adapter_config_path(self, condition_type: str) -> Path:
        """Get the local path to an adapter's config file.

        Args:
            condition_type: The condition type ('depth', 'pose', or 'edge').

        Returns:
            Path to the config.json file.
        """
        return self.model_dir / condition_type / "config.json"

    def get_missing_adapters(self) -> List[str]:
        """Check which adapter weight files are missing locally.

        Returns:
            List of condition types whose weight files are not found.
        """
        missing = []
        for ct in self.condition_types:
            if not self.get_adapter_weight_path(ct).exists():
                missing.append(ct)
        return missing


@dataclass
class ModelCardMetadata:
    """Metadata for generating a HuggingFace model card.

    Contains all information needed to produce a rich model card
    with YAML front matter, training details, metrics, and usage examples.

    Attributes:
        condition_type: The condition type this adapter handles.
        repo_id: Full HuggingFace repository ID.
        base_model: Base model identifier.
        license: License type for the model card YAML.
        tags: Tags for the model card YAML.
        dataset: Training dataset name.
        hardware: Hardware used for training.
        training_steps: Number of training steps completed (None if unknown).
        fid_score: FID metric score (None if not computed).
        alignment_score: Alignment metric score (None if not computed).
        visual_grid_path: Path to visual grid image (None if unavailable).
    """

    condition_type: str
    repo_id: str
    base_model: str = "runwayml/stable-diffusion-v1-5"
    license: str = "apache-2.0"
    tags: List[str] = field(
        default_factory=lambda: ["controlnet", "stable-diffusion", "image-generation"]
    )
    dataset: str = "COCO 2017 subset"
    hardware: str = "Google Colab T4 GPU"
    training_steps: Optional[int] = None
    fid_score: Optional[float] = None
    alignment_score: Optional[float] = None
    visual_grid_path: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate metadata after initialization."""
        if self.condition_type not in VALID_CONDITION_TYPES:
            raise ValueError(
                f"Invalid condition_type '{self.condition_type}'. "
                f"Must be one of {VALID_CONDITION_TYPES}"
            )

    @property
    def has_metrics(self) -> bool:
        """Check if any metrics are available."""
        return self.fid_score is not None or self.alignment_score is not None

    @property
    def has_visual_grid(self) -> bool:
        """Check if a visual grid image is available."""
        return self.visual_grid_path is not None

    @property
    def alignment_metric_name(self) -> str:
        """Get the specific alignment metric name for this condition type.

        Returns:
            The metric name: SSIM for edge, Pearson correlation for depth,
            normalized keypoint distance for pose.
        """
        metric_names = {
            "depth": "Pearson correlation",
            "pose": "normalized keypoint distance",
            "edge": "SSIM",
        }
        return metric_names[self.condition_type]
