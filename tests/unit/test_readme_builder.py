"""Unit tests for the ReadmeBuilder class.

Tests the header section and architecture diagrams implementation
against requirements 11.1, 11.2, 11.3, 11.4, 16.1, 16.2, 16.3.
"""

import pytest

import sys
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

from scripts.readme_builder import ReadmeBuilder


@pytest.fixture
def builder():
    """Create a ReadmeBuilder instance."""
    return ReadmeBuilder()


@pytest.fixture
def readme_content(builder):
    """Generate full README content."""
    return builder.build()


class TestHeaderSection:
    """Tests for _header_section() — Requirements 16.1, 16.2, 16.3."""

    def test_first_line_is_level1_heading(self, readme_content):
        """Requirement 16.1: First line must be # heading with project title."""
        first_line = readme_content.split("\n")[0]
        assert first_line == "# Controllable Image Generation with Fine-Grained Conditioning"

    def test_description_follows_title_immediately(self, readme_content):
        """Requirement 16.2: Single-sentence description on first non-empty line after title."""
        lines = readme_content.split("\n")
        # Find first non-empty line after the title
        description = None
        for line in lines[1:]:
            if line.strip():
                description = line
                break
        assert description is not None
        # Should not be a heading, badge, or other element
        assert not description.startswith("#")
        assert not description.startswith("!")
        assert not description.startswith("[")
        assert not description.startswith("---")

    def test_description_max_200_chars(self, readme_content):
        """Requirement 16.3: Description ≤200 characters."""
        lines = readme_content.split("\n")
        description = None
        for line in lines[1:]:
            if line.strip():
                description = line
                break
        assert description is not None
        assert len(description) <= 200

    def test_description_states_capability(self, readme_content):
        """Requirement 16.3: Description states what the system does."""
        lines = readme_content.split("\n")
        description = None
        for line in lines[1:]:
            if line.strip():
                description = line.lower()
                break
        # Should mention conditioning approach
        assert "controlnet" in description or "conditioning" in description


class TestArchitectureDiagrams:
    """Tests for _architecture_diagrams() — Requirements 11.1, 11.2, 11.3, 11.4."""

    def test_inference_diagram_has_text_encoder(self, readme_content):
        """Requirement 11.1: Inference diagram has Text Encoder."""
        assert "Text Encoder" in readme_content

    def test_inference_diagram_has_sd15_unet_frozen(self, readme_content):
        """Requirement 11.1: Inference diagram has SD1.5 UNet (frozen)."""
        assert "SD1.5 UNet (frozen)" in readme_content

    def test_inference_diagram_has_controlnet_trainable(self, readme_content):
        """Requirement 11.1: Inference diagram has ControlNet (trainable)."""
        assert "ControlNet" in readme_content
        assert "trainable" in readme_content

    def test_inference_diagram_has_zero_convolution(self, readme_content):
        """Requirement 11.1: Inference diagram has Zero Convolution layers."""
        assert "Zero Convolution" in readme_content

    def test_inference_diagram_has_noise_scheduler_ddim(self, readme_content):
        """Requirement 11.1: Inference diagram has Noise Scheduler (DDIM)."""
        assert "Noise Scheduler (DDIM)" in readme_content

    def test_inference_diagram_has_condition_map_input(self, readme_content):
        """Requirement 11.1: Inference diagram has Condition Map input."""
        assert "Condition Map" in readme_content

    def test_inference_diagram_has_text_prompt_input(self, readme_content):
        """Requirement 11.1: Inference diagram has Text Prompt input."""
        assert "Text Prompt" in readme_content

    def test_inference_diagram_has_generated_image_output(self, readme_content):
        """Requirement 11.1: Inference diagram has Generated Image output."""
        assert "Generated Image" in readme_content

    def test_training_diagram_has_raw_image(self, readme_content):
        """Requirement 11.2: Training diagram has Raw Image."""
        assert "Raw Image" in readme_content

    def test_training_diagram_has_condition_extractor(self, readme_content):
        """Requirement 11.2: Training diagram has Condition Extractor."""
        assert "Condition Extractor" in readme_content

    def test_training_diagram_has_all_condition_types(self, readme_content):
        """Requirement 11.2: Training diagram shows depth/pose/edge."""
        assert "depth" in readme_content
        assert "pose" in readme_content
        assert "edge" in readme_content

    def test_training_diagram_has_diffusion_loss(self, readme_content):
        """Requirement 11.2: Training diagram has Diffusion Loss."""
        assert "Diffusion Loss" in readme_content

    def test_code_block_lines_max_80_chars(self, readme_content):
        """Requirement 11.3: All lines in fenced code blocks ≤80 chars."""
        in_code_block = False
        for i, line in enumerate(readme_content.split("\n"), 1):
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                assert len(line) <= 80, (
                    f"Line {i} in code block is {len(line)} chars (max 80): {repr(line)}"
                )

    def test_diagrams_in_fenced_code_blocks(self, readme_content):
        """Requirement 11.3: Diagrams are inside fenced code blocks."""
        # Should have at least 2 code blocks (inference + training)
        code_block_count = readme_content.count("```")
        assert code_block_count >= 4  # 2 opening + 2 closing

    def test_architecture_between_description_and_results(self, readme_content):
        """Requirement 11.4: Architecture section between description and results."""
        lines = readme_content.split("\n")
        # Find description position
        desc_pos = None
        for i, line in enumerate(lines[1:], 1):
            if line.strip() and not line.startswith("#"):
                desc_pos = i
                break
        # Find architecture section position
        arch_pos = None
        for i, line in enumerate(lines):
            if "## Architecture" in line:
                arch_pos = i
                break
        assert desc_pos is not None
        assert arch_pos is not None
        assert arch_pos > desc_pos, "Architecture should come after description"


class TestBuildMethod:
    """Tests for the build() method assembly."""

    def test_build_returns_string(self, builder):
        """build() returns a string."""
        result = builder.build()
        assert isinstance(result, str)

    def test_build_with_no_args(self, builder):
        """build() works with no arguments."""
        result = builder.build()
        assert len(result) > 0

    def test_build_with_empty_metrics(self, builder):
        """build() works with empty metrics dict."""
        result = builder.build(metrics={})
        assert len(result) > 0

    def test_build_sections_in_order(self, builder):
        """Sections appear in correct order: header, architecture."""
        content = builder.build()
        title_pos = content.find("# Controllable Image Generation")
        arch_pos = content.find("## Architecture")
        assert title_pos < arch_pos


class TestResultsTable:
    """Tests for _results_table() — Requirements 12.1, 12.2, 12.3, 12.4."""

    def test_results_section_heading(self, builder):
        """Results table is under a ## Results heading."""
        content = builder.build(metrics={})
        assert "## Results" in content

    def test_table_has_three_columns(self, builder):
        """Requirement 12.1: Table has Model/Condition Type, FID Score, Alignment Score columns."""
        content = builder.build(metrics={})
        assert "Model / Condition Type" in content
        assert "FID Score" in content
        assert "Alignment Score" in content

    def test_table_has_depth_row(self, builder):
        """Requirement 12.1: Table has a row for depth condition type."""
        metrics = {"depth": {"fid_score": 15.2, "alignment_score": 0.85}}
        content = builder.build(metrics=metrics)
        assert "Depth" in content
        assert "15.2" in content
        assert "0.850" in content

    def test_table_has_pose_row(self, builder):
        """Requirement 12.1: Table has a row for pose condition type."""
        metrics = {"pose": {"fid_score": 18.5, "alignment_score": 0.72}}
        content = builder.build(metrics=metrics)
        assert "Pose" in content
        assert "18.5" in content
        assert "0.720" in content

    def test_table_has_edge_row(self, builder):
        """Requirement 12.1: Table has a row for edge condition type."""
        metrics = {"edge": {"fid_score": 12.3, "alignment_score": 0.91}}
        content = builder.build(metrics=metrics)
        assert "Edge" in content
        assert "12.3" in content
        assert "0.910" in content

    def test_table_has_baseline_row(self, builder):
        """Requirement 12.2: Table has Vanilla SD1.5 baseline row."""
        metrics = {"baseline": {"fid_score": 45.0, "alignment_score": 0.30}}
        content = builder.build(metrics=metrics)
        assert "Vanilla SD1.5" in content
        assert "45.0" in content

    def test_alignment_annotated_with_pearson_for_depth(self, builder):
        """Requirement 12.3: Depth alignment annotated with Pearson correlation."""
        metrics = {"depth": {"fid_score": 15.0, "alignment_score": 0.85}}
        content = builder.build(metrics=metrics)
        assert "Pearson correlation" in content

    def test_alignment_annotated_with_ssim_for_edge(self, builder):
        """Requirement 12.3: Edge alignment annotated with SSIM."""
        metrics = {"edge": {"fid_score": 12.0, "alignment_score": 0.91}}
        content = builder.build(metrics=metrics)
        assert "SSIM" in content

    def test_alignment_annotated_with_keypoint_for_pose(self, builder):
        """Requirement 12.3: Pose alignment annotated with normalized keypoint distance."""
        metrics = {"pose": {"fid_score": 18.0, "alignment_score": 0.72}}
        content = builder.build(metrics=metrics)
        assert "normalized keypoint distance" in content

    def test_metric_direction_indicated(self, builder):
        """Requirement 12.4: Metric direction is indicated (lower/higher is better)."""
        content = builder.build(metrics={})
        assert "lower is better" in content
        assert "higher is better" in content

    def test_missing_metrics_show_na(self, builder):
        """Missing metrics display N/A placeholders."""
        content = builder.build(metrics={})
        assert "N/A" in content

    def test_partial_metrics_show_na_for_missing(self, builder):
        """Partial metrics show N/A only for missing values."""
        metrics = {"depth": {"fid_score": 15.0}}
        content = builder.build(metrics=metrics)
        # Depth FID should be present
        assert "15.0" in content
        # Alignment for depth should be N/A since not provided
        assert "N/A" in content

    def test_results_section_after_architecture(self, builder):
        """Results section appears after architecture section."""
        content = builder.build(metrics={})
        arch_pos = content.find("## Architecture")
        results_pos = content.find("## Results")
        assert arch_pos < results_pos


class TestVisualGrid:
    """Tests for _visual_grid_section() — Requirements 13.1, 13.2, 13.3."""

    def test_visual_examples_heading(self, builder):
        """Visual grid is under a ## Visual Examples heading."""
        content = builder.build()
        assert "## Visual Examples" in content

    def test_grid_has_four_columns(self, builder):
        """Requirement 13.2: Grid has Input Image, Condition Map, Generated Result, Without ControlNet columns."""
        content = builder.build()
        assert "Input Image" in content
        assert "Condition Map" in content
        assert "Generated Result" in content
        assert "Without ControlNet" in content

    def test_grid_has_depth_section(self, builder):
        """Requirement 13.3: Grid has a depth conditioning section."""
        content = builder.build()
        assert "Depth Conditioning" in content

    def test_grid_has_pose_section(self, builder):
        """Requirement 13.3: Grid has a pose conditioning section."""
        content = builder.build()
        assert "Pose Conditioning" in content

    def test_grid_has_edge_section(self, builder):
        """Requirement 13.3: Grid has an edge conditioning section."""
        content = builder.build()
        assert "Edge Conditioning" in content

    def test_grid_has_at_least_two_rows_per_condition(self, builder):
        """Requirement 13.1: At least 2 rows per condition type (6+ total)."""
        content = builder.build()
        for condition_type in ["depth", "pose", "edge"]:
            # Each condition type should have at least 2 image rows
            assert content.count(f"{condition_type}_input_1") >= 1
            assert content.count(f"{condition_type}_input_2") >= 1

    def test_grid_has_minimum_six_rows(self, builder):
        """Requirement 13.1: Minimum 6 total rows."""
        content = builder.build()
        # Count image reference rows (each row has 4 images)
        image_rows = content.count("![input]")
        assert image_rows >= 6

    def test_visual_section_after_results(self, builder):
        """Visual grid section appears after results section."""
        content = builder.build()
        results_pos = content.find("## Results")
        visual_pos = content.find("## Visual Examples")
        assert results_pos < visual_pos


class TestReproducibilitySection:
    """Tests for _reproducibility_section() — Requirements 14.1, 14.2, 14.3."""

    def test_reproducibility_heading(self, builder):
        """Section has a ## How to Reproduce in Colab heading."""
        content = builder.build()
        assert "## How to Reproduce in Colab" in content

    def test_has_clone_command(self, builder):
        """Requirement 14.1: Includes git clone command."""
        content = builder.build()
        assert "git clone" in content

    def test_has_install_command(self, builder):
        """Requirement 14.1: Includes pip install command."""
        content = builder.build()
        assert "pip install -r requirements.txt" in content

    def test_has_dataset_command(self, builder):
        """Requirement 14.1: Includes dataset download command."""
        content = builder.build()
        assert "download_dataset" in content

    def test_has_extract_command(self, builder):
        """Requirement 14.1: Includes condition extraction command."""
        content = builder.build()
        assert "extract_conditions" in content

    def test_has_train_command(self, builder):
        """Requirement 14.1: Includes training command."""
        content = builder.build()
        assert "train.py" in content

    def test_has_evaluate_command(self, builder):
        """Requirement 14.1: Includes evaluation command."""
        content = builder.build()
        assert "evaluate.py" in content

    def test_commands_are_sequential(self, builder):
        """Requirement 14.1: Commands appear in sequential order."""
        content = builder.build()
        clone_pos = content.find("git clone")
        install_pos = content.find("pip install")
        dataset_pos = content.find("download_dataset")
        extract_pos = content.find("extract_conditions")
        train_pos = content.find("train.py")
        evaluate_pos = content.find("evaluate.py")
        assert clone_pos < install_pos < dataset_pos < extract_pos < train_pos < evaluate_pos

    def test_hardware_spec_t4_gpu(self, builder):
        """Requirement 14.2: Specifies T4 GPU hardware."""
        content = builder.build()
        assert "T4 GPU" in content

    def test_hardware_spec_15gb_vram(self, builder):
        """Requirement 14.2: Specifies 15 GB VRAM."""
        content = builder.build()
        assert "15 GB VRAM" in content or "15GB VRAM" in content

    def test_estimated_training_time(self, builder):
        """Requirement 14.2: Includes estimated training time."""
        content = builder.build()
        assert "Training Time" in content or "training time" in content

    def test_verification_steps(self, builder):
        """Requirement 14.3: Includes verification steps."""
        content = builder.build()
        assert "Verification" in content

    def test_reproducibility_after_visual_examples(self, builder):
        """Reproducibility section appears after visual examples."""
        content = builder.build()
        visual_pos = content.find("## Visual Examples")
        repro_pos = content.find("## How to Reproduce in Colab")
        assert visual_pos < repro_pos

    def test_commands_in_code_block(self, builder):
        """Commands are inside a fenced code block."""
        content = builder.build()
        # Find the reproducibility section
        repro_start = content.find("## How to Reproduce in Colab")
        repro_section = content[repro_start:]
        assert "```bash" in repro_section
        assert "```" in repro_section


class TestLinksSection:
    """Tests for _links_section(config) — Requirements 15.1, 15.2, 15.3."""

    def test_links_heading(self, builder):
        """Section has a ## Links heading."""
        content = builder.build()
        assert "## Links" in content

    def test_depth_model_repo_url(self, builder):
        """Requirement 15.1: URL to depth model repo."""
        content = builder.build()
        assert "https://huggingface.co/deepradadiya/controlnet-sd15-depth" in content

    def test_pose_model_repo_url(self, builder):
        """Requirement 15.1: URL to pose model repo."""
        content = builder.build()
        assert "https://huggingface.co/deepradadiya/controlnet-sd15-pose" in content

    def test_edge_model_repo_url(self, builder):
        """Requirement 15.1: URL to edge model repo."""
        content = builder.build()
        assert "https://huggingface.co/deepradadiya/controlnet-sd15-edge" in content

    def test_space_demo_url(self, builder):
        """Requirement 15.2: URL to HuggingFace Space demo."""
        content = builder.build()
        assert "https://huggingface.co/spaces/deepradadiya/controlnet-demo" in content

    def test_wandb_url(self, builder):
        """Requirement 15.3: URL to Weights & Biases training logs."""
        content = builder.build()
        assert "https://wandb.ai/" in content

    def test_links_are_clickable_markdown(self, builder):
        """Links use Markdown link syntax [text](url)."""
        content = builder.build()
        # Check at least one link uses markdown format
        assert "](https://huggingface.co/" in content

    def test_links_section_after_reproducibility(self, builder):
        """Links section appears after reproducibility section."""
        content = builder.build()
        repro_pos = content.find("## How to Reproduce in Colab")
        links_pos = content.find("## Links")
        assert repro_pos < links_pos

    def test_custom_config_uses_custom_username(self):
        """Links section uses username from config."""
        from scripts.readme_builder import ReadmeBuilder, PublishConfig
        builder = ReadmeBuilder()
        config = PublishConfig(hf_username="testuser")
        content = builder.build(config=config)
        assert "https://huggingface.co/testuser/controlnet-sd15-depth" in content
        assert "https://huggingface.co/testuser/controlnet-sd15-pose" in content
        assert "https://huggingface.co/testuser/controlnet-sd15-edge" in content

    def test_custom_config_uses_custom_space_name(self):
        """Links section uses space_repo_name from config."""
        from scripts.readme_builder import ReadmeBuilder, PublishConfig
        builder = ReadmeBuilder()
        config = PublishConfig(space_repo_name="my-demo")
        content = builder.build(config=config)
        assert "my-demo" in content


class TestTechStackSection:
    """Tests for _tech_stack_section() — Requirements 15.4, 15.5."""

    def test_tech_stack_heading(self, builder):
        """Section has a ## Tech Stack heading."""
        content = builder.build()
        assert "## Tech Stack" in content

    def test_has_pytorch(self, builder):
        """Requirement 15.4: Lists PyTorch."""
        content = builder.build()
        assert "PyTorch" in content

    def test_has_diffusers(self, builder):
        """Requirement 15.4: Lists diffusers library."""
        content = builder.build()
        assert "Diffusers" in content or "diffusers" in content

    def test_has_gradio(self, builder):
        """Requirement 15.4: Lists Gradio."""
        content = builder.build()
        assert "Gradio" in content

    def test_has_at_least_five_frameworks(self, builder):
        """Requirement 15.4: At least 5 frameworks/libraries listed."""
        content = builder.build()
        # Count known frameworks
        frameworks = ["PyTorch", "Diffusers", "Gradio", "Transformers",
                      "Weights & Biases", "safetensors"]
        count = sum(1 for f in frameworks if f in content)
        assert count >= 5, f"Only found {count} frameworks, need at least 5"

    def test_resume_bullet_point(self, builder):
        """Requirement 15.5: Includes resume bullet point."""
        content = builder.build()
        assert "Resume" in content or "resume" in content

    def test_resume_bullet_has_quantitative_metric(self, builder):
        """Requirement 15.5: Resume bullet references quantitative metric."""
        content = builder.build()
        # Should reference FID score or alignment score
        tech_section_start = content.find("## Tech Stack")
        tech_section = content[tech_section_start:]
        assert "FID" in tech_section or "alignment" in tech_section

    def test_tech_stack_after_links(self, builder):
        """Tech stack section appears after links section."""
        content = builder.build()
        links_pos = content.find("## Links")
        tech_pos = content.find("## Tech Stack")
        assert links_pos < tech_pos
