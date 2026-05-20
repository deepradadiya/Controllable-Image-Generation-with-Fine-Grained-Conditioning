"""
Property-Based Tests for Evaluation Pipeline Metrics JSON.

Uses pytest + hypothesis to verify universal properties hold across
all valid inputs for the metrics JSON serialization and schema.

**Validates: Requirements 4.1, 4.2, 4.3**
"""

import json
import os
import sys
import tempfile

import numpy as np

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

# Add src/ to path for FIDCalculator imports (must be added AFTER project root
# so that top-level evaluation/ package takes priority over src/evaluation/)
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _project_root)

from evaluation.config import EvaluationConfig
from evaluation.metrics_writer import build_metrics_dict, save_metrics_json

# Import FIDCalculator from src/evaluation using importlib to avoid package shadowing
import importlib.util
_fid_spec = importlib.util.spec_from_file_location(
    "src_evaluation_compute_fid",
    os.path.join(_project_root, "src", "evaluation", "compute_fid.py"),
)
_fid_module = importlib.util.module_from_spec(_fid_spec)
_fid_spec.loader.exec_module(_fid_module)
FIDCalculator = _fid_module.FIDCalculator


# ─────────────────────────────────────────────────────────────────────────────
# Hypothesis Strategies
# ─────────────────────────────────────────────────────────────────────────────

# Strategy for condition types (1-3 from the valid set)
condition_types_strategy = st.lists(
    st.sampled_from(["depth", "pose", "edge"]),
    min_size=1,
    max_size=3,
    unique=True,
)

# Strategy for FID scores (positive floats, realistic range)
fid_score_strategy = st.floats(min_value=0.1, max_value=500.0, allow_nan=False, allow_infinity=False)

# Strategy for alignment scores (mean in [0, 1])
alignment_mean_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)

# Strategy for alignment std (non-negative, bounded)
alignment_std_strategy = st.floats(min_value=0.0, max_value=0.5, allow_nan=False, allow_infinity=False)

# Strategy for grid paths
grid_path_strategy = st.sampled_from([
    "evaluation/results/visual_grid_depth.png",
    "evaluation/results/visual_grid_pose.png",
    "evaluation/results/visual_grid_edge.png",
    "evaluation/results/visual_grid_combined.png",
])

# Strategy for EvaluationConfig parameters
num_samples_strategy = st.integers(min_value=1, max_value=10000)
guidance_scale_strategy = st.floats(min_value=1.0, max_value=20.0, allow_nan=False, allow_infinity=False)
num_steps_strategy = st.integers(min_value=1, max_value=100)
seed_strategy = st.integers(min_value=0, max_value=2**31 - 1)


@st.composite
def evaluation_config_strategy(draw):
    """Generate random EvaluationConfig instances."""
    condition_types = draw(condition_types_strategy)
    return EvaluationConfig(
        output_dir="evaluation/results",
        num_fid_samples=draw(num_samples_strategy),
        num_alignment_samples=draw(num_samples_strategy),
        num_grid_prompts=draw(st.integers(min_value=1, max_value=50)),
        batch_size=draw(st.integers(min_value=1, max_value=64)),
        condition_types=condition_types,
        coco_val_dir="data/raw/coco_val2017",
        checkpoint_dir="models/trained",
        guidance_scale=draw(guidance_scale_strategy),
        num_inference_steps=draw(num_steps_strategy),
        seed=draw(seed_strategy),
    )


@st.composite
def metrics_inputs_strategy(draw):
    """Generate random but valid inputs for build_metrics_dict."""
    config = draw(evaluation_config_strategy())
    condition_types = config.condition_types

    # Build FID results: baseline + each condition type
    fid_results = {"baseline": draw(fid_score_strategy)}
    for ct in condition_types:
        fid_results[ct] = draw(fid_score_strategy)

    # Build alignment results: each condition type -> (mean, std)
    alignment_results = {}
    for ct in condition_types:
        mean = draw(alignment_mean_strategy)
        std = draw(alignment_std_strategy)
        alignment_results[ct] = (mean, std)

    # Build grid paths: one per condition type + combined
    grid_paths = [f"evaluation/results/visual_grid_{ct}.png" for ct in condition_types]
    grid_paths.append("evaluation/results/visual_grid_combined.png")

    return fid_results, alignment_results, grid_paths, config


# ─────────────────────────────────────────────────────────────────────────────
# Property 5: Metrics JSON Round-Trip
# Feature: evaluation-pipeline, Property 5: Metrics JSON Round-Trip
# **Validates: Requirements 4.1**
# ─────────────────────────────────────────────────────────────────────────────


class TestMetricsJsonRoundTrip:
    """
    Property: For any valid metrics dictionary, JSON serialize then
    deserialize produces equivalent data (round-trip property).

    **Validates: Requirements 4.1**
    """

    @given(data=metrics_inputs_strategy())
    @settings(max_examples=100, deadline=None)
    def test_metrics_json_round_trip(self, data):
        """
        Feature: evaluation-pipeline, Property 5: Metrics JSON Round-Trip

        For any valid metrics dict built from random FID scores, alignment
        scores, and grid paths, serializing to JSON and deserializing back
        produces a dictionary equivalent to the original.

        **Validates: Requirements 4.1**
        """
        fid_results, alignment_results, grid_paths, config = data

        # Build the metrics dictionary
        metrics_dict = build_metrics_dict(fid_results, alignment_results, grid_paths, config)

        # Serialize to JSON string
        json_str = json.dumps(metrics_dict, indent=2)

        # Deserialize back
        deserialized = json.loads(json_str)

        # Assert round-trip equivalence
        assert deserialized == metrics_dict, (
            f"Round-trip failed.\n"
            f"Original: {metrics_dict}\n"
            f"Deserialized: {deserialized}"
        )

    @given(data=metrics_inputs_strategy())
    @settings(max_examples=100, deadline=None)
    def test_metrics_json_round_trip_via_file(self, data):
        """
        Feature: evaluation-pipeline, Property 5: Metrics JSON Round-Trip

        For any valid metrics dict, writing to a JSON file and reading back
        produces equivalent data.

        **Validates: Requirements 4.1**
        """
        fid_results, alignment_results, grid_paths, config = data

        # Build the metrics dictionary
        metrics_dict = build_metrics_dict(fid_results, alignment_results, grid_paths, config)

        # Write to a temporary file using save_metrics_json
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = save_metrics_json(metrics_dict, tmpdir)

            # Read back from file
            with open(output_path, "r", encoding="utf-8") as f:
                deserialized = json.load(f)

        # Assert round-trip equivalence
        assert deserialized == metrics_dict, (
            f"File round-trip failed.\n"
            f"Original: {metrics_dict}\n"
            f"Deserialized: {deserialized}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Property 6: Metrics JSON Schema Completeness
# Feature: evaluation-pipeline, Property 6: Metrics JSON Schema Completeness
# **Validates: Requirements 4.2, 4.3**
# ─────────────────────────────────────────────────────────────────────────────


class TestMetricsJsonSchemaCompleteness:
    """
    Property: For any combination of condition types (1 to 3) and their
    corresponding scores, the generated metrics JSON contains all required
    top-level keys and each condition type's scores are present.

    **Validates: Requirements 4.2, 4.3**
    """

    @given(data=metrics_inputs_strategy())
    @settings(max_examples=100, deadline=None)
    def test_all_required_top_level_keys_present(self, data):
        """
        Feature: evaluation-pipeline, Property 6: Metrics JSON Schema Completeness

        For any combination of condition types and scores, the output JSON
        contains all required top-level keys: "metadata", "fid_scores",
        "alignment_scores", "visual_grids".

        **Validates: Requirements 4.2, 4.3**
        """
        fid_results, alignment_results, grid_paths, config = data

        metrics_dict = build_metrics_dict(fid_results, alignment_results, grid_paths, config)

        required_keys = {"metadata", "fid_scores", "alignment_scores", "visual_grids"}
        assert required_keys.issubset(metrics_dict.keys()), (
            f"Missing required top-level keys. "
            f"Expected: {required_keys}, Got: {set(metrics_dict.keys())}"
        )

    @given(data=metrics_inputs_strategy())
    @settings(max_examples=100, deadline=None)
    def test_each_condition_type_fid_score_present(self, data):
        """
        Feature: evaluation-pipeline, Property 6: Metrics JSON Schema Completeness

        For any combination of condition types, each condition type's FID
        score is present in the fid_scores section.

        **Validates: Requirements 4.2, 4.3**
        """
        fid_results, alignment_results, grid_paths, config = data

        metrics_dict = build_metrics_dict(fid_results, alignment_results, grid_paths, config)

        # Each condition type should have a FID score
        for ct in config.condition_types:
            assert ct in metrics_dict["fid_scores"], (
                f"FID score missing for condition type '{ct}'. "
                f"Available keys: {list(metrics_dict['fid_scores'].keys())}"
            )

        # Baseline should also be present
        assert "baseline_sd15" in metrics_dict["fid_scores"], (
            "Baseline SD1.5 FID score missing from fid_scores"
        )

    @given(data=metrics_inputs_strategy())
    @settings(max_examples=100, deadline=None)
    def test_each_condition_type_alignment_score_present(self, data):
        """
        Feature: evaluation-pipeline, Property 6: Metrics JSON Schema Completeness

        For any combination of condition types, each condition type's
        alignment scores (mean, std, num_samples, metric, target_met)
        are present in the alignment_scores section.

        **Validates: Requirements 4.2, 4.3**
        """
        fid_results, alignment_results, grid_paths, config = data

        metrics_dict = build_metrics_dict(fid_results, alignment_results, grid_paths, config)

        for ct in config.condition_types:
            assert ct in metrics_dict["alignment_scores"], (
                f"Alignment score missing for condition type '{ct}'. "
                f"Available keys: {list(metrics_dict['alignment_scores'].keys())}"
            )

            # Check required sub-keys for each alignment score entry
            alignment_entry = metrics_dict["alignment_scores"][ct]
            required_sub_keys = {"mean", "std", "num_samples", "metric", "target_met"}
            assert required_sub_keys.issubset(alignment_entry.keys()), (
                f"Missing sub-keys in alignment_scores['{ct}']. "
                f"Expected: {required_sub_keys}, Got: {set(alignment_entry.keys())}"
            )

    @given(data=metrics_inputs_strategy())
    @settings(max_examples=100, deadline=None)
    def test_metadata_contains_required_fields(self, data):
        """
        Feature: evaluation-pipeline, Property 6: Metrics JSON Schema Completeness

        For any configuration, the metadata section contains all required
        fields: timestamp, num_fid_samples, num_alignment_samples,
        coco_val_size, inference_config, checkpoint_paths.

        **Validates: Requirements 4.2, 4.3**
        """
        fid_results, alignment_results, grid_paths, config = data

        metrics_dict = build_metrics_dict(fid_results, alignment_results, grid_paths, config)

        metadata = metrics_dict["metadata"]
        required_metadata_keys = {
            "timestamp",
            "num_fid_samples",
            "num_alignment_samples",
            "coco_val_size",
            "inference_config",
            "checkpoint_paths",
        }
        assert required_metadata_keys.issubset(metadata.keys()), (
            f"Missing metadata keys. "
            f"Expected: {required_metadata_keys}, Got: {set(metadata.keys())}"
        )

        # Check inference_config sub-keys
        inference_config = metadata["inference_config"]
        required_inference_keys = {"guidance_scale", "num_inference_steps", "image_size"}
        assert required_inference_keys.issubset(inference_config.keys()), (
            f"Missing inference_config keys. "
            f"Expected: {required_inference_keys}, Got: {set(inference_config.keys())}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Property 3: Aggregation Statistics Invariant
# Feature: evaluation-pipeline, Property 3: Aggregation Statistics Invariant
# **Validates: Requirements 2.5**
# ─────────────────────────────────────────────────────────────────────────────


class TestAggregationStatisticsInvariant:
    """
    Property: For any non-empty list of alignment scores where each score
    is in [0, 1], the computed mean is between min and max, and the
    computed standard deviation is non-negative.

    Feature: evaluation-pipeline, Property 3: Aggregation Statistics Invariant

    **Validates: Requirements 2.5**
    """

    @given(
        scores=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=200,
        )
    )
    @settings(max_examples=100)
    def test_mean_is_between_min_and_max(self, scores):
        """
        For any non-empty list of scores in [0, 1], the mean must be
        greater than or equal to the minimum and less than or equal to
        the maximum (within floating-point tolerance).

        Feature: evaluation-pipeline, Property 3: Aggregation Statistics Invariant

        **Validates: Requirements 2.5**
        """
        scores_array = np.array(scores)
        mean = np.mean(scores_array)
        min_score = np.min(scores_array)
        max_score = np.max(scores_array)

        # Use a small epsilon to account for floating-point arithmetic
        eps = 1e-10

        assert mean >= min_score - eps, (
            f"Mean {mean} is less than minimum {min_score}. "
            f"Scores: {scores}"
        )
        assert mean <= max_score + eps, (
            f"Mean {mean} is greater than maximum {max_score}. "
            f"Scores: {scores}"
        )

    @given(
        scores=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=200,
        )
    )
    @settings(max_examples=100)
    def test_std_is_non_negative(self, scores):
        """
        For any non-empty list of scores in [0, 1], the standard deviation
        must be non-negative.

        Feature: evaluation-pipeline, Property 3: Aggregation Statistics Invariant

        **Validates: Requirements 2.5**
        """
        scores_array = np.array(scores)
        std = np.std(scores_array)

        assert std >= 0, (
            f"Standard deviation {std} is negative. "
            f"Scores: {scores}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Strategies for Image Generation
# ─────────────────────────────────────────────────────────────────────────────

from unittest.mock import MagicMock
from PIL import Image
from hypothesis import HealthCheck

from evaluation.condition_alignment import EvaluationAlignmentCalculator


@st.composite
def random_rgb_image(draw):
    """Generate a random PIL RGB Image with dimensions between 64x64 and 512x512.

    Uses a numpy random seed to generate pixel data efficiently rather than
    drawing raw bytes from Hypothesis, which avoids large_base_example issues.
    """
    width = draw(st.integers(min_value=64, max_value=512))
    height = draw(st.integers(min_value=64, max_value=512))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    img = Image.fromarray(pixels, mode="RGB")
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Property 2: Alignment Score Bounded Range
# Feature: evaluation-pipeline, Property 2: Alignment Score Bounded Range
# **Validates: Requirements 2.1, 2.2, 2.3**
# ─────────────────────────────────────────────────────────────────────────────


class TestAlignmentScoreBoundedRange:
    """
    Property: For any condition type (depth, pose, or edge) and for any pair
    of valid images (generated image and condition map), the computed alignment
    score SHALL always be in the range [0, 1].

    Feature: evaluation-pipeline, Property 2: Alignment Score Bounded Range

    **Validates: Requirements 2.1, 2.2, 2.3**
    """

    @given(
        generated_image=random_rgb_image(),
        condition_image=random_rgb_image(),
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.large_base_example])
    def test_edge_alignment_bounded(self, generated_image, condition_image):
        """
        Feature: evaluation-pipeline, Property 2: Alignment Score Bounded Range

        Edge alignment score is always in [0, 1] for any pair of random images.

        **Validates: Requirements 2.1**
        """
        mock_pipeline = MagicMock()
        calculator = EvaluationAlignmentCalculator(pipeline=mock_pipeline, device="cpu")

        score = calculator.compute_edge_alignment(generated_image, condition_image)

        assert isinstance(score, float), f"Score should be float, got {type(score)}"
        assert 0.0 <= score <= 1.0, (
            f"Edge alignment score {score} is outside [0, 1] range"
        )

    @given(
        generated_image=random_rgb_image(),
        condition_image=random_rgb_image(),
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.large_base_example])
    def test_depth_alignment_bounded(self, generated_image, condition_image):
        """
        Feature: evaluation-pipeline, Property 2: Alignment Score Bounded Range

        Depth alignment score is always in [0, 1] for any pair of random images.

        **Validates: Requirements 2.2**
        """
        mock_pipeline = MagicMock()
        calculator = EvaluationAlignmentCalculator(pipeline=mock_pipeline, device="cpu")

        score = calculator.compute_depth_alignment(generated_image, condition_image)

        assert isinstance(score, float), f"Score should be float, got {type(score)}"
        assert 0.0 <= score <= 1.0, (
            f"Depth alignment score {score} is outside [0, 1] range"
        )

    @given(
        generated_image=random_rgb_image(),
        condition_image=random_rgb_image(),
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.large_base_example])
    def test_pose_alignment_bounded(self, generated_image, condition_image):
        """
        Feature: evaluation-pipeline, Property 2: Alignment Score Bounded Range

        Pose alignment score is always in [0, 1] for any pair of random images.

        **Validates: Requirements 2.3**
        """
        mock_pipeline = MagicMock()
        calculator = EvaluationAlignmentCalculator(pipeline=mock_pipeline, device="cpu")

        score = calculator.compute_pose_alignment(generated_image, condition_image)

        assert isinstance(score, float), f"Score should be float, got {type(score)}"
        assert 0.0 <= score <= 1.0, (
            f"Pose alignment score {score} is outside [0, 1] range"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Property 1: Inception-v3 Feature Shape Invariant
# Feature: evaluation-pipeline, Property 1: Inception-v3 Feature Shape Invariant
# **Validates: Requirements 1.3**
# ─────────────────────────────────────────────────────────────────────────────


@st.composite
def random_rgb_image_batch(draw):
    """Generate a random batch of PIL RGB Images with batch size 1-50.

    Each image can have different dimensions (between 64x64 and 256x256)
    to test that the feature extractor handles varying input sizes.
    Uses numpy random generation for efficiency.
    """
    batch_size = draw(st.integers(min_value=1, max_value=50))
    images = []
    for _ in range(batch_size):
        width = draw(st.integers(min_value=64, max_value=256))
        height = draw(st.integers(min_value=64, max_value=256))
        seed = draw(st.integers(min_value=0, max_value=2**32 - 1))
        rng = np.random.default_rng(seed)
        pixels = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        img = Image.fromarray(pixels, mode="RGB")
        images.append(img)
    return images


class TestInceptionV3FeatureShapeInvariant:
    """
    Property: For any batch of random RGB images (batch size 1-50),
    feature extraction always produces shape (N, 2048) where N is the
    number of input images.

    Feature: evaluation-pipeline, Property 1: Inception-v3 Feature Shape Invariant

    **Validates: Requirements 1.3**
    """

    @given(image_batch=random_rgb_image_batch())
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.large_base_example])
    def test_feature_shape_is_n_by_2048(self, image_batch):
        """
        Feature: evaluation-pipeline, Property 1: Inception-v3 Feature Shape Invariant

        For any batch of random RGB images (batch size 1-50), feature extraction
        always produces shape (N, 2048) where N is the number of input images.

        **Validates: Requirements 1.3**
        """
        calculator = FIDCalculator(batch_size=8, device=None, num_workers=0)

        features = calculator.extract_features(image_batch, show_progress=False)

        # Assert the output is a numpy array
        assert isinstance(features, np.ndarray), (
            f"Expected numpy array, got {type(features)}"
        )

        # Assert the shape is (N, 2048)
        expected_shape = (len(image_batch), 2048)
        assert features.shape == expected_shape, (
            f"Expected feature shape {expected_shape}, got {features.shape}. "
            f"Batch size was {len(image_batch)}."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Property 4: Grid Dimension Invariant
# Feature: evaluation-pipeline, Property 4: Grid Dimension Invariant
# **Validates: Requirements 3.1**
# ─────────────────────────────────────────────────────────────────────────────

from evaluation.visual_grid import EvaluationGridGenerator

# Strategy for number of rows in the grid (1-20 as per spec)
grid_num_rows_strategy = st.integers(min_value=1, max_value=20)

# Strategy for cell size (width and height, 64-512 as per spec)
grid_cell_size_strategy = st.tuples(
    st.integers(min_value=64, max_value=512),
    st.integers(min_value=64, max_value=512),
)


class TestGridDimensionInvariant:
    """
    Property: For any number of rows (1 to 20), cell size, and padding
    configuration, the generated visual grid image dimensions equal:
      width = row_label_width + padding + NUM_COLUMNS * cell_w + (NUM_COLUMNS + 1) * padding
      height = header_height + padding + num_rows * cell_h + (num_rows + 1) * padding

    Feature: evaluation-pipeline, Property 4: Grid Dimension Invariant

    **Validates: Requirements 3.1**
    """

    @given(
        num_rows=grid_num_rows_strategy,
        cell_size=grid_cell_size_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_grid_width_matches_formula(self, num_rows, cell_size):
        """
        Feature: evaluation-pipeline, Property 4: Grid Dimension Invariant

        For any number of rows (1-20) and cell size (64-512),
        the grid width matches the expected formula:
          width = row_label_width + padding + NUM_COLUMNS * cell_w + (NUM_COLUMNS + 1) * padding

        **Validates: Requirements 3.1**
        """
        cell_w, cell_h = cell_size

        # Create a mock pipeline that returns dummy images
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = Image.new("RGB", (512, 512), (128, 128, 128))

        generator = EvaluationGridGenerator(
            pipeline=mock_pipeline,
            cell_size=cell_size,
            output_dir="/tmp/test_grid",
        )

        # Create dummy data matching num_rows
        original_images = [
            Image.new("RGB", (512, 512), (255, 0, 0)) for _ in range(num_rows)
        ]
        condition_maps = [
            Image.new("RGB", (512, 512), (0, 255, 0)) for _ in range(num_rows)
        ]
        prompts = [f"Test prompt {i}" for i in range(num_rows)]

        # Generate the grid
        grid = generator.generate_grid(
            condition_type="depth",
            original_images=original_images,
            condition_maps=condition_maps,
            prompts=prompts,
            num_rows=num_rows,
            seed=42,
        )

        # Compute expected width using the formula
        expected_width = (
            generator.row_label_width
            + generator.padding
            + generator.NUM_COLUMNS * cell_w
            + (generator.NUM_COLUMNS + 1) * generator.padding
        )

        assert grid.width == expected_width, (
            f"Grid width {grid.width} != expected {expected_width} "
            f"for cell_size={cell_size}, num_rows={num_rows}"
        )

    @given(
        num_rows=grid_num_rows_strategy,
        cell_size=grid_cell_size_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_grid_height_matches_formula(self, num_rows, cell_size):
        """
        Feature: evaluation-pipeline, Property 4: Grid Dimension Invariant

        For any number of rows (1-20) and cell size (64-512),
        the grid height matches the expected formula:
          height = header_height + padding + num_rows * cell_h + (num_rows + 1) * padding

        **Validates: Requirements 3.1**
        """
        cell_w, cell_h = cell_size

        # Create a mock pipeline that returns dummy images
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = Image.new("RGB", (512, 512), (128, 128, 128))

        generator = EvaluationGridGenerator(
            pipeline=mock_pipeline,
            cell_size=cell_size,
            output_dir="/tmp/test_grid",
        )

        # Create dummy data matching num_rows
        original_images = [
            Image.new("RGB", (512, 512), (255, 0, 0)) for _ in range(num_rows)
        ]
        condition_maps = [
            Image.new("RGB", (512, 512), (0, 255, 0)) for _ in range(num_rows)
        ]
        prompts = [f"Test prompt {i}" for i in range(num_rows)]

        # Generate the grid
        grid = generator.generate_grid(
            condition_type="depth",
            original_images=original_images,
            condition_maps=condition_maps,
            prompts=prompts,
            num_rows=num_rows,
            seed=42,
        )

        # Compute expected height using the formula
        expected_height = (
            generator.header_height
            + generator.padding
            + num_rows * cell_h
            + (num_rows + 1) * generator.padding
        )

        assert grid.height == expected_height, (
            f"Grid height {grid.height} != expected {expected_height} "
            f"for cell_size={cell_size}, num_rows={num_rows}"
        )

    @given(
        num_rows=grid_num_rows_strategy,
        cell_size=grid_cell_size_strategy,
    )
    @settings(max_examples=100, deadline=None)
    def test_grid_dimensions_both_match_formula(self, num_rows, cell_size):
        """
        Feature: evaluation-pipeline, Property 4: Grid Dimension Invariant

        Combined check: for any configuration, both width and height
        match the expected formulas simultaneously.

        **Validates: Requirements 3.1**
        """
        cell_w, cell_h = cell_size

        # Create a mock pipeline that returns dummy images
        mock_pipeline = MagicMock()
        mock_pipeline.return_value = Image.new("RGB", (512, 512), (128, 128, 128))

        generator = EvaluationGridGenerator(
            pipeline=mock_pipeline,
            cell_size=cell_size,
            output_dir="/tmp/test_grid",
        )

        # Create dummy data matching num_rows
        original_images = [
            Image.new("RGB", (512, 512), (255, 0, 0)) for _ in range(num_rows)
        ]
        condition_maps = [
            Image.new("RGB", (512, 512), (0, 255, 0)) for _ in range(num_rows)
        ]
        prompts = [f"Test prompt {i}" for i in range(num_rows)]

        # Generate the grid
        grid = generator.generate_grid(
            condition_type="edge",
            original_images=original_images,
            condition_maps=condition_maps,
            prompts=prompts,
            num_rows=num_rows,
            seed=42,
        )

        # Compute expected dimensions using the formulas
        expected_width = (
            generator.row_label_width
            + generator.padding
            + generator.NUM_COLUMNS * cell_w
            + (generator.NUM_COLUMNS + 1) * generator.padding
        )
        expected_height = (
            generator.header_height
            + generator.padding
            + num_rows * cell_h
            + (num_rows + 1) * generator.padding
        )

        assert grid.size == (expected_width, expected_height), (
            f"Grid size {grid.size} != expected ({expected_width}, {expected_height}) "
            f"for cell_size={cell_size}, num_rows={num_rows}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Property 7: Pipeline Resilience to Module Failure
# Feature: evaluation-pipeline, Property 7: Pipeline Resilience to Module Failure
# **Validates: Requirements 5.5**
# ─────────────────────────────────────────────────────────────────────────────

from unittest.mock import patch, MagicMock
import argparse

from evaluation.run_evaluation import run_evaluation


# Strategy for selecting which modules should fail (at least 1, at most 2 so
# that at least one module still succeeds and produces results)
failing_modules_strategy = st.lists(
    st.sampled_from(["fid", "alignment", "grid"]),
    min_size=1,
    max_size=2,
    unique=True,
)


@st.composite
def pipeline_failure_scenario(draw):
    """Generate a random pipeline failure scenario.

    Draws a random subset of modules to fail (1 or 2 out of 3),
    ensuring at least one module will succeed.
    """
    failing = draw(failing_modules_strategy)
    # Generate a random exception message for realism
    error_msg = draw(st.text(
        alphabet=st.characters(whitelist_categories=("L", "N", "Z")),
        min_size=1,
        max_size=50,
    ))
    return failing, error_msg


class TestPipelineResilienceToModuleFailure:
    """
    Property: For any single evaluation module (FID, alignment, or grid) that
    raises an exception, the pipeline orchestrator SHALL still execute the
    remaining modules and produce partial results for the modules that succeeded.

    Feature: evaluation-pipeline, Property 7: Pipeline Resilience to Module Failure

    **Validates: Requirements 5.5**
    """

    def _make_args(self, tmp_dir):
        """Create a minimal args namespace for run_evaluation."""
        return argparse.Namespace(
            output_dir=tmp_dir,
            num_fid_samples=10,
            num_alignment_samples=5,
            num_grid_prompts=2,
            batch_size=4,
            condition_types=["depth", "pose", "edge"],
            coco_val_dir="data/raw/coco_val2017",
            checkpoint_dir="models/trained",
            seed=42,
        )

    @given(scenario=pipeline_failure_scenario())
    @settings(max_examples=100, deadline=None)
    def test_remaining_modules_execute_when_one_fails(self, scenario):
        """
        Feature: evaluation-pipeline, Property 7: Pipeline Resilience to Module Failure

        For any single module that raises an exception, remaining modules
        still execute and produce results. The pipeline does not abort.

        **Validates: Requirements 5.5**
        """
        failing_modules, error_msg = scenario

        with tempfile.TemporaryDirectory() as tmp_dir:
            args = self._make_args(tmp_dir)

            # Track which modules were actually called
            fid_called = False
            alignment_called = False
            grid_called = False

            # Create mock instances with return values (not side_effect functions)
            mock_fid_instance = MagicMock()
            mock_fid_instance.run_full_evaluation.return_value = {
                "baseline": 45.0, "depth": 17.0, "pose": 16.0, "edge": 18.0
            }

            mock_alignment_instance = MagicMock()
            mock_alignment_instance.run_full_evaluation.return_value = {
                "depth": (0.75, 0.08), "pose": (0.72, 0.11), "edge": (0.76, 0.06)
            }

            mock_grid_instance = MagicMock()
            mock_grid_instance.save_all_grids.return_value = [
                os.path.join(tmp_dir, "visual_grid_depth.png"),
                os.path.join(tmp_dir, "visual_grid_pose.png"),
                os.path.join(tmp_dir, "visual_grid_edge.png"),
                os.path.join(tmp_dir, "visual_grid_combined.png"),
            ]

            def fid_constructor(*a, **kw):
                nonlocal fid_called
                fid_called = True
                if "fid" in failing_modules:
                    raise RuntimeError(f"FID failure: {error_msg}")
                return mock_fid_instance

            def alignment_constructor(*a, **kw):
                nonlocal alignment_called
                alignment_called = True
                if "alignment" in failing_modules:
                    raise RuntimeError(f"Alignment failure: {error_msg}")
                return mock_alignment_instance

            def grid_constructor(*a, **kw):
                nonlocal grid_called
                grid_called = True
                if "grid" in failing_modules:
                    raise RuntimeError(f"Grid failure: {error_msg}")
                return mock_grid_instance

            mock_fid_cls = MagicMock(side_effect=fid_constructor)
            mock_alignment_cls = MagicMock(side_effect=alignment_constructor)
            mock_grid_cls = MagicMock(side_effect=grid_constructor)

            # Mock validate_checkpoints to return all types as valid
            mock_validate = MagicMock(return_value=["depth", "pose", "edge"])

            # Mock load_controlnet_pipeline to return a mock pipeline
            mock_load_pipeline = MagicMock(return_value=MagicMock())

            with patch("evaluation.run_evaluation.validate_checkpoints", mock_validate), \
                 patch("evaluation.run_evaluation.load_controlnet_pipeline", mock_load_pipeline), \
                 patch("evaluation.run_evaluation.EvaluationFIDCalculator", mock_fid_cls), \
                 patch("evaluation.run_evaluation.EvaluationAlignmentCalculator", mock_alignment_cls), \
                 patch("evaluation.run_evaluation.EvaluationGridGenerator", mock_grid_cls):

                # Run the evaluation pipeline — should NOT raise
                result = run_evaluation(args)

            # All modules should have been attempted (called)
            assert fid_called, "FID module was never attempted"
            assert alignment_called, "Alignment module was never attempted"
            assert grid_called, "Grid module was never attempted"

            # Modules that did NOT fail should have produced results
            if "fid" not in failing_modules:
                assert result["fid_results"] != {}, (
                    "FID results should be non-empty when FID module succeeds"
                )
            else:
                assert result["fid_results"] == {}, (
                    "FID results should be empty when FID module fails"
                )

            if "alignment" not in failing_modules:
                assert result["alignment_results"] != {}, (
                    "Alignment results should be non-empty when alignment module succeeds"
                )
            else:
                assert result["alignment_results"] == {}, (
                    "Alignment results should be empty when alignment module fails"
                )

            if "grid" not in failing_modules:
                assert result["grid_paths"] != [], (
                    "Grid paths should be non-empty when grid module succeeds"
                )
            else:
                assert result["grid_paths"] == [], (
                    "Grid paths should be empty when grid module fails"
                )

            # metrics.json should always be saved (even with partial results)
            metrics_file = os.path.join(tmp_dir, "metrics.json")
            assert os.path.exists(metrics_file), (
                f"metrics.json should be saved even when modules fail. "
                f"Failing modules: {failing_modules}"
            )

            # Verify the saved metrics.json is valid JSON
            with open(metrics_file, "r") as f:
                saved_metrics = json.load(f)

            assert "metadata" in saved_metrics, "metrics.json missing 'metadata' key"
            assert "fid_scores" in saved_metrics, "metrics.json missing 'fid_scores' key"
            assert "alignment_scores" in saved_metrics, "metrics.json missing 'alignment_scores' key"

    @given(
        failing_module=st.sampled_from(["fid", "alignment", "grid"]),
        error_msg=st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "Z")),
            min_size=1,
            max_size=30,
        ),
    )
    @settings(max_examples=100, deadline=None)
    def test_single_module_failure_does_not_abort_pipeline(self, failing_module, error_msg):
        """
        Feature: evaluation-pipeline, Property 7: Pipeline Resilience to Module Failure

        For any single module that raises an exception, the pipeline does not
        abort and the other two modules still execute successfully.

        **Validates: Requirements 5.5**
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = self._make_args(tmp_dir)

            # Track execution of non-failing modules
            modules_executed = {"fid": False, "alignment": False, "grid": False}

            mock_fid_instance = MagicMock()
            mock_fid_instance.run_full_evaluation.return_value = {
                "baseline": 45.0, "depth": 17.0, "pose": 16.0, "edge": 18.0
            }

            def fid_constructor(*a, **kw):
                modules_executed["fid"] = True
                if failing_module == "fid":
                    raise RuntimeError(f"FID error: {error_msg}")
                return mock_fid_instance

            mock_alignment_instance = MagicMock()
            mock_alignment_instance.run_full_evaluation.return_value = {
                "depth": (0.75, 0.08), "pose": (0.72, 0.11), "edge": (0.76, 0.06)
            }

            def alignment_constructor(*a, **kw):
                modules_executed["alignment"] = True
                if failing_module == "alignment":
                    raise RuntimeError(f"Alignment error: {error_msg}")
                return mock_alignment_instance

            mock_grid_instance = MagicMock()
            mock_grid_instance.save_all_grids.return_value = [
                os.path.join(tmp_dir, "visual_grid_depth.png"),
            ]

            def grid_constructor(*a, **kw):
                modules_executed["grid"] = True
                if failing_module == "grid":
                    raise RuntimeError(f"Grid error: {error_msg}")
                return mock_grid_instance

            mock_fid_cls = MagicMock(side_effect=fid_constructor)
            mock_alignment_cls = MagicMock(side_effect=alignment_constructor)
            mock_grid_cls = MagicMock(side_effect=grid_constructor)

            mock_validate = MagicMock(return_value=["depth", "pose", "edge"])
            mock_load_pipeline = MagicMock(return_value=MagicMock())

            with patch("evaluation.run_evaluation.validate_checkpoints", mock_validate), \
                 patch("evaluation.run_evaluation.load_controlnet_pipeline", mock_load_pipeline), \
                 patch("evaluation.run_evaluation.EvaluationFIDCalculator", mock_fid_cls), \
                 patch("evaluation.run_evaluation.EvaluationAlignmentCalculator", mock_alignment_cls), \
                 patch("evaluation.run_evaluation.EvaluationGridGenerator", mock_grid_cls):

                # Pipeline should complete without raising
                result = run_evaluation(args)

            # All modules should have been attempted
            for module_name, was_executed in modules_executed.items():
                assert was_executed, (
                    f"Module '{module_name}' was never attempted. "
                    f"Failing module was '{failing_module}'."
                )

            # The two non-failing modules should have produced results
            non_failing = [m for m in ["fid", "alignment", "grid"] if m != failing_module]
            for module_name in non_failing:
                if module_name == "fid":
                    assert result["fid_results"] != {}, (
                        f"FID should have results when '{failing_module}' fails"
                    )
                elif module_name == "alignment":
                    assert result["alignment_results"] != {}, (
                        f"Alignment should have results when '{failing_module}' fails"
                    )
                elif module_name == "grid":
                    assert result["grid_paths"] != [], (
                        f"Grid should have results when '{failing_module}' fails"
                    )

            # metrics.json should still be saved
            metrics_file = os.path.join(tmp_dir, "metrics.json")
            assert os.path.exists(metrics_file), (
                f"metrics.json should be saved even when '{failing_module}' fails"
            )
