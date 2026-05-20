# Implementation Plan: Evaluation Pipeline

## Overview

This plan implements the complete evaluation pipeline for the 3 trained ControlNet adapters. The implementation covers `evaluation/compute_fid.py`, `evaluation/condition_alignment.py`, `evaluation/visual_grid.py`, and `evaluation/run_evaluation.py`, building on the existing `model/pipeline.py` ControlNetPipeline and the existing evaluation modules in `src/evaluation/`.

## Tasks

- [x] 1. Set up evaluation module structure and shared utilities
  - [x] 1.1 Create `evaluation/__init__.py` and `evaluation/config.py` with EvaluationConfig dataclass
    - Define EvaluationConfig with all configurable parameters (output_dir, num_fid_samples, num_alignment_samples, num_grid_prompts, batch_size, condition_types, coco_val_dir, checkpoint_dir, guidance_scale, num_inference_steps, seed)
    - Create helper function `ensure_output_dir()` that creates `evaluation/results/` if it doesn't exist
    - Create helper function `load_test_prompts()` that returns a list of 20 diverse test prompts for evaluation
    - _Requirements: 4.6, 5.3_

  - [x] 1.2 Create `evaluation/pipeline_loader.py` with checkpoint loading utilities
    - Implement `load_controlnet_pipeline(condition_type, checkpoint_dir, device)` that loads the ControlNetPipeline with a trained adapter
    - Implement `load_baseline_pipeline(device)` that loads vanilla SD1.5 without ControlNet for baseline generation
    - Implement `validate_checkpoints(checkpoint_dir, condition_types)` that checks which condition types have available checkpoints and returns the valid subset
    - Return None with a logged warning if checkpoint not found for a condition type
    - _Requirements: 5.2, 5.6, 1.7_

- [x] 2. Implement FID Score Computation (`evaluation/compute_fid.py`)
  - [x] 2.1 Implement EvaluationFIDCalculator class with image generation
    - Create class that accepts a ControlNetPipeline, COCO validation directory path, batch_size, and device
    - Implement `generate_images(prompts, condition_maps, condition_type, num_images, seed)` that generates images using the pipeline with fixed seeds for reproducibility
    - Implement `generate_baseline_images(prompts, num_images, seed)` that generates images using vanilla SD1.5 (no ControlNet conditioning)
    - Implement `load_coco_images(num_images)` that randomly samples images from the COCO 2017 validation set
    - _Requirements: 1.1, 1.2, 1.5_

  - [x] 2.2 Implement FID computation using pytorch-fid or existing FIDCalculator
    - Leverage the existing `src/evaluation/compute_fid.py` FIDCalculator class for Inception-v3 feature extraction and Fréchet distance computation
    - Implement `compute_fid_for_condition(condition_type, prompts, condition_maps, num_images)` that generates images and computes FID against COCO
    - Use batch processing (configurable batch_size, default 32) for feature extraction to stay within T4 memory
    - _Requirements: 1.3, 1.4, 1.8_

  - [x] 2.3 Implement full FID evaluation and results table
    - Implement `run_full_evaluation(prompts, condition_maps, num_images)` that computes FID for all condition types + baseline
    - Implement `print_results_table(results)` that prints the formatted table: Model | FID Score with rows for SD1.5 baseline, Depth ControlNet, Pose ControlNet, Edge ControlNet
    - Handle missing checkpoints gracefully — skip condition type and log warning
    - _Requirements: 1.4, 1.5, 1.6, 1.7_

  - [x]* 2.4 Write property test for Inception-v3 feature shape invariant
    - **Property 1: Inception-v3 Feature Shape Invariant**
    - For any batch of random RGB images (batch size 1-50), feature extraction always produces shape (N, 2048)
    - Use Hypothesis to generate random image batches
    - **Validates: Requirements 1.3**

- [x] 3. Implement Condition Alignment (`evaluation/condition_alignment.py`)
  - [x] 3.1 Implement EvaluationAlignmentCalculator with edge alignment
    - Create class that accepts a ControlNetPipeline and device
    - Implement `compute_edge_alignment(generated_image, input_edge_map)` that applies Canny edge detection to the generated image and computes SSIM against the input edge map
    - Ensure output is clamped to [0, 1] range
    - _Requirements: 2.1_

  - [x] 3.2 Implement depth alignment using DPT model
    - Implement `compute_depth_alignment(generated_image, input_depth_map)` that runs a DPT model (or lightweight proxy) on the generated image and computes Pearson correlation with the input depth map
    - Use the existing `src/evaluation/condition_alignment.py` DepthAlignmentEvaluator as reference or directly
    - Normalize correlation output to [0, 1] by taking max(0, correlation)
    - _Requirements: 2.2_

  - [x] 3.3 Implement pose alignment using keypoint detection
    - Implement `compute_pose_alignment(generated_image, input_pose_map)` that detects keypoints in the generated image and computes normalized keypoint distance against the input pose
    - Normalize distance by image diagonal, return 1 - normalized_distance so higher = better
    - Ensure output is clamped to [0, 1]
    - _Requirements: 2.3_

  - [x] 3.4 Implement batch evaluation and reporting
    - Implement `evaluate_condition_type(condition_type, prompts, condition_maps, num_samples)` that generates images and computes alignment for each pair, returning (mean, std)
    - Implement `run_full_evaluation(prompts, condition_maps, num_samples)` for all condition types
    - Implement `print_results_table(results)` showing condition type, mean score, std, and whether target 0.70 is met
    - Print warning if any score < 0.70
    - _Requirements: 2.4, 2.5, 2.6, 2.7_

  - [x]* 3.5 Write property test for alignment score bounded range
    - **Property 2: Alignment Score Bounded Range**
    - For any condition type and any pair of random images, alignment score is always in [0, 1]
    - Use Hypothesis to generate random image arrays
    - **Validates: Requirements 2.1, 2.2, 2.3**

  - [x]* 3.6 Write property test for aggregation statistics invariant
    - **Property 3: Aggregation Statistics Invariant**
    - For any non-empty list of scores in [0, 1], mean is between min and max, std is non-negative
    - Use Hypothesis to generate random float lists
    - **Validates: Requirements 2.5**

- [x] 4. Implement Visual Grid Generation (`evaluation/visual_grid.py`)
  - [x] 4.1 Implement EvaluationGridGenerator class
    - Create class that accepts a ControlNetPipeline, cell_size (default 256x256), and output_dir
    - Implement `generate_grid(condition_type, original_images, condition_maps, prompts, num_rows, seed)` that creates a 4-column grid: Original | Condition Map | With ControlNet | Without ControlNet
    - Include column headers ("Original", "Condition Map", "With ControlNet", "Without ControlNet") and row labels (text prompts, truncated to fit)
    - _Requirements: 3.1, 3.2, 3.5_

  - [x] 4.2 Implement combined grid and file saving
    - Implement `generate_combined_grid(original_images, condition_maps, prompts, num_rows, seed)` showing all 3 condition types on the same input images
    - Implement `save_all_grids(original_images, condition_maps, prompts)` that generates and saves all grids as lossless PNGs to `evaluation/results/visual_grid_{type}.png` and `visual_grid_combined.png`
    - Handle missing checkpoints by skipping that condition type with a warning
    - _Requirements: 3.3, 3.4, 3.6_

  - [x]* 4.3 Write property test for grid dimension invariant
    - **Property 4: Grid Dimension Invariant**
    - For any number of rows (1-20) and cell size, output grid dimensions match the expected formula
    - Use Hypothesis to generate random configurations
    - **Validates: Requirements 3.1**

- [x] 5. Implement Metrics Output (`evaluation/metrics_writer.py`)
  - [x] 5.1 Implement metrics JSON writer
    - Create `build_metrics_dict(fid_results, alignment_results, grid_paths, config)` that assembles the full metrics dictionary with metadata (timestamp, sample counts, checkpoint paths, inference config)
    - Create `save_metrics_json(metrics_dict, output_dir)` that writes to `evaluation/results/metrics.json` with 2-space indentation
    - Ensure output directory is created if it doesn't exist
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

  - [x]* 5.2 Write property tests for metrics JSON
    - **Property 5: Metrics JSON Round-Trip**
    - For any valid metrics dict, JSON serialize then deserialize produces equivalent data
    - **Property 6: Metrics JSON Schema Completeness**
    - For any combination of condition types and scores, output JSON contains all required keys
    - Use Hypothesis to generate random metrics dictionaries
    - **Validates: Requirements 4.1, 4.2, 4.3**

- [x] 6. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implement Pipeline Orchestrator (`evaluation/run_evaluation.py`)
  - [x] 7.1 Implement CLI argument parsing and main entry point
    - Create `parse_args()` with arguments: --output_dir, --num_fid_samples, --num_alignment_samples, --num_grid_prompts, --batch_size, --condition_types, --coco_val_dir, --checkpoint_dir, --seed
    - Set defaults matching EvaluationConfig
    - _Requirements: 5.3_

  - [x] 7.2 Implement run_evaluation orchestrator function
    - Implement `run_evaluation(args)` that:
      1. Validates checkpoints exist for at least one condition type
      2. Loads test data (prompts, original images, condition maps)
      3. Runs FID computation (wrapped in try/except)
      4. Runs condition alignment (wrapped in try/except)
      5. Generates visual grids (wrapped in try/except)
      6. Saves metrics.json with whatever results succeeded
      7. Prints summary with total time, FID scores, alignment scores, and output paths
    - Each module failure is caught, logged, and the pipeline continues
    - _Requirements: 5.1, 5.2, 5.4, 5.5, 5.6_

  - [x]* 7.3 Write property test for pipeline resilience
    - **Property 7: Pipeline Resilience to Module Failure**
    - For any single module that raises an exception, remaining modules still execute
    - Mock individual modules to raise exceptions, verify others complete
    - **Validates: Requirements 5.5**

- [x] 8. Final Integration and Verification
  - [x] 8.1 End-to-end integration test with mock pipeline
    - Test full evaluation flow with a mock ControlNetPipeline that returns random 512x512 images
    - Verify metrics.json is created with valid structure
    - Verify PNG grid files are created at expected paths
    - Verify graceful handling when some condition types have no checkpoints
    - _Requirements: 1.4, 2.5, 3.3, 4.1, 5.1_

  - [x] 8.2 Verify output file structure
    - Confirm `evaluation/results/metrics.json` contains all required fields
    - Confirm `evaluation/results/visual_grid_depth.png`, `visual_grid_pose.png`, `visual_grid_edge.png`, `visual_grid_combined.png` are valid PNG files
    - Confirm results table prints correctly to stdout
    - _Requirements: 3.3, 4.1, 4.2, 1.6, 2.6_

- [x] 9. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional property-based tests that can be skipped for faster MVP
- The pipeline reuses the existing `src/evaluation/compute_fid.py` FIDCalculator and `src/evaluation/condition_alignment.py` evaluators where possible
- All image generation goes through the existing `model/pipeline.py` ControlNetPipeline
- The evaluation modules are designed to work independently — each can be run standalone or via the orchestrator
- Property tests use Hypothesis and run minimum 100 iterations each

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["2.1", "3.1", "4.1"] },
    { "id": 2, "tasks": ["2.2", "3.2", "3.3", "4.2"] },
    { "id": 3, "tasks": ["2.3", "3.4", "5.1"] },
    { "id": 4, "tasks": ["2.4", "3.5", "3.6", "4.3", "5.2"] },
    { "id": 5, "tasks": ["7.1", "7.2"] },
    { "id": 6, "tasks": ["7.3", "8.1", "8.2"] }
  ]
}
```
