# Requirements Document

## Introduction

This document specifies the requirements for the complete evaluation pipeline that produces resume-worthy metrics for the ControlNet training project. The pipeline evaluates the quality of 3 trained ControlNet adapters (depth, pose, edge) using industry-standard metrics: FID score computation, condition alignment measurement, and visual grid generation. All metrics are saved to a structured JSON file for reporting.

## Glossary

- **FID_Calculator**: The module (`evaluation/compute_fid.py`) that computes Fréchet Inception Distance between real and generated image distributions using Inception-v3 features
- **Condition_Alignment_Evaluator**: The module (`evaluation/condition_alignment.py`) that measures how well generated images follow their input conditioning signals
- **Visual_Grid_Generator**: The module (`evaluation/visual_grid.py`) that produces 4-column comparison grids for qualitative evaluation
- **Condition_Type**: One of three spatial conditioning modes: "depth" (MiDaS/DPT depth maps), "pose" (DWPose skeletons), or "edge" (Canny edge maps)
- **COCO_Validation_Set**: The COCO 2017 validation dataset used as the real image distribution for FID computation
- **Inception_V3_Features**: 2048-dimensional feature vectors extracted from the penultimate layer of a pretrained Inception-v3 network
- **SSIM**: Structural Similarity Index Measure, used to compare edge maps between generated images and input condition maps
- **DPT_Model**: Dense Prediction Transformer model used to extract depth maps from generated images for alignment comparison
- **DWPose_Model**: Pose estimation model used to detect keypoints in generated images for pose alignment comparison
- **Vanilla_SD1_5**: Stable Diffusion 1.5 without any ControlNet adapter, used as a baseline for FID comparison
- **Metrics_JSON**: The structured output file (`evaluation/results/metrics.json`) containing all computed evaluation scores
- **Inference_Pipeline**: The existing `model/pipeline.py` ControlNetPipeline used to generate images from trained adapters

## Requirements

### Requirement 1: FID Score Computation

**User Story:** As a researcher, I want to compute FID scores for each trained ControlNet adapter against the COCO validation set, so that I can quantify generation quality with an industry-standard metric.

#### Acceptance Criteria

1. WHEN the FID computation is invoked for a Condition_Type, THE FID_Calculator SHALL generate 1000 images using the trained ControlNet adapter for that Condition_Type via the Inference_Pipeline
2. THE FID_Calculator SHALL use the COCO 2017 validation set as the real image distribution, loading a minimum of 1000 randomly sampled images from the dataset
3. WHEN computing FID, THE FID_Calculator SHALL extract Inception_V3_Features (2048-dimensional vectors) from both the real and generated image sets using a pretrained Inception-v3 network
4. THE FID_Calculator SHALL compute and report FID scores separately for each Condition_Type: depth-FID, pose-FID, and edge-FID
5. THE FID_Calculator SHALL compute a baseline FID score using Vanilla_SD1_5 (without any ControlNet adapter) on the same prompts used for conditioned generation
6. WHEN FID computation completes, THE FID_Calculator SHALL print a formatted results table showing the model name and FID score for each row: SD1.5 baseline, Depth ControlNet, Pose ControlNet, and Edge ControlNet
7. IF a trained ControlNet checkpoint is not found for a given Condition_Type, THEN THE FID_Calculator SHALL skip that condition, log a warning message indicating the missing checkpoint path, and continue with remaining conditions
8. THE FID_Calculator SHALL use batch processing with a configurable batch size (default 32) to extract Inception-v3 features, ensuring peak GPU memory stays within T4 constraints (15GB VRAM)

### Requirement 2: Condition Alignment Measurement

**User Story:** As a researcher, I want to measure how well generated images follow their conditioning signals, so that I can verify the ControlNet adapters learned meaningful spatial control.

#### Acceptance Criteria

1. WHEN evaluating edge alignment, THE Condition_Alignment_Evaluator SHALL compute Canny edges on the generated image and calculate SSIM between the extracted edges and the input edge condition map
2. WHEN evaluating depth alignment, THE Condition_Alignment_Evaluator SHALL run a DPT_Model on the generated image and compute Pearson correlation between the extracted depth map and the input depth condition map
3. WHEN evaluating pose alignment, THE Condition_Alignment_Evaluator SHALL run a DWPose_Model on the generated image and compute mean keypoint distance (in pixels, normalized by image diagonal) between detected keypoints and input pose keypoints
4. THE Condition_Alignment_Evaluator SHALL evaluate alignment on a minimum of 100 image-condition pairs per Condition_Type
5. THE Condition_Alignment_Evaluator SHALL report a mean alignment score and standard deviation for each Condition_Type
6. THE Condition_Alignment_Evaluator SHALL print per-condition alignment scores in a formatted table showing condition type, mean score, standard deviation, and whether the target threshold of 0.70 is met
7. IF the alignment score for any Condition_Type falls below 0.70, THEN THE Condition_Alignment_Evaluator SHALL print a warning indicating the condition type and achieved score

### Requirement 3: Visual Comparison Grid Generation

**User Story:** As a researcher, I want to generate visual comparison grids showing original images, condition maps, conditioned generation, and unconditioned generation side-by-side, so that I can create demo images for README, GitHub, and HuggingFace.

#### Acceptance Criteria

1. THE Visual_Grid_Generator SHALL produce a 4-column grid for each Condition_Type where Column 1 is the original image, Column 2 is the condition map, Column 3 is the generated image following the condition, and Column 4 is the generated image from Vanilla_SD1_5 without any condition
2. THE Visual_Grid_Generator SHALL generate grids using 20 test prompts per Condition_Type, producing one row per prompt
3. WHEN a grid is generated, THE Visual_Grid_Generator SHALL save it as a lossless PNG file at high resolution (each cell at least 256x256 pixels) to the path `evaluation/results/visual_grid_{condition_type}.png`
4. THE Visual_Grid_Generator SHALL generate a combined grid showing all 3 Condition_Types applied to the same input images, saved to `evaluation/results/visual_grid_combined.png`
5. WHEN generating grid images, THE Visual_Grid_Generator SHALL include column headers labeling each column ("Original", "Condition Map", "With ControlNet", "Without ControlNet") and row labels showing the text prompt used
6. IF a trained ControlNet checkpoint is not available for a Condition_Type, THEN THE Visual_Grid_Generator SHALL skip that condition type and log a warning

### Requirement 4: Metrics Output and Persistence

**User Story:** As a researcher, I want all evaluation metrics saved to a structured JSON file, so that I can reference exact numbers in resume bullet points and publications.

#### Acceptance Criteria

1. WHEN all evaluations complete, THE system SHALL save a Metrics_JSON file to `evaluation/results/metrics.json` containing FID scores, alignment scores, and metadata
2. THE Metrics_JSON SHALL contain for each Condition_Type: the FID score, the mean alignment score, the alignment standard deviation, and the number of samples evaluated
3. THE Metrics_JSON SHALL contain the baseline Vanilla_SD1_5 FID score for comparison
4. THE Metrics_JSON SHALL contain metadata including: evaluation timestamp, number of generated images, COCO validation set size used, model checkpoint paths, and inference configuration (guidance scale, number of steps)
5. THE Metrics_JSON SHALL be formatted with 2-space indentation for human readability
6. IF the output directory `evaluation/results/` does not exist, THEN THE system SHALL create it before writing any output files

### Requirement 5: Pipeline Orchestration

**User Story:** As a researcher, I want a single entry point that runs the complete evaluation pipeline end-to-end, so that I can produce all metrics and visualizations with one command.

#### Acceptance Criteria

1. THE system SHALL provide a main entry point script (`evaluation/run_evaluation.py`) that orchestrates FID computation, condition alignment, and visual grid generation in sequence
2. WHEN the pipeline starts, THE system SHALL validate that required model checkpoints exist for at least one Condition_Type before proceeding with evaluation
3. THE system SHALL accept command-line arguments for: output directory (default `evaluation/results/`), number of FID samples (default 1000), number of alignment samples (default 100), number of grid prompts (default 20), batch size (default 32), and which condition types to evaluate (default all three)
4. WHEN the pipeline completes, THE system SHALL print a summary showing total evaluation time, FID scores achieved, alignment scores achieved, and paths to generated output files
5. IF any evaluation module fails with an exception, THEN THE system SHALL catch the error, log it, and continue with remaining evaluation modules rather than aborting the entire pipeline
6. THE system SHALL use the existing Inference_Pipeline from `model/pipeline.py` for all image generation, loading trained ControlNet checkpoints from the configured model directory
