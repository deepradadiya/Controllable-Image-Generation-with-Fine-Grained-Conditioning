# Requirements Document

## Introduction

This feature enhances the existing Controllable Image Generation project with three major deliverables: (1) an enhanced Gradio demo application with 3-panel layout, preset examples, and multi-condition comparison; (2) a comprehensive publish_to_hub.py script that pushes all trained adapters, a combined pipeline, and a HuggingFace Space with rich model cards; and (3) a polished final README.md for GitHub showcasing architecture diagrams, results tables, visual grids, and reproducibility instructions.

## Glossary

- **Gradio_App**: The enhanced Gradio web interface at `src/app/gradio_app.py` providing interactive ControlNet image generation with 3-panel display
- **Condition_Map**: A spatial control signal (depth map, pose skeleton, or edge map) extracted from a source image
- **Condition_Type**: One of three supported spatial conditioning modes: "Depth Map" (DPT), "Pose Skeleton" (DWPose), or "Edge Map" (Canny)
- **Three_Panel_Display**: A side-by-side layout showing Input Image, Condition Map, and Generated Result
- **Publish_Script**: The `publish_to_hub.py` script responsible for uploading models, deploying Spaces, and generating model cards
- **Model_Card**: A README.md file within a HuggingFace model repository describing the model's purpose, training details, metrics, and usage
- **HuggingFace_Hub**: The HuggingFace model hosting platform where trained adapters are published
- **HuggingFace_Space**: A HuggingFace-hosted Gradio application for interactive model demos
- **Combined_Pipeline**: A multi-ControlNet pipeline published as a single repository supporting all three condition types
- **Visual_Grid**: An image grid showing generated results across condition types for visual comparison
- **FID_Score**: Fréchet Inception Distance metric measuring quality of generated images against real images
- **Alignment_Score**: A metric measuring how well generated images follow the provided condition map
- **Guidance_Scale**: A classifier-free guidance parameter controlling prompt adherence (range 1.0–15.0)
- **Preset_Example**: A pre-configured combination of source image, condition type, and text prompt for quick demonstration
- **Multi_Condition_Button**: A UI button that runs all three condition adapters on the same input and displays comparison results
- **README_Document**: The final GitHub README.md file for the project repository

## Requirements

### Requirement 1: Three-Panel Image Display

**User Story:** As a demo user, I want to see the input image, extracted condition map, and generated result side by side, so that I can visually compare the conditioning pipeline end-to-end.

#### Acceptance Criteria

1. WHEN a user uploads an image and triggers generation, THE Gradio_App SHALL display three image panels in a single row: Input Image, Condition Map, and Generated Result
2. THE Gradio_App SHALL render each panel at equal width within the display row
3. THE Gradio_App SHALL label each panel with its respective title ("Input Image", "Condition Map", "Generated Result")
4. WHEN generation fails or has not been triggered, THE Gradio_App SHALL display the Input Image panel with the uploaded image, the Condition Map panel with the extracted condition map (if available), and the Generated Result panel as empty with a placeholder message
5. IF generation fails, THEN THE Gradio_App SHALL display an error message in the Status textbox and leave the Generated Result panel empty

### Requirement 2: Condition Type Selection

**User Story:** As a demo user, I want to select a condition type from a dropdown, so that I can choose which spatial conditioning method to apply.

#### Acceptance Criteria

1. THE Gradio_App SHALL provide a dropdown with exactly three options in this order: "Depth Map", "Pose Skeleton", "Edge Map"
2. WHEN a user selects a condition type and an image is already uploaded, THE Gradio_App SHALL invoke the corresponding extractor (DPT for Depth Map, DWPose for Pose Skeleton, Canny for Edge Map) and display the resulting condition map in the Condition Map panel
3. IF a user selects a condition type when no image has been uploaded, THEN THE Gradio_App SHALL store the selection without invoking any extractor and apply it when an image is subsequently uploaded
4. THE Gradio_App SHALL default the dropdown selection to "Depth Map" on initial load

### Requirement 3: Guidance Scale Slider

**User Story:** As a demo user, I want to adjust the guidance scale with a slider, so that I can control how closely the generation follows my text prompt.

#### Acceptance Criteria

1. THE Gradio_App SHALL provide a slider labeled "Guidance Scale" for guidance_scale with minimum value 1.0, maximum value 15.0, step increment of 0.5, and default value 7.5, displaying the current numeric value to the user
2. WHEN a user adjusts the guidance_scale slider, THE Gradio_App SHALL pass the selected value to the generation pipeline as the classifier-free guidance parameter
3. IF the user does not adjust the guidance_scale slider before triggering generation, THEN THE Gradio_App SHALL pass the default value of 7.5 to the generation pipeline as the classifier-free guidance parameter

### Requirement 4: Automatic Condition Map Preview

**User Story:** As a demo user, I want to see the condition map extracted from my uploaded image before generation starts, so that I can verify the conditioning signal looks correct.

#### Acceptance Criteria

1. WHEN a user uploads an image and a condition type is selected, THE Gradio_App SHALL automatically extract and display the condition map in the Condition Map panel without requiring the user to click Generate
2. WHEN a user changes the condition type after uploading an image, THE Gradio_App SHALL re-extract and update the condition map preview using the newly selected extractor
3. IF condition map extraction fails, THEN THE Gradio_App SHALL display an error message in the Status textbox indicating the failure reason and leave the Condition Map panel empty
4. WHEN a user removes the uploaded image while a condition map is displayed, THE Gradio_App SHALL clear the Condition Map panel and display a status message indicating no image is uploaded
5. WHILE condition map extraction is in progress, THE Gradio_App SHALL indicate to the user that extraction is processing by displaying a loading state on the Condition Map panel

### Requirement 5: Preset Example Combinations

**User Story:** As a demo user, I want to click on preset examples to quickly see the demo in action, so that I can understand the system's capabilities without preparing my own inputs.

#### Acceptance Criteria

1. THE Gradio_App SHALL provide exactly 6 preset example combinations, each consisting of a source image, a condition type, and a text prompt with the text prompt not exceeding 200 characters
2. THE Gradio_App SHALL include at least one preset example for each of the three condition types (Depth Map, Pose Skeleton, Edge Map)
3. THE Gradio_App SHALL display preset examples in a visible examples section below the input controls, showing the source image thumbnail, condition type, and text prompt for each preset
4. WHEN a user clicks a preset example, THE Gradio_App SHALL populate the input image, condition type dropdown, and text prompt fields with the preset values without automatically triggering image generation
5. IF a preset example's source image file is unavailable or fails to load, THEN THE Gradio_App SHALL display an error message indicating which preset is unavailable and keep the input fields unchanged

### Requirement 6: Multi-Condition Comparison Button

**User Story:** As a demo user, I want to generate results using all three condition types at once, so that I can compare how different conditioning methods affect the output.

#### Acceptance Criteria

1. THE Gradio_App SHALL provide a button labeled "Generate with all 3 conditions"
2. WHEN a user clicks the multi-condition button and an image is uploaded and a text prompt is provided, THE Gradio_App SHALL extract condition maps using all three extractors (DPT, DWPose, Canny) from the uploaded image
3. WHEN a user clicks the multi-condition button and an image is uploaded and a text prompt is provided, THE Gradio_App SHALL generate an image for each condition type using the same text prompt and guidance_scale
4. THE Gradio_App SHALL display the multi-condition results in a 3-column by 2-row grid where each column represents one condition type, the top row shows the extracted condition maps, and the bottom row shows the corresponding generated images
5. IF a user clicks the multi-condition button without uploading an image or without providing a text prompt, THEN THE Gradio_App SHALL display an error message indicating which required input is missing and SHALL NOT proceed with extraction or generation
6. IF extraction or generation fails for one condition type during multi-condition comparison, THEN THE Gradio_App SHALL display an error message in place of the failed result and SHALL continue processing the remaining condition types

### Requirement 7: Publish Individual ControlNet Adapters

**User Story:** As a project maintainer, I want to push each trained ControlNet adapter to its own HuggingFace Hub repository, so that users can download and use individual adapters independently.

#### Acceptance Criteria

1. WHEN the Publish_Script is executed, THE Publish_Script SHALL upload the depth adapter weights in safetensors format along with the adapter configuration file to the repository "deepradadiya/controlnet-sd15-depth"
2. WHEN the Publish_Script is executed, THE Publish_Script SHALL upload the pose adapter weights in safetensors format along with the adapter configuration file to the repository "deepradadiya/controlnet-sd15-pose"
3. WHEN the Publish_Script is executed, THE Publish_Script SHALL upload the edge adapter weights in safetensors format along with the adapter configuration file to the repository "deepradadiya/controlnet-sd15-edge"
4. THE Publish_Script SHALL authenticate with HuggingFace Hub using an access token provided via the HF_TOKEN environment variable or a --token command-line argument
5. IF a repository does not exist, THEN THE Publish_Script SHALL create it as a public repository before uploading
6. IF authentication fails or no valid token is provided, THEN THE Publish_Script SHALL exit with a non-zero status code and print an error message indicating the authentication failure reason
7. WHEN an adapter upload completes successfully, THE Publish_Script SHALL verify the upload by confirming the repository contains the expected safetensors weight file and log a success message including the repository URL

### Requirement 8: Publish Combined Multi-Condition Pipeline

**User Story:** As a project maintainer, I want to publish a combined pipeline supporting all three condition types, so that users can access all adapters from a single repository.

#### Acceptance Criteria

1. WHEN the Publish_Script is executed, THE Publish_Script SHALL upload a combined pipeline to the repository "deepradadiya/controlnet-sd15-multi", creating the repository if it does not already exist
2. THE Publish_Script SHALL include all three adapter weight files (depth, pose, edge) in safetensors format within the combined repository, each stored in a separate subdirectory named by condition type ("depth/", "pose/", "edge/")
3. THE Publish_Script SHALL include a JSON configuration file in the repository root that maps each condition type name ("depth", "pose", "edge") to its corresponding adapter weight file path within the repository
4. IF one or more adapter weight files are not found locally, THEN THE Publish_Script SHALL abort the upload and report an error message indicating which adapter weight files are missing

### Requirement 9: Deploy Gradio Space

**User Story:** As a project maintainer, I want to deploy the Gradio demo as a HuggingFace Space, so that anyone can try the model without local setup.

#### Acceptance Criteria

1. WHEN the Publish_Script is executed with the `--deploy-space` flag, THE Publish_Script SHALL create or update the HuggingFace Space "deepradadiya/controlnet-demo" with SDK type set to "gradio" and visibility set to public
2. THE Publish_Script SHALL upload the Gradio application code file and a `requirements.txt` file to the Space repository
3. THE Publish_Script SHALL configure the Space hardware setting to "t4" in the Space metadata
4. IF Space creation or update fails due to authentication error, network error, or quota limitation, THEN THE Publish_Script SHALL exit with a non-zero status code and print an error message indicating the failure reason

### Requirement 10: Model Card Generation

**User Story:** As a project maintainer, I want each published adapter to have a detailed model card, so that users understand what the model does, how it was trained, and how to use it.

#### Acceptance Criteria

1. THE Publish_Script SHALL generate a model card for each adapter containing: the base model identifier ("runwayml/stable-diffusion-v1-5"), the condition type it accepts (depth, pose, or edge), training details (dataset name, number of training steps completed, and hardware used), FID score, and alignment score
2. THE Publish_Script SHALL include a code snippet of 5 lines or fewer in each model card demonstrating how to load and use the adapter
3. THE Publish_Script SHALL embed a visual grid image in each model card showing at least 3 example generations for the adapter's condition type
4. THE Publish_Script SHALL format each model card as a valid HuggingFace README.md with YAML front matter containing at minimum: license, tags, and base_model fields
5. IF FID score or alignment score is unavailable at publish time, THEN THE Publish_Script SHALL generate the model card without the missing metric and include a note indicating the metric has not been computed
6. IF the visual grid image is unavailable at publish time, THEN THE Publish_Script SHALL generate the model card without the embedded image and include a placeholder indicating that example generations are not yet available

### Requirement 11: README Architecture Diagram

**User Story:** As a GitHub visitor, I want to see an architecture diagram in the README, so that I can quickly understand how the ControlNet pipeline works.

#### Acceptance Criteria

1. THE README_Document SHALL include an ASCII architecture diagram showing the inference data flow with the following labeled components: Text Encoder, Stable Diffusion 1.5 UNet (frozen), ControlNet copy (trainable), Zero Convolution layers, Noise Scheduler (DDIM), Condition Map input, Text Prompt input, and Generated Image output
2. THE README_Document SHALL include an ASCII training pipeline diagram showing the stages: Raw Image → Condition Extractor (depth/pose/edge) → Condition Map → ControlNet (trainable) + SD1.5 UNet (frozen) → Diffusion Loss, with frozen and trainable components clearly labeled
3. THE README_Document SHALL render each architecture diagram inside a markdown fenced code block with lines no longer than 80 characters to preserve formatting without horizontal scrolling on GitHub
4. THE README_Document SHALL place the architecture diagrams in a dedicated section positioned between the project description and the results section

### Requirement 12: README Results Table

**User Story:** As a GitHub visitor, I want to see quantitative results in the README, so that I can evaluate the model's performance compared to baseline.

#### Acceptance Criteria

1. THE README_Document SHALL include a results table with columns for: Model/Condition Type, FID Score, and Alignment Score, containing one row for each of the three condition types (depth, pose, edge) and one row for the Vanilla SD1.5 baseline
2. THE README_Document SHALL include the Vanilla SD1.5 (without ControlNet) FID score as the baseline comparison row in the results table, enabling readers to see the improvement provided by each ControlNet adapter
3. THE README_Document SHALL include the mean alignment score for each condition type in the results table, with a parenthetical note indicating the metric used (SSIM for edge, Pearson correlation for depth, normalized keypoint distance for pose)
4. THE README_Document SHALL indicate the direction of each metric in the table header or a footnote (lower is better for FID, higher is better for alignment scores)

### Requirement 13: README Visual Grid

**User Story:** As a GitHub visitor, I want to see visual examples of generated images in the README, so that I can qualitatively assess the model's output.

#### Acceptance Criteria

1. THE README_Document SHALL include an embedded 4-column visual grid showing at least 2 example rows per condition type (depth, pose, edge) for a minimum of 6 total rows
2. THE README_Document SHALL display columns labeled: "Input Image", "Condition Map", "Generated Result", and "Without ControlNet" (baseline generation without conditioning)
3. THE README_Document SHALL visually separate or label each group of rows by its condition type so that the viewer can identify which conditioning method produced each set of results

### Requirement 14: README Reproducibility Section

**User Story:** As a researcher, I want step-by-step Colab reproduction instructions in the README, so that I can replicate the training and evaluation pipeline.

#### Acceptance Criteria

1. THE README_Document SHALL include a "How to reproduce in Colab" section with sequential shell commands covering: cloning the repository, installing dependencies, downloading and preparing the dataset, extracting condition maps, training each adapter, and running evaluation
2. THE README_Document SHALL specify the expected hardware (Google Colab T4 GPU with 15GB VRAM) and estimated training time per condition type
3. THE README_Document SHALL include expected output verification steps so that users can confirm each stage completed successfully

### Requirement 15: README Links and Tech Stack

**User Story:** As a GitHub visitor, I want quick links to the HuggingFace models, demo Space, and training logs, so that I can access all project artifacts from one place.

#### Acceptance Criteria

1. THE README_Document SHALL include a dedicated links section containing clickable URLs to all three HuggingFace model repositories (deepradadiya/controlnet-sd15-depth, deepradadiya/controlnet-sd15-pose, deepradadiya/controlnet-sd15-edge)
2. THE README_Document SHALL include a clickable URL to the HuggingFace Space demo (deepradadiya/controlnet-demo) in the links section
3. THE README_Document SHALL include a clickable URL to the Weights & Biases training logs project in the links section
4. THE README_Document SHALL include a tech stack section listing at least 5 frameworks and libraries directly used in the training and inference pipelines (including at minimum the deep learning framework, diffusion library, and web demo framework)
5. THE README_Document SHALL include a resume bullet point suggestion that references at least one quantitative metric (FID score or alignment score) achieved by the trained models

### Requirement 16: README Project Header

**User Story:** As a GitHub visitor, I want a clear project title and one-line description at the top of the README, so that I immediately understand what this project does.

#### Acceptance Criteria

1. THE README_Document SHALL begin with a Markdown level-1 heading (`#`) containing the project title "Controllable Image Generation with Fine-Grained Conditioning" as the first line of the file
2. THE README_Document SHALL include a single-sentence plain-text description on the first non-empty line after the title heading, with no intervening headings, badges, or other elements between the title and the description
3. THE README_Document description line SHALL be at most 200 characters in length and SHALL state the project's capability (what the system does) and its primary conditioning approach
