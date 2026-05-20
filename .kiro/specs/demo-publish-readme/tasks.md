# Implementation Plan: Demo, Publish & README

## Overview

This plan implements three interconnected components: (1) an enhanced Gradio demo app with 3-panel layout, presets, and multi-condition comparison; (2) a comprehensive publishing script for HuggingFace Hub with model cards; and (3) a README builder generating the final project README with architecture diagrams, results tables, and reproducibility instructions. Implementation uses Python with existing project dependencies (gradio, huggingface-hub, hypothesis).

## Tasks

- [x] 1. Set up data models and shared interfaces
  - [x] 1.1 Create enhanced data models module
    - Create `src/app/models.py` with `EnhancedGenerationParams`, `PresetExample`, `MultiConditionResult` dataclasses
    - Define `PublishConfig`, `ModelCardMetadata` dataclasses for the publishing pipeline
    - Include type annotations and validation logic (guidance_scale range 1.0–15.0, prompt max 200 chars for presets)
    - _Requirements: 3.1, 5.1, 8.3_

  - [x]* 1.2 Write property test for guidance scale value passthrough
    - **Property 1: Guidance Scale Value Passthrough**
    - Use hypothesis to generate random floats in [1.0, 15.0] with step 0.5, verify the value passes through unchanged
    - **Validates: Requirements 3.2**

- [x] 2. Implement enhanced Gradio app with 3-panel layout
  - [x] 2.1 Refactor gradio_app.py to 3-panel display
    - Replace the existing 2-column layout with a 3-panel row: Input Image | Condition Map | Generated Result
    - Label each panel with its title ("Input Image", "Condition Map", "Generated Result")
    - Add guidance scale slider (min 1.0, max 15.0, step 0.5, default 7.5)
    - Update `_generate_image` to return 3-panel output (input, condition, generated)
    - Ensure condition map auto-extraction on upload and type change still works
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 4.4, 4.5_

  - [x] 2.2 Add preset examples component
    - Create 6 preset example combinations (at least 1 per condition type: depth, pose, edge)
    - Add `gr.Examples` component below input controls showing source image thumbnail, condition type, and prompt
    - Ensure clicking a preset populates inputs without triggering generation
    - Handle missing preset image files with error message
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [x] 2.3 Implement multi-condition comparison button and grid
    - Add "Generate with all 3 conditions" button
    - Implement `_generate_all_conditions()` function that runs all 3 extractors + generators
    - Display results in a 3-column × 2-row grid (top: condition maps, bottom: generated images)
    - Handle partial failures: continue processing remaining types, show error for failed type
    - Validate that image and prompt are provided before proceeding
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_

  - [x]* 2.4 Write property test for multi-condition partial failure resilience
    - **Property 2: Multi-Condition Partial Failure Resilience**
    - Use hypothesis to randomly fail 1 of 3 condition types, verify remaining types still produce results
    - **Validates: Requirements 6.6**

- [x] 3. Checkpoint - Verify Gradio app
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Implement publishing script core infrastructure
  - [x] 4.1 Create publish_to_hub.py with CLI and authentication
    - Create `scripts/publish_to_hub.py` with argparse CLI (--model-dir, --metrics-dir, --visual-grid-dir, --deploy-space, --token)
    - Implement authentication flow: CLI token → HF_TOKEN env var → cached login → error
    - Define `PublishConfig` initialization from CLI args
    - _Requirements: 7.4, 7.6_

  - [x] 4.2 Implement AdapterPublisher class
    - Implement `publish_adapter(condition_type)` to upload safetensors weights + config.json to individual repos
    - Implement `_create_repo_if_needed(repo_id)` for public repo creation
    - Implement `verify_upload(repo_id)` to confirm safetensors file exists in repo
    - Handle missing weight files with clear error messages
    - Target repos: deepradadiya/controlnet-sd15-{depth,pose,edge}
    - _Requirements: 7.1, 7.2, 7.3, 7.5, 7.7_

  - [x] 4.3 Implement CombinedPipelinePublisher class
    - Implement `publish_combined()` to upload all 3 adapters to deepradadiya/controlnet-sd15-multi
    - Implement `_organize_weights()` to place each adapter in subdirectory (depth/, pose/, edge/)
    - Implement `_build_config_json()` to generate JSON mapping condition types to weight paths
    - Abort with error listing missing files if any adapter weights are absent
    - _Requirements: 8.1, 8.2, 8.3, 8.4_

  - [x]* 4.4 Write property test for combined pipeline directory organization
    - **Property 3: Combined Pipeline Directory Organization**
    - Use hypothesis to generate random sets of adapter files (1–3 types present), verify directory structure and JSON config correctness
    - **Validates: Requirements 8.2, 8.3, 8.4**

- [x] 5. Implement model card generation and Space deployment
  - [x] 5.1 Implement ModelCardGenerator class
    - Implement `generate_card(condition_type, metrics, visual_grid_path)` producing valid HuggingFace README.md
    - Implement `_build_yaml_frontmatter()` with license, tags, base_model fields
    - Implement `_build_usage_snippet(repo_id)` with ≤5 line code example
    - Implement `_build_metrics_section()` with FID and alignment scores (or "not yet computed" notes)
    - Handle missing visual grid with placeholder text
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6_

  - [x]* 5.2 Write property test for model card content completeness
    - **Property 4: Model Card Content Completeness**
    - Use hypothesis to generate random ModelCardMetadata with optional fields, verify YAML front matter, condition type, usage snippet, and metric handling
    - **Validates: Requirements 10.1, 10.2, 10.4, 10.5**

  - [x] 5.3 Implement SpaceDeployer class
    - Implement `deploy_space()` to create/update deepradadiya/controlnet-demo Space
    - Implement `_prepare_space_files()` to gather app.py and requirements.txt
    - Configure Space metadata: SDK=gradio, hardware=t4, visibility=public
    - Handle deployment failures with non-zero exit and error message
    - _Requirements: 9.1, 9.2, 9.3, 9.4_

- [x] 6. Checkpoint - Verify publishing pipeline
  - Ensure all tests pass, ask the user if questions arise.

- [x] 7. Implement README builder
  - [x] 7.1 Create ReadmeBuilder class with header and architecture sections
    - Create README builder within `scripts/publish_to_hub.py` (or separate module `scripts/readme_builder.py`)
    - Implement `_header_section()`: level-1 heading with project title, single-sentence description ≤200 chars
    - Implement `_architecture_diagrams()`: ASCII inference and training pipeline diagrams in fenced code blocks, lines ≤80 chars
    - Place architecture section between project description and results
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 16.1, 16.2, 16.3_

  - [x]* 7.2 Write property test for README code block line length
    - **Property 5: README Code Block Line Length**
    - Use hypothesis to generate README content variations, verify all lines in fenced code blocks are ≤80 characters
    - **Validates: Requirements 11.3**

  - [x]* 7.3 Write property test for README header structure
    - **Property 7: README Header Structure**
    - Use hypothesis to generate README content, verify first line is `# ` heading and first non-empty line after is plain text ≤200 chars
    - **Validates: Requirements 16.1, 16.3**

  - [x] 7.4 Implement results table and visual grid sections
    - Implement `_results_table(metrics)`: table with Model/Condition Type, FID Score, Alignment Score columns
    - Include rows for depth, pose, edge, and Vanilla SD1.5 baseline
    - Annotate alignment scores with metric name (SSIM for edge, Pearson for depth, normalized keypoint distance for pose)
    - Indicate metric direction (lower=better for FID, higher=better for alignment)
    - Implement `_visual_grid_section()`: 4-column grid (Input, Condition Map, Generated, Without ControlNet), ≥2 rows per condition type
    - Handle missing metrics with "N/A" placeholders
    - _Requirements: 12.1, 12.2, 12.3, 12.4, 13.1, 13.2, 13.3_

  - [x]* 7.5 Write property test for results table structure
    - **Property 6: Results Table Structure and Metric Annotation**
    - Use hypothesis to generate random metrics dictionaries, verify table has correct rows, columns, and metric annotations
    - **Validates: Requirements 12.1, 12.3**

  - [x] 7.6 Implement reproducibility, links, and tech stack sections
    - Implement `_reproducibility_section()`: sequential Colab commands (clone, install, dataset, extract, train, evaluate), hardware spec (T4 GPU, 15GB VRAM), estimated training time, verification steps
    - Implement `_links_section(config)`: URLs to all 3 model repos, Space demo, W&B logs
    - Implement `_tech_stack_section()`: ≥5 frameworks/libraries (PyTorch, diffusers, Gradio, etc.)
    - Include resume bullet point with quantitative metric reference
    - _Requirements: 14.1, 14.2, 14.3, 15.1, 15.2, 15.3, 15.4, 15.5_

  - [x]* 7.7 Write property test for model repository URL formation
    - **Property 8: Model Repository URL Formation**
    - Use hypothesis to generate random valid usernames, verify URL pattern `https://huggingface.co/{username}/controlnet-sd15-{type}` and all 3 types present
    - **Validates: Requirements 15.1**

- [x] 8. Wire everything together and integrate
  - [x] 8.1 Implement main() orchestration in publish_to_hub.py
    - Wire AdapterPublisher, CombinedPipelinePublisher, SpaceDeployer, ModelCardGenerator, and ReadmeBuilder into the main() CLI flow
    - Ensure sequential execution: publish adapters → publish combined → deploy space → generate README
    - Write generated README.md to repository root
    - _Requirements: 7.1, 7.2, 7.3, 8.1, 9.1, 10.1_

  - [x] 8.2 Update deployment/app.py Space entry point
    - Update `deployment/app.py` to import and launch the enhanced Gradio app
    - Ensure compatibility with HuggingFace Space runtime (gradio SDK)
    - _Requirements: 9.1, 9.2_

  - [x]* 8.3 Write integration tests for publishing pipeline
    - Mock HfApi and test full publish flow (adapter upload, combined upload, Space deploy)
    - Verify correct repo IDs, file paths, and API call parameters
    - Test authentication fallback chain
    - _Requirements: 7.4, 7.5, 7.6, 7.7, 8.1, 9.1_

- [x] 9. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties from the design document using `hypothesis`
- Unit tests validate specific examples and edge cases
- The existing `scripts/deploy_to_hub.py` is replaced by the new `scripts/publish_to_hub.py`
- The existing `src/app/gradio_app.py` is refactored in-place (not replaced)
- All code targets Python with existing project dependencies (gradio, huggingface-hub, diffusers, hypothesis)

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "2.1", "4.1"] },
    { "id": 2, "tasks": ["2.2", "2.3", "4.2", "4.3"] },
    { "id": 3, "tasks": ["2.4", "4.4", "5.1", "5.3"] },
    { "id": 4, "tasks": ["5.2", "7.1"] },
    { "id": 5, "tasks": ["7.2", "7.3", "7.4"] },
    { "id": 6, "tasks": ["7.5", "7.6"] },
    { "id": 7, "tasks": ["7.7", "8.1", "8.2"] },
    { "id": 8, "tasks": ["8.3"] }
  ]
}
```
