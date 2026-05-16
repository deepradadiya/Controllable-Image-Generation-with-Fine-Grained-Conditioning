# Implementation Plan: Data Pipeline

## Overview

Implement the ControlNet training data pipeline as five self-contained Python scripts in the `data/` directory. Each script downloads or extracts conditioning maps from the fusing/fill50k dataset, supports checkpoint/resume via file-existence checks, and is independently runnable. Property-based tests validate core pure functions using hypothesis.

## Tasks

- [x] 1. Implement Dataset Downloader
  - [x] 1.1 Create `data/download_dataset.py` with DatasetDownloader class
    - Load "fusing/fill50k" from HuggingFace datasets library
    - Select first 5000 examples using default split ordering
    - Save images as zero-padded PNGs (`00000.png`–`04999.png`) to `data/raw/images/`
    - Save prompts to `data/raw/prompts.json` as filename→prompt mapping
    - Validate each image (width > 0, height > 0), log failures to `data/raw/bad_files.log`
    - Display tqdm progress bar during processing
    - Print final statistics (valid count, avg width, avg height, first prompt)
    - Implement retry logic: 3 retries with 5-second delay on network errors
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9_

  - [ ]* 1.2 Write property tests for download utilities
    - **Property 1: Filename zero-padding format** — For any index in [0, 4999], formatting produces exactly 5 digits + ".png"
    - **Property 2: Prompts JSON round-trip** — Serializing and deserializing filename→prompt mapping produces identical output
    - **Property 3: Image dimension validation** — Returns True iff width > 0 and height > 0
    - **Property 12: Statistics computation correctness** — Average width/height equals arithmetic mean of all dimensions
    - **Validates: Requirements 1.3, 1.4, 1.6, 1.8**

- [x] 2. Implement Depth Map Extraction
  - [x] 2.1 Create `data/extract_depth.py` with DepthMapExtractor class
    - Include module-level docstring (≥30 words) explaining depth maps and DPT
    - Load Intel DPT-Large ("Intel/dpt-large") from HuggingFace
    - Read source images from `data/raw/images/`, save depth maps to `data/depth/`
    - Resize images to 512×512, run DPT inference, apply per-image min-max normalization to [0, 255] uint8
    - Process in batches of 8 for T4 GPU VRAM efficiency
    - Skip images whose output file already exists (checkpoint/resume)
    - Log and skip failed images, continue processing
    - Display tqdm progress bar for batch processing
    - Display 3 depth maps at equal intervals with originals using matplotlib
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 6.1_

  - [ ]* 2.2 Write property tests for depth extraction utilities
    - **Property 4: Depth min-max normalization** — For any 2D float array where max > min, output min=0, max=255, all values in [0, 255]
    - **Property 5: Output path stem preservation** — Source filename stem is preserved in output path with .png extension
    - **Property 6: Checkpoint skip-if-exists** — Only images without existing outputs are processed
    - **Validates: Requirements 2.2, 2.3, 2.5**

- [x] 3. Implement Pose Skeleton Extraction
  - [x] 3.1 Create `data/extract_pose.py` with PoseMapExtractor class
    - Include module-level docstring (≥30 words) explaining pose skeletons and DWPose
    - Load DWPose model from controlnet_aux library
    - Read source images from `data/raw/images/`, save pose maps to `data/pose/`
    - Run DWPose inference, render colored stick figure on black background
    - Save blank black image (512×512 RGB, all zeros) when no keypoints detected
    - Skip images whose output file already exists (checkpoint/resume)
    - Log and skip failed images, continue processing
    - Display tqdm progress bar
    - Display 3 pose skeletons (preferring detected poses) with originals using matplotlib
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 6.2_

  - [ ]* 3.2 Write property tests for pose extraction utilities
    - **Property 7: Blank pose for undetected keypoints** — When no keypoints detected, output is (512, 512, 3) uint8 array of all zeros
    - **Validates: Requirements 3.3**

- [x] 4. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 5. Implement Edge Map Extraction
  - [x] 5.1 Create `data/extract_edges.py` with EdgeMapExtractor class
    - Include module-level docstring (≥30 words) explaining edge maps and Canny algorithm
    - Include inline comments within 3 lines of Canny thresholds explaining low_threshold=100 and high_threshold=200
    - Apply OpenCV Canny with low_threshold=100, high_threshold=200 after grayscale conversion
    - Read source images from `data/raw/images/`, save edge maps to `data/edges/`
    - Execute entirely on CPU, no GPU required
    - Preserve source image dimensions in output (single-channel grayscale, 8-bit)
    - Skip images whose output file already exists (checkpoint/resume)
    - Log warning and skip corrupted/unreadable images
    - Display tqdm progress bar showing processed/total count
    - Display first 3 edge maps side by side with originals using matplotlib
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 6.3, 6.4_

  - [ ]* 5.2 Write property tests for edge extraction utilities
    - **Property 8: Edge map binary output with dimension preservation** — For any (H, W) source, output is (H, W) single-channel with all pixels either 0 or 255
    - **Validates: Requirements 4.1, 4.8**

- [x] 6. Implement Dataset Verification
  - [x] 6.1 Create `data/verify_dataset.py` with DatasetVerifier class
    - Enumerate samples by listing filenames in `data/raw/images/`
    - Check each sample has 4 files: source image, depth map, pose skeleton, edge map
    - Verify each sample has a prompt entry in `data/raw/prompts.json` (non-empty, ≥5 chars)
    - Validate conditioning map files: loadable, >0 bytes, ≥256×256 pixels
    - Halt with error if `prompts.json` is missing or invalid JSON
    - Log incomplete samples with specific missing/invalid file details to stdout
    - Print final summary: "X samples ready for training"
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7_

  - [ ]* 6.2 Write property tests for verification utilities
    - **Property 9: Verification completeness invariant** — Complete count equals stems where all files exist, are loadable, meet size requirements, and have valid prompt
    - **Property 10: Prompt validation logic** — Returns True iff string has ≥5 characters; empty/short/null returns False
    - **Property 11: Conditioning map file validation** — Returns True iff file exists, >0 bytes, decodable as image, ≥256×256
    - **Validates: Requirements 5.1, 5.2, 5.3, 5.6**

- [x] 7. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each script in `data/` is self-contained and independently runnable
- All scripts use file-existence checks for checkpoint/resume (no external state)
- Property tests use the `hypothesis` library with minimum 100 iterations per property
- Depth extraction uses batch size 8 on T4 GPU (~5.5GB VRAM usage)
- Edge extraction is CPU-only, no GPU required
- Module-level docstrings must be ≥30 words explaining the conditioning map type and extraction method

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1"] },
    { "id": 1, "tasks": ["1.2", "2.1", "3.1", "5.1"] },
    { "id": 2, "tasks": ["2.2", "3.2", "5.2", "6.1"] },
    { "id": 3, "tasks": ["6.2"] }
  ]
}
```
