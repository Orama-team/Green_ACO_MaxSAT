# Data Directory

This directory contains the core data inputs and configuration artifacts required to run the pipeline.

- **manifest.csv**: The definitive index of the 50 instances comprising the benchmark. It specifies which instances belong to the 'core' subset (mandatory) and provides metadata such as the best-known solution quality for proper scoring normalization.
- **mse24/**: Directory meant to hold the raw computational instances (in .wcnf.xz format) sourced from the MaxSAT Evaluation 2024.
- **comparison/**: Subdirectory containing pre-computed hyperparameter calibration artifacts and configurations for the comparison methods (e.g., GA and ACO variants).

**Note:** The data in this directory is considered read-only. The pipeline expects these files to remain immutable to ensure full reproducibility of the experimental results.
