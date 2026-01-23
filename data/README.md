# Data Directory

This directory is reserved for local data artifacts used during experimentation.

## Notes
- The primary dataset (IMDb Movie Reviews) is loaded programmatically using the Hugging Face `datasets` library.
- No raw dataset files are stored in this repository to ensure:
  - reproducibility
  - reduced repository size
  - compliance with dataset licensing and redistribution policies

## Usage
During local experimentation, intermediate data artifacts (if any) may be generated here and are excluded from version control.
