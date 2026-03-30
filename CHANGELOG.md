# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-30

### Added
- **CLI Interface** (`aqdnet_cli` package): User-friendly command-line tools for feature extraction, model training, prediction, and demo
- **Setup Automation** (`setup.py`): Package installation support for easier distribution
- **Test Suite** (`tests/test_smoke.py`): 8 basic unit tests covering core workflows
- **Environment Files**: `environment.yml` and `requirements.txt` for reproducible Python setup
- **Docker Support**: Full Dockerfile for containerized, dependency-locked execution
- **Improved Documentation**:
  - Comprehensive README with structure, installation, and quick start guides
  - Workflow overview diagram showing data pipeline
  - Examples for all entry points (CLI, notebooks, Docker)
  - Explicit artifact management notes
- **Lowercase Directory Structure**: Better consistency with Python packaging conventions
  - `Scripts/` → `scripts/`
  - `Features/` → `features/`
  - `Models/` → `models/`
  - `Results/` → `results/`
  - `SampleStructures/` → `sample_structures/`
- **Notebook Naming**: Clearer sequential naming
  - `Ex1_generate_feature.ipynb` → `01_generate_features.ipynb`
  - `Ex2_train_model.ipynb` → `02_train_model.ipynb`
  - `Ex3_predict.ipynb` → `03_predict_model.ipynb`

### Changed
- **Core Scripts**: Removed hardcoded development environment paths for portability
- **Documentation**: Restructured README for clarity; added reproducibility matrix

### Fixed
- Hardcoded path reference in `scripts/predict.py` that prevented portability
- Directory name inconsistencies across documentation
- Missing artifact download instructions

### Known Limitations
- Python 3.6 legacy (tested with TensorFlow 2.3.2 only)
- GPU support requires CUDA 10.1 and cuDNN
- Large dataset handling requires 16+ GB RAM or Dask chunking
- No built-in SMILES-to-3D conversion

---

## Planned Releases

See [ROADMAP.md](./ROADMAP.md) for future development direction.
