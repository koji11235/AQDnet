# AQDnet

**AQDnet: Deep Neural Network for Protein-Ligand Docking and Scoring**

A research repository for AQDnet, a deep neural network approach to predict binding affinities in protein-ligand docking simulations. This repository demonstrates a complete workflow: structure preprocessing → feature extraction → model training → prediction, with evaluation on CASF-2016 and LIT-PCBA benchmarks.

<img src="./SchematicAbstract.png" alt="AQDnet Overview">

---

## Overview

AQDnet learns spatial and chemical features of protein-ligand complexes to predict docking energies and binding scores. The repository includes:

- **Feature extraction pipeline**: Converts PDB structures into learned feature representations
- **Two model variants**: Docking-specific (Energy30RMSD2.5) and Scoring-specific (Energy02RMSD2.0)
- **Benchmark evaluation**: CASF-2016 docking power, scoring power, and ranking power; LIT-PCBA virtual screening
- **Sample workflow**: Jupyter notebooks demonstrating the complete pipeline

---

## Why This Repository Exists

This research was motivated by limitations in traditional scoring functions for protein-ligand binding prediction. AQDnet uses deep learning to automatically learn feature representations from 3D structures, without hand-crafted descriptor design. The work explores whether learned features improve consensus binding predictions across diverse protein targets.

---

## Repository Structure

```
AQDnet/
├── README.md                          # This file
├── environment/                       # Docker & dependency files
│   ├── Dockerfile                     # Container specification
│   ├── environment.yml                # Conda environment specification
│   └── requirements.txt               # Pip dependency list
├── scripts/                           # Core implementation
│   ├── aqdnet.py                      # Feature extraction interface
│   ├── lpcomp.py                      # Ligand-protein complex utilities
│   ├── model.py                       # Model training interface
│   ├── structure.py                   # DNN architecture definitions
│   ├── runner.py                      # Utility functions (loading, I/O)
│   ├── predict.py                     # Inference interface
│   ├── CASF2016.py                    # CASF-2016 evaluation
│   ├── preppdb.py                     # PDB structure preparation
│   ├── util.py                        # Helper utilities
│   └── visualize.py                   # Result visualization
├── features/                          # Extracted feature files
│   ├── feature_trainset_*.csv         # Training set features (10 shards)
│   ├── feature_validset_*.csv         # Validation set features (10 shards)
│   ├── label_{trainset,validset}.csv  # Binding labels
│   └── *.tfrecords                    # TensorFlow record format
├── models/                            # Trained model checkpoints
│   ├── Docking_Energy30RMSD2.5/       # Docking-specific model
│   └── Scoring_Energy02RMSD2.0/       # Scoring-specific model
├── results/                           # Evaluation results
│   ├── CASF2016_Docking_Energy30RMSD2.5/
│   ├── CASF2016_Scoring_Energy02RMSD2.0/
│   ├── LIT-PCBA_result*.csv
│   ├── ScoringPower_result.csv
│   └── Docking_AQDnet_summary.csv
├── sample_structures/                 # 5 example PDB complexes
│   ├── 184l/, 185l/, 186l/, 187l/, 188l/
│   └── predict_example/
├── 01_generate_features.ipynb         # Feature extraction example
├── 02_train_model.ipynb               # Model training example
├── 03_predict_model.ipynb             # Prediction example
└── SchematicAbstract.png              # Research schematic
```

---

## Installation

### Option 1: Docker (Recommended)

All dependencies are pre-installed in the provided Docker image:

```bash
docker run --rm -it \
  -v "$PWD":/workspace \
  -w /workspace \
  koji11235/aqdnet_env:v0.1.0 \
  bash
```

### Option 2: Local Environment

```bash
# Create conda environment
conda env create -f environment/environment.yml
conda activate aqdnet

# Or via pip
pip install -r environment/requirements.txt
```

Environment files are kept under `environment/` to separate runtime setup from
the research code and benchmark artifacts in the repository root.

**Requirements:**
- Python 3.6+
- TensorFlow 2.3.2
- RDKit
- Dask
- Pandas, NumPy

---

## Quick Start

### 1. Verify Installation

```bash
# In Docker or local environment
python -m aqdnet_cli --help
```

### 2. Generate Features from Sample Structures

```bash
# Extract generic features using only the ligand code
python -m aqdnet_cli featurize \
  --input sample_structures \
  --output sample_features.csv \
  --ligand-code LGD
```

If you want feature generation that exactly matches a prediction model, pass its
`params_for_predict.json` file explicitly:

```bash
python -m aqdnet_cli featurize \
  --input sample_structures/predict_example \
  --feature-param-file models/Docking_Energy30RMSD2.5/params_for_predict.json \
  --output predict_features.csv
```

### 3. Make a Prediction

```bash
# Predict directly from PDB inputs using the model's fg_params and model weights
python -m aqdnet_cli predict \
  --model models/Docking_Energy30RMSD2.5/ \
  --features sample_structures/predict_example \
  --output predictions.csv
```

The prediction output includes both `input_path` and the predicted value, so
each row can be traced back to its source PDB file.

### 4. Training Workflow

```bash
# Validate training inputs from the CLI
python -m aqdnet_cli train \
  --features features/feature_trainset.tfrecords \
  --labels features/label_trainset.csv \
  --output-dir runs/example
```

This command is currently validation-only. It checks the requested paths and
creates the output directory, but it does not run model fitting.
Use `02_train_model.ipynb` for actual model fitting and benchmark reproduction.

---

## Example Workflow

For a complete walkthrough, see the Jupyter notebooks:

1. **01_generate_features.ipynb**  
   Demonstrates feature extraction from PDB structures using the AQDnet algorithm.

2. **02_train_model.ipynb**  
   Shows model training on sample features and validation workflow.

3. **03_predict_model.ipynb**  
   Illustrates inference using trained models.

Each notebook is self-contained and runnable in the Docker environment.

---

## Workflow Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    AQDnet Workflow Pipeline                  │
└─────────────────────────────────────────────────────────────┘

INPUT DATA
   ↓
   └─ PDB Structures (sample_structures/)
   └─ Ligand & Protein Coordinates

   ↓ [Step 1: Feature Extraction]
   └─ scripts/aqdnet.py (FeatureGenerator)
   └─ scripts/lpcomp.py (Ligand-Protein Complex)
   └─ scripts/preppdb.py (Structure Prep)

   ↓ OUTPUT: features/
   ├─ feature_trainset_*.csv (10 shards)
   ├─ feature_validset_*.csv (10 shards)
   ├─ label_{trainset,validset}.csv
   └─ *.tfrecords (TensorFlow format)

   ↓ [Step 2: Model Training]
   └─ scripts/model.py (Model Training Interface)
   └─ scripts/structure.py (DNN Architecture)
   └─ scripts/runner.py (Training Loop)

   ↓ OUTPUT: models/
   ├─ Docking_Energy30RMSD2.5/ (Docking model)
   └─ Scoring_Energy02RMSD2.0/ (Scoring model)

   ↓ [Step 3: Prediction & Evaluation]
   └─ scripts/predict.py (Inference)
   └─ scripts/CASF2016.py (Benchmark Eval)
   └─ scripts/visualize.py (Result Viz)

   ↓ OUTPUT: results/
   ├─ CASF2016_Docking_Energy30RMSD2.5/
   ├─ CASF2016_Scoring_Energy02RMSD2.0/
   ├─ LIT-PCBA_result*.csv
   └─ ScoringPower_result.csv
```

**Entry Points:**
- **Python CLI** (Recommended): `python -m aqdnet_cli [command]`
- **Jupyter Notebooks** (Tutorial): Run `01_*.ipynb`, `02_*.ipynb`, `03_*.ipynb`
- **Docker Container** (Reproducible): Use `koji11235/aqdnet_env:v0.1.0`

---

## Results Summary

### CASF-2016 Docking Benchmark
- **Docking Power (Top-1 Rank)**: Reported in `results/CASF2016_Docking_Energy30RMSD2.5/`
- **Scoring Power**: Reported in `results/CASF2016_Scoring_Energy02RMSD2.0/`
- **Ranking Power**: Kendall τ and Spearman ρ in result files

### LIT-PCBA Virtual Screening
- **Enrichment Factor (EF1%)**: Summarized in `results/LIT-PCBA_result_summary.csv`
- **Per-target results**: `results/LIT-PCBA_result.csv`

See individual CSV files in `results/` for detailed metrics.

---

## Reproducibility & Limitations

### What Is Reproducible
- ✓ Feature extraction from PDB structures
- ✓ Model training pipeline with sample features  
- ✓ Prediction on new structures
- ✓ Evaluation code for CASF-2016 and LIT-PCBA

### What Requires External Files
- ✗ Full benchmark training & evaluation  
  → Requires complete CASF-2016 training set and LIT-PCBA data (must be obtained separately)
- ✗ Pretrained model weights  
  → `best_model.h5` files moved to Google Drive (see "Artifact Note")

### Known Limitations
1. **Python 3.6 Legacy**: TensorFlow 2.3.2 is old; no guarantee of compatibility with Python 3.9+.
2. **CUDA Dependency**: GPU acceleration requires CUDA 10.1 and cuDNN; CPU-only mode is slower.
3. **Memory Requirements**: Feature extraction on large datasets requires 16+ GB RAM or Dask chunking.
4. **Notebook Dependency**: Primary examples are Jupyter notebooks; CLI may require additional polish for production use.
   Training remains notebook-first; CLI support focuses on feature extraction, prediction, and input validation.
5. **SMILES Support**: Current implementation uses only 3D coordinates; no method for SMILES-to-3D conversion.

---

## Artifact Note

### Model Weights

The trained model checkpoints are **too large for GitHub** and are hosted on Google Drive:

**Docking-specific model:**
- `models/Docking_Energy30RMSD2.5/best_model.h5`  
- [Download from Google Drive](https://drive.google.com/drive/folders/1i9p5FpYisXrYICDraLmpMztvfA5Zn_-j?usp=share_link)

**Scoring-specific model:**
- `models/Scoring_Energy02RMSD2.0/best_model.h5`  
- [Download from Google Drive](https://drive.google.com/drive/folders/1i9p5FpYisXrYICDraLmpMztvfA5Zn_-j?usp=share_link)

**Why external storage?**
- Model files are ~200–500 MB each; GitHub supports max 100 MB per file and total LFS quota is limited.
- Easier version control and download management via Google Drive.

**No Proprietary Data:**
- All code is original.
- Sample structures (184l, 185l, etc.) are from public PDB; reproducible from 4-letter PDB code.
- Features are derived; no proprietary descriptor definitions.

---

## Future Improvements

- [ ] Migrate to modern TensorFlow (2.12+) with Python 3.10+ support
- [ ] Add type hints and full unit/integration tests
- [ ] Package as `pip install aqdnet` with unified CLI
- [ ] Support for molecular graphs / DGL backend
- [ ] Interactive visualization of feature importance
- [ ] Automated benchmark download & evaluation pipeline
- [ ] Containerize with CPU-only and GPU variants
- [ ] Add SMILES parsing for de novo ligand design

---

## License

MIT License. See [LICENSE](LICENSE) file for details.

**Use terms:**
- Code and examples are freely usable for research and education.
- Model weights are provided as-is for reproducibility and benchmarking.
- If redistributing trained models, please acknowledge the original source.

---

## Data & Intellectual Property

### Code License
All source code in this repository is licensed under the **MIT License**. You are free to:
- Use for any purpose (academic, commercial, personal)
- Modify and redistribute
- Include in larger projects

The only requirement is attribution to the original authors.

### Data Sources

This repository includes or uses the following public data:

| Dataset | Source | License | Use in AQDnet |
|---------|--------|---------|--------------|
| **Sample Structures** (184l, 185l, etc.) | [RCSB Protein Data Bank](https://www.rcsb.org) | CC0 (Public Domain) | Feature extraction examples |
| **CASF-2016 Benchmark** | [CASF-2016 Website](http://www.casf.org) | Available for research | Model evaluation |
| **LIT-PCBA Dataset** | [MoleculeNet](https://moleculenet.org) | CC0 (Public Domain) | Virtual screening benchmarks |

### No Proprietary Data

**Important**: This repository contains **no proprietary or restricted-use data**:
- ✓ All code is original (written entirely in this research)
- ✓ Sample structures are from public databases (freely available via PDB ID)
- ✓ Benchmarks use publicly available datasets (CASF-2016, MoleculeNet)
- ✓ No private company data, no clinical records, no confidential information

### How to Cite This Repository

If you use AQDnet in your research, please cite as follows:

**BibTeX:**
```bibtex
@software{aqdnet2023,
  author = {Shiota, Koji},
  title = {AQDnet: Deep Neural Network for Protein-Ligand Docking and Scoring},
  url = {https://github.com/koji11235/AQDnet},
  year = {2023}
}
```

**APA:**
Shiota, K. (2023). AQDnet: Deep neural network for protein-ligand docking and scoring. Retrieved from https://github.com/koji11235/AQDnet

---

## Citation

If you use AQDnet in your research, please cite the original work:
(Add citation info if published)
