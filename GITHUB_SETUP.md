# GitHub Repository Configuration Guide

This document describes the recommended GitHub repository settings for AQDnet.

## Instructions for GitHub Web UI Configuration

Follow these steps to configure your GitHub repository for optimal discoverability and usability.

### 1. Repository Settings (Settings tab → General)

**Description:**
```
Deep neural network for protein-ligand docking and scoring prediction. 
Feature extraction → Model training → Benchmarking using CASF-2016 and LIT-PCBA.
Docker included.
```

**Website (Homepage URL):**
- Leave blank (or link to paper/documentation if available)

**Repository visibility:**
- ✓ Public (already set)

---

### 2. Topics (Add up to 30 topics)

Click **"Manage topics"** and add the following tags:

**Core Topics (Essential):**
- `deep-learning`
- `molecular-docking`
- `protein-ligand`
- `drug-discovery`
- `tensorflow`

**Methods:**
- `neural-networks`
- `machine-learning`
- `scoring-functions`
- `binding-affinity`

**Applications:**
- `computational-chemistry`
- `chemoinformatics`
- `bioinformatics`
- `virtual-screening`

**Tools & Data:**
- `pdb`
- `smina`
- `casf-2016`
- `lit-pcba`

**Infrastructure:**
- `docker`
- `reproducible-research`
- `jupyter-notebook`

**Optional (supplementary):**
- `benchmark`
- `crystallography`
- `molecular-modeling`
- `python`
- `tensorflow-2`

---

### 3. About Section (Settings tab → General → About)

**Edit the repository details:**

```
About (Short description):
Deep neural network for protein-ligand binding prediction

Website:
[Leave blank or add paper link]

Use as a template:
Uncheck (not a template)

Include in searches:
Check ✓ (improves discoverability)

Discussions:
Uncheck (optional for mature research)

Sponsored button:
Uncheck (for research, not applicable)

Social preview:
Upload the SchematicAbstract.png image for better sharing

Delete:
Ensure backup exists before considering
```

---

### 4. Branch Protection Rules (Optional but Recommended)

**Settings → Branches → Add rule for `main` branch:**

- ✓ Require pull request reviews before merging
- ✓ Dismiss stale pull request approvals when new commits are pushed
- ✓ Require branches to be up to date before merging
- Uncheck: Require status checks to pass (no CI/CD configured yet)

---

### 5. Manage Access (Settings → Collaborators and teams)

- Add collaborators with appropriate roles (Admin, Maintain, Triage, Write, Read)
- Current: Only repository owner

---

### 6. Releases & Artifacts

Once configured, create GitHub Releases:

**Settings → Releases → Create new release:**

**For v0.1.0 (Initial Release):**
```
Tag: v0.1.0
Title: AQDnet v0.1.0 - Initial Release

Release notes:
## What's New in v0.1.0

### 🎉 Major Updates
- CLI interface for all operations (feature extraction, training, prediction)
- Comprehensive documentation and examples
- Docker container with locked dependencies
- Smoke tests for critical workflows
- Improved directory structure (lowercase conventions)

### 📚 Documentation
- Restructured README with workflow diagrams
- CHANGELOG.md for version tracking
- ROADMAP.md for future development
- License clarification with data attribution

### 🔧 Installation
- Docker: `docker pull koji11235/aqdnet_env:v0.1.0`
- Local: `conda env create -f environment/environment.yml`
- Pip: `pip install -r environment/requirements.txt`

### 📝 Usage
```bash
# Feature extraction
python -m aqdnet_cli featurize --input sample_structures/ --output features.csv

# Run demo
python -m aqdnet_cli demo

# Run tests
python -m unittest discover -s tests
```

### ⚠️ Known Limitations
- Python 3.6 only (legacy TensorFlow)
- GPU support requires CUDA 10.1
- Model weights hosted on Google Drive

### 🙏 Contributors
- Original author: Shiota, Koji

### 📄 Related Resources
- RCSB PDB: https://www.rcsb.org/
- CASF-2016: http://www.casf.org/
- MoleculeNet: https://moleculenet.org/

**Pre-release:** Uncheck (stable release)

**Assets:**
- (Optional) Upload source code ZIP, pre-trained model links
```

---

### 7. README Badge (Optional)

Add to top of README.md for status visibility:

```markdown
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.6+](https://img.shields.io/badge/Python-3.6%2B-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.3](https://img.shields.io/badge/TensorFlow-2.3-orange.svg)](https://www.tensorflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-green.svg)](https://www.docker.com/)
```

---

## Summary of Changes

Phase 2-4 completes the repository metadata configuration:

- ✓ Clear repository description
- ✓ 20-25 searchable topics
- ✓ Professional "About" section
- ✓ (Optional) Branch protection rules
- ✓ (Optional) GitHub Release with detailed notes
- ✓ (Optional) Status badges

## Next Steps

1. **Before pushing to GitHub**, verify all Phase 1 & 2 changes in your local environment
2. **Configure settings in GitHub web UI** using the steps above
3. **Create a v0.1.0 release** with detailed release notes
4. **Test Docker image** publicly to ensure reproducibility
5. **Announce release** on relevant channels (social media, forums, etc.)

## Questions?

Refer to [GitHub Documentation](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features) for additional help.
