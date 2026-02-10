# eyeprep

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Standardized preprocessing pipeline for BIDS-formatted eye-tracking data.

Authors: Sina KLING, Marianne DUYCK, Giorgio MARINATO, Martin SZINTE
-------
Version 0.02


## Installation steps (dependencies)

Install SR reseach developers kit: https://www.sr-research.com/support/docs.php?topic=linuxsoftware

```
git clone https://github.com/bids-standard/eye2bids.git

```

```
cd eye2bids
pip install .
```

## Why This Exists

Eye-tracking is increasingly used in fMRI and cognitive neuroscience, but preprocessing is inconsistent. We've created a [BIDS Extension Proposal (BEP20)](https://bids.neuroimaging.io/extensions/beps/bep_020.html) to standardize eye-tracking data format, and this tool makes that standard immediately useful.
Eyetracking BIDS specification documentation: https://bids-specification.readthedocs.io/en/latest/modality-specific-files/physiological-recordings.html

**Format your data once with BIDS → Get preprocessing forever**

## How It Works (for now)

<p align="center">
  <img src="docs/images/pipeline.png" alt="eyeprep pipeline" width="650"/>
</p>

eyeprep takes BIDS-formatted eye-tracking data through a series of validated preprocessing steps, followed by quality assessment to ensure your data is analysis-ready.

### Preprocessing Steps
- **Blink Removal** → **Interpolation** → **Normalization** → **Drift Correction** → **Smoothing** → **Downsampling**

[Learn more about each step →](docs/preprocessing.md)

### Quality Assessment
- produces individual quality metrics : Proportion of missing data, gaze overview relative to screen, proportion of valid data after blink removal (inkl. blink distribution plot), noise quantification 
- Saccade detection 

## 🤝 Contributing

We welcome contributions! This project is part of [Brainhack 2026](https://brainhack-marseille.github.io/).

- See our [Contributing Guide](CONTRIBUTING.md)
- Join the discussion on [GitHub Discussions](https://github.com/sinaklg/eyeprep/discussions)

## Citation

If you use this tool, please cite:
- Our BEP20 paper: [BIORXIV] (https://www.biorxiv.org/content/10.64898/2026.02.03.703514v1)


## 📧 Contact

- Project Lead: [Sina Kling] - [@github](https://github.com/sinaklg) - email sina.kling@univ-amu.fr
- Issues: [GitHub Issues](https://github.com/sinaklg/eyeprep/issues)
- Chat: [Brainhack Mattermost](https://mattermost.brainhack.org/brainhack/channels/bhg26-marseille-eyeprep)
