# eyeprep

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Standardized preprocessing pipeline for BIDS-formatted eye-tracking data.

## Why This Exists

Eye-tracking is increasingly used in fMRI and cognitive neuroscience, but preprocessing is inconsistent. We've created a [BIDS Extension Proposal (BEP20)](link) to standardize eye-tracking data format, and this tool makes that standard immediately useful.

**Format your data once with BIDS → Get preprocessing forever**

## Features

- Reads BIDS eye-tracking data natively
- Modular preprocessing steps (blink removal, drift correction, smoothing, etc.)
- Comprehensive quality control metrics and visualizations
- Docker container for reproducibility
- Extensive documentation and tutorials
- Validated on multiple eye-tracker types


## Documentation (TODO)

- [Getting Started Guide](docs/getting_started.md)
- [Preprocessing Options](docs/preprocessing.md)
- [Configuration Files](docs/configuration.md)
- [Tutorial Notebooks](examples/)
- [API Reference](docs/api.md)

## 🤝 Contributing

We welcome contributions! This project is part of [Brainhack 2026](https://brainhack-marseille.github.io/).

- See our [Contributing Guide](CONTRIBUTING.md)
- Join the discussion on [GitHub Discussions](https://github.com/sinaklg/eyeprep/discussions)


## 🏆 Brainhack Project

This tool is being developed as part of Brainhack 2026. Check out the [project page](https://brainhack-marseille.github.io/) to get involved!

**Skills we need:**
- Python development
- Documentation writing
- Data visualization
- DevOps (Docker, CI/CD)

## Citation

If you use this tool, please cite:
- Our BEP20 paper: [github_discussion] (https://github.com/bids-standard/bids-specification/discussions/2218#discussioncomment-15164225)


## 📧 Contact

- Project Lead: [Sina Kling] - [@github](https://github.com/sinaklg) - email sina.kling@univ-amu.fr
- Issues: [GitHub Issues](https://github.com/sinaklg/eyeprep/issues)
- Chat: [Mattermost](link)
