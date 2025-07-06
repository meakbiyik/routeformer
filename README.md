# Routeformer: Leveraging Driver Field-of-View for Multimodal Ego-Trajectory Prediction

<div align="center">

[![Website](docs/imgs/badges/badge_project_page.svg)](https://meakbiyik.com/routeformer/)
[![Paper](docs/imgs/badges/badge_pdf.svg)](https://arxiv.org/abs/2312.08558)
[![Dataset](docs/imgs/badges/badge_dataset.svg)](https://huggingface.co/datasets/meakbiyik/GEM_gaze-assisted-ego-motion-in-driving)

</div>

This repository will host the code and supplementary materials for our paper **"Leveraging Driver Field-of-View for Multimodal Ego-Trajectory Prediction"** accepted at ICLR 2025. It includes the implementation of our novel multimodal ego-trajectory prediction network, **Routeformer**, and the GEM dataset.

<!-- limit to max-width of 400px -->

<div align="center">
<img src="docs/imgs/model.png" alt="Figure" style="max-width:auto; max-height:300px;" />
</div>

## Overview

Understanding drivers' decision-making is crucial for road safety. While predicting an ego-vehicle’s path is important for driver-assistance systems, most existing methods focus primarily on external factors like other vehicles' motions. Our work addresses this limitation by integrating the driver's attention with the surrounding scene, combining GPS data, environmental context, and driver field-of-view information (first-person video and gaze fixations).

In this repository, you will eventually find:

- **Code:** The implementation of Routeformer and associated tools.
- **GEM Dataset:** A comprehensive dataset of urban driving scenarios enriched with synchronized driver field-of-view and gaze data. The link to the GEM dataset will be provided once available.

## Getting Started

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/meakbiyik/routeformer.git
    cd routeformer
    ```

2. Install the dependencies using [Poetry](https://python-poetry.org/):

    ```bash
    poetry install
    ```

    **Note on the `av` dependency:** This project uses the `av` library for video processing, which has `ffmpeg` as a dependency. If you have `ffmpeg` already installed on your system, you might encounter issues with the default installation. In that case, it is recommended to install `av` with the following command to avoid building it from source:

    ```bash
    pip install av --no-binary av
    ```

## Repository Structure

Here's a brief overview of the most important files and directories:

- `routeformer/models/routeformer.py`: This file contains the core implementation of the **Routeformer** model.
- `experiments/full_comparison.py`: This is the main script to run the experiments and reproduce the results from the paper.
- `routeformer/io/dataset.py`: Contains the dataset loading and processing logic.
- `docs/`: Contains additional documentation, including details on the dataset and data extraction.

## Abstract

Understanding drivers' decision-making is crucial for road safety. Although predicting the ego-vehicle's path is valuable for driver-assistance systems, existing methods mainly focus on external factors like other vehicles' motions, often neglecting the driver's attention and intent. To address this gap, we infer the ego-trajectory by integrating the driver's attention and the surrounding scene. We introduce Routeformer, a novel multimodal ego-trajectory prediction network combining GPS data, environmental context, and driver field-of-view—comprising first-person video and gaze fixations. We also present the Path Complexity Index (PCI), a new metric for trajectory complexity that enables a more nuanced evaluation of challenging scenarios. To tackle data scarcity and enhance diversity, we introduce GEM, a comprehensive dataset of urban driving scenarios enriched with synchronized driver field-of-view and gaze data. Extensive evaluations on GEM and DR(eye)VE demonstrate that Routeformer significantly outperforms state-of-the-art methods, achieving notable improvements in prediction accuracy across diverse conditions. Ablation studies reveal that incorporating driver field-of-view data yields significantly better average displacement error, especially in challenging scenarios with high PCI scores, underscoring the importance of modeling driver attention. All data, code, and models will be made publicly available.

## Citation

If you use our work, please consider citing our paper:

```bibtex
@inproceedings{akbiyik2023routeformer,
    title={Leveraging Driver Field-of-View for Multimodal Ego-Trajectory Prediction},
    author={M. Eren Akbiyik, Nedko Savov, Danda Pani Paudel, Nikola Popovic, Christian Vater, Otmar Hilliges, Luc Van Gool, Xi Wang},
    booktitle={International Conference on Learning Representations},
    year={2025}
}
