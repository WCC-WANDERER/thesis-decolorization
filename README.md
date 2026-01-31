# Comparative Analysis of Image Decolorization Algorithms

This repository contains the Python implementation and evaluation framework developed for comparative analysis of image decolorization algorithms in the context of book cover design.

The project implements six decolorization methods (3 linear, 3 non-linear) and evaluates them using five quantitative metrics and execution time.

## Overview

This framework automates the batch processing of images and generates objective performance metrics to compare different algorithms.

**Key Features:**
* **Batch Processing:** Automatically converts all images in an input folder.
* **6 Algorithms:** Implementations ranging from basic standards to advanced methods.
* **5 Quantitative Metrics:** Automated calculation of contrast and structural fidelity scores.
* **Numba Optimization:** Accelerated implementation of the computationally heavy Color2Gray algorithm.

## Algorithms Implemented

| Category | Method | Description |
| :--- | :--- | :--- |
| **Linear** | **Average** | Arithmetic mean of RGB channels. |
| | **Luminance** | Rec. 601: $0.299R + 0.587G + 1.114B$. |
| | **CIELAB** | $L^*$ channel from the CIELAB color space. |
| **Non-Linear** | **GD** | Contrast maximization via chromatic channels. |
| | **CorrC2G** | Correlation-based mapping approach. |
| | **Color2Gray** | Optimization using neighbor pixel differences. |

## Evaluation Metrics

The framework automatically calculates the following metrics for every output image:

1.  **RMS (Root Mean Square) Contrast:** Measures overall global contrast.
2.  **NRMS (Normalized Root Mean Square):** Measures pixel-level fidelity to the original image.
3.  **GRR (Gradient Recall Ratio):** Evaluates how well edge information is preserved.
4.  **C2G-SSIM:** Structural Similarity Index optimized for Color-to-Gray conversion.
5.  **E-Score:** A composite metric measuring edge preservation.

## Installation & Requirements

The project is written in Python. You can install the required dependencies using pip.

```bash
# Clone the repository
git clone https://github.com/WCC-WANDERER/thesis-decolorization
cd your-repo-name

# Install dependencies
pip install -r requirements.txt

