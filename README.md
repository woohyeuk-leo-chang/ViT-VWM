# ViT-VWM: Modeling Visual Working Memory with Vision Transformers

An ongoing PyTorch implementation of Vision Transformers (ViT) designed to model human Visual Working Memory (VWM) on continuous report tasks. This project investigates whether standard deep learning architectures can replicate key psychophysical phenomena (like set-size effects and precision limitations) without explicit biological hard-coding.

## 📌 Project Overview

Visual Working Memory is often tested using a **continuous report task**: subjects see a set of colored items, a delay occurs, and then they must report the color of a probed item on a continuous color wheel.

This repository trains a Vision Transformer to perform this exact task end-to-end:

1. **Input:** An image containing  colored items (Memory Display) + an image indicating which item to report (Probe Display).
2. **Output:** A predicted angle on the color wheel ( to ).
3. **Loss:** Von Mises / Circular Loss (respecting the circular nature of color space).

## 🚀 Key Features

* **Synthetic Data Generation:** Generates infinite training samples of "memory displays" on the fly (circles/squares of random colors).
* **Circular Loss Function:** Uses cosine similarity to handle the periodic nature of angular targets (e.g., ).
* **Set Size Analysis:** Automatically tracks performance across different set sizes (1 to 8 items) to mimic human cognitive load curves.

## 📊 Methodology

### The Task

The model receives two  images:

* **Memory Image:** Contains 1–8 colored items at random locations.
* **Probe Image:** Contains spatial cues (or a single item location) indicating which target to report.

### Performance Metrics

We measure the **angular error** (degrees) between the predicted color and the ground truth.
