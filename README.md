# PlaneCycle

PlaneCycle: Training-Free 2D-to-3D Lifting of Foundation Models Without Adapters ([arXiv]())

## Overview

<div align="center">
  <img src="assets/feature_visualization.png" width="800" alt="DINOv3 Feature Visualization">
  <p align="center" style="max-width: 800px; margin: 0 auto;">
    <i>
      <b>PCA visualizations of frozen lifted DINOv3 features</b><br/>
      On three 3D datasets across HW, DW, and DH planes (inconsistencies circled).
    </i>
  </p>
</div>

<br/>
<br/>

<div align="center">
  <img src="assets/Planecycle.png" width="800" alt="PlaneCycle Overview">
  <p align="center" style="max-width: 800px; margin: 0 auto;">
    <i>
      <b>Overview of PlaneCycle across three orthogonal planes (HW, DW, DH).</b><br/>
      Flattened slice tokens are processed by shared ViT layers with plane-specific RoPE.
    </i>
  </p>
</div>

<br/>

## Code structure

* ``planecycle``
  Contains the foundational logic for the PlaneCycle architecture.
  * ``operators``: Implementation of PlaneCycle operator.
  * ``converters``: ViT converters.
* ``experiments/dinov3``
  the scripts to run experiments.
  * ``medminist``: Comprehensive pipelines for training and benchmarking on the 6 different 3D MedMNIST+ datasets.
* ``dinov3/dinov3/models``
  * ``vision_transformer``: Our modified Vision Transformer implementation.
