# `loss.py`

`MultiSegmentationLoss` provides a combined loss that can select a combination from **Cross-entropy Loss**, **Dice Loss** and **Boundary Dice Loss**. Notably, all losses are computed as volumetric loss.



# `schedulers.py`

`WarmupOneCycleLR` realizes linear warmup with cosine annealing following.

`build_scheduler` contains various choices for schedulers, including PyTorch built-in schedulers and `WarmupOnceCycleLR`.

Notably, this file is edited basing on an original **DINOv3** **version.**



# `utils.py`

Regular utilization functions.