# Analog Clock Reading in the Wild: YOLO + Vision Transformer

Two-stage pipeline for reading analog clocks in real photographs, evaluated on **TickTockVQA**
([Choi et al., CVPR 2026 Findings](https://arxiv.org/abs/2603.08011)). A pretrained detector localizes the clock,
and a fine-tuned ViT-B/16 reads the time on the crop.

## Results

TickTockVQA test set (5,247 images), evaluated with the paper's protocol: hour accuracy, minute accuracy
(±2 min) and full-time accuracy (hour and minute both correct). TTA: averaging with the horizontally flipped
image.

| Input | Model | Params | Hour | Minute | Full time |
|---|---|---:|---:|---:|---:|
| Full image | ResNet-50 | 24 M | 37.15 | 46.86 | 25.08 |
| Full image | ConvNeXt-S + TTA | 50 M | 57.73 | 66.46 | 47.57 |
| Full image | ViT-B/16 + TTA | 86 M | 48.16 | 57.77 | 36.17 |
| YOLO crop | ConvNeXt-S + TTA | 50 M | 80.54 | 82.41 | 73.93 |
| YOLO crop | **ViT-B/16 + TTA** | 86 M | **81.59** | **85.00** | **75.68** |
| — | Llama-3.2-11B + Swap-DPO (best in [1]) | 11 B | 56.41 | 66.69 | 46.22 |

The Visual Genome subset of the training split is held out for validation and used for all model selection.
Training therefore uses 5,986 images, compared with 7,236 in [1]. The results come from a single run, and
re-runs vary by about ±1 point.

## Notebooks

The notebooks contain the complete executed runs, including training logs and results. Run them in order
from the repository root:

1. `01_data_preparation.ipynb`: downloads TickTockVQA, builds the splits and the image cache.
2. `02_finetuning_full_image.ipynb`: fine-tunes ResNet-50 and ConvNeXt-S on the whole image.
3. `03_yolo_detection_crop.ipynb`: detects the clock (YOLO11x, with YOLO-World as fallback) and fine-tunes ConvNeXt-S on the crops.
4. `04_vision_transformers.ipynb`: fine-tunes ViT-B/16 on full images and on crops.

The dataset, the pretrained backbones (timm) and the YOLO weights (Ultralytics) are downloaded automatically.
A CUDA GPU is required; about 8 GB of VRAM is enough.

## Dependencies

Python ≥ 3.10:

```bash
pip install -r requirements.txt
```

`torch`, `torchvision`, `timm`, `ultralytics`, `datasets`, `numpy`, `pandas`, `matplotlib`, `pillow`,
`scikit-learn`, `jupyter`.

## Reference

[1] J. Choi, J. W. Lee, S. You, J. Lee. *It's Time to Get It Right: Improving Analog Clock Reading and Clock-Hand
Spatial Reasoning in Vision-Language Models.* CVPR 2026 Findings. arXiv:2603.08011.

Code under the MIT License. TickTockVQA images keep their original licenses and are not redistributed.

Jorge Rodríguez González · M.Sc. in Artificial Intelligence, Universitat Politècnica de València, 2026