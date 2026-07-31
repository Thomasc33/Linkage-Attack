# Linkage Attack on Skeleton-based Motion Visualization (LAN)

[![CIKM 2023](https://img.shields.io/badge/CIKM-2023-be123c.svg)](https://dl.acm.org/doi/10.1145/3583780.3615263)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3583780.3615263-blue.svg)](https://doi.org/10.1145/3583780.3615263)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.x+-ee4c2c.svg)](https://pytorch.org/)
[![Project Page](https://img.shields.io/badge/project-linkage.thomasc.tech-0d9488.svg)](https://linkage.thomasc.tech/)

> **Skeleton data captured in VR looks anonymous. A Siamese matching network shows it is not.**

[Paper](https://dl.acm.org/doi/10.1145/3583780.3615263) | [Project Page](https://linkage.thomasc.tech/) | [Data & Checkpoints](https://drive.google.com/drive/folders/1aO2MU_HQDbxHgdZy6HaMFS0REc7DXtKQ?usp=sharing)

## Overview

Motion capture in VR strips away face, voice, and appearance, so the resulting skeleton sequence is often treated as anonymous. It still encodes body measurements, gait, and movement patterns — personally identifiable information (PII) an adversary can link back to a person.

**LAN** (Linkage Attack Neural Network) takes an anonymized target skeleton `s_T` and a reference skeleton `s_R` with known identity, and predicts whether they belong to the same individual. Unlike supervised person re-identification, which can only recognize identities seen during training, a linkage formulation applies to anyone — including actors and action classes the model has never seen.

The repository also evaluates **motion retargeting as a general defense**: casting the raw motion onto a dummy skeleton removes spatial PII without adversarial training against a specific attacker.

### Architecture

LAN follows the structure of a Siamese network:

- **Semantic-Guided Encoders** `E_T`, `E_R` — dynamics representation (position + velocity) → joint-level GCN module → frame-level CNN module, sharing joint-type and frame-index semantics
- **Matching Classifier** `C` — 1D convolution over the concatenated embeddings, two batch-norm layers, three fully connected layers, sigmoid output

Encoders are pre-trained for identity classification, then the whole model is trained end-to-end under a binary cross-entropy loss. Unfreezing the encoders during linkage training adds ~2% F-1 over keeping them frozen.

## Results

### Linkage attack performance (Table 1)

| Attack model | Precision | Recall | F-1 score |
|--------------|-----------|--------|-----------|
| **LAN (ours)** | 0.6830 | **0.8138** | **0.7427** |
| MLP | 0.7059 | 0.6852 | 0.6954 |
| Random forest | 0.7346 | 0.7708 | 0.6576 |

LAN beats the MLP baseline by 4.73 points of F-1 and the random forest by 8.51 points.

### Linkage attack on anonymized data (Table 2)

| Data for visualization | Precision | Recall | F-1 score | Action recognition |
|------------------------|-----------|--------|-----------|--------------------|
| Raw data | 0.6830 | 0.8138 | 0.7427 | **94.25%** |
| UNet | 0.5000 | 1.0000 | 0.6667 | 0.84% |
| ResNet | 0.5000 | 1.0000 | 0.6667 | 0.84% |
| Classical MR | 0.5004 | 0.9963 | 0.6662 | 3.20% |
| Deep MR | **0.5057** | 0.8977 | **0.6469** | **4.55%** |

A perfect anonymizer drives the attack to predict every pair matches (precision 0.5, recall 1.0). All four anonymizers land at or near that point, so all four hide PII effectively — but every one of them pays a large utility cost. Action recognition accuracy is measured with SGN on NTU60+120 (random guessing is 1/120 ≈ 0.84%).

### Scalability

At a sampling size of 100 per actor — 25% of the default setting — LAN still reaches **0.7136 F-1** while taking only **20%** of the default runtime, so the attack does not require a large corpus to be effective.

## Repository Structure

```
Linkage Attack/            # the attack proposed in this paper
  SGN Based Linkage Attack/  # LAN — main model + evaluation CLI + checkpoints
  RF Based Linkage Attack/   # random forest baseline (frame-wise and video-wise)
Attacking Models/          # person re-identification attackers
  SGN Attack Model/          # SGN-based ID + action classification
  RF Attack Model/           # random forest baseline
Defense Models/            # anonymizers
  Motion Retargeting/        # classical MR via BVH conversion
  Mean Skeleton/             # average dummy skeleton construction
Skeleton Info/             # NTU .skeleton parsing, joint statistics, splits
Figures/                   # figures used in the paper
assets/                    # figures used on the project page
index.html                 # project page (https://linkage.thomasc.tech/)
```

Each directory has its own README with the Google Drive link to the pickles and checkpoints it needs.

## Quick Start

All contributions were written inside Jupyter notebooks but have been exported to standalone Python files. Details on installing Jupyter can be found [here](https://jupyter.org/install).

### Evaluate the linkage attack

```bash
cd "Linkage Attack/SGN Based Linkage Attack"

python linkage_attack_eval.py \
    --model models/unfrozen_69.5acc.pt \
    --data path/to/X.pkl \
    --samples_same 200 --samples_diff 200 \
    --segments 20 --max_frames 300 \
    --runs 5 --compute_auc --device cuda
```

### Run the re-identification attackers

```bash
# SGN attack model — raw data, with action classification utility
cd "Attacking Models/SGN Attack Model"
python sgn_attack.py --data data/test_X.pkl --action_data data/X_action.pkl

# same model against UNet/ResNet anonymized data
python sgn_attack.py --data X_unet.pkl --action_data X_unet_action.pkl --source skele_anon

# random forest baseline
cd "../RF Attack Model"
python rf_attack.py --model clf.pkl --data X.pkl --labels Genders.csv
```

Training for both attack models is done in the accompanying notebooks (`SGN.ipynb`, `RF.ipynb`).

## Data

Experiments use the [NTU RGB+D 60+120](https://rose1.ntu.edu.sg/dataset/actionRecognition/) dataset (raw data on [GitHub](https://github.com/shahroudy/NTURGB-D)). LAN is trained on the 40 actors of NTU60 and tested on the 66 unseen actors and 60 unseen action classes of NTU120. Only position information is used, over 25 joints.

`X.pkl` is a dict keyed by actor identifier:

```python
{
  'P001': np.ndarray,   # (videos, 300 frames, 150 = 75 joints x 2 actors)
  'P002': np.ndarray,
  ...
}
```

`X_action.pkl` uses the same layout keyed by action class (`A001` … `A120`). Parsing code for the raw `.skeleton` files lives in `Skeleton Info/`.

All pickle files containing the preprocessed data and saved models are linked from the README inside each relevant directory; the parent folder can be viewed [here](https://drive.google.com/drive/folders/1aO2MU_HQDbxHgdZy6HaMFS0REc7DXtKQ?usp=sharing).

## Citation

```bibtex
@inproceedings{carr2023linkage,
    author    = {Carr, Thomas and Lu, Aidong and Xu, Depeng},
    title     = {Linkage Attack on Skeleton-based Motion Visualization},
    booktitle = {Proceedings of the 32nd ACM International Conference
                 on Information and Knowledge Management (CIKM '23)},
    year      = {2023},
    pages     = {3758--3762},
    publisher = {Association for Computing Machinery},
    address   = {New York, NY, USA},
    doi       = {10.1145/3583780.3615263}
}
```

## Related Work

- [Privacy-centric Motion Retargeting (PMR)](https://pmr.thomasc.tech/) — ICCV 2025, anonymizing skeleton motion while preserving action utility
- [DisentangledTMR](https://tmr.thomasc.tech/) — privacy-preserving skeleton motion retargeting with factorized transformers

## Acknowledgements

This work was supported in part by UNC Charlotte startup fund and NSF grant 1840080.

This code relies on multiple previous works as listed below:

- NTU RGB+D ([Code](https://github.com/shahroudy/NTURGB-D)) ([Paper (60)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shahroudy_NTU_RGBD_A_CVPR_2016_paper.pdf)) ([Paper (120)](https://arxiv.org/pdf/1905.04757.pdf))
- SGN ([Code](https://github.com/microsoft/SGN)) ([Paper](https://arxiv.org/pdf/1904.01189.pdf))
- Skeleton-anonymization ([Code](https://github.com/ml-postech/Skeleton-anonymization)) ([Paper](https://arxiv.org/pdf/2111.15129.pdf))
- 2D MR ([Code](https://github.com/ChrisWu1997/2D-Motion-Retargeting)) ([Paper](https://arxiv.org/pdf/1905.01680.pdf))
- Deep MR ([Code](https://github.com/DeepMotionEditing/deep-motion-editing)) ([Paper](https://arxiv.org/pdf/2005.05732.pdf))
