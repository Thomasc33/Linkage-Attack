# Linkage-Attack

Code accompanying [Linkage Attack on Skeleton-based Motion Visualization](https://dl.acm.org/doi/10.1145/3583780.3615263) CIKM'23

## Attacking Models

Contains the attacking models. Split into the Random Forest and the SGN-based attacking models.

## Defense Models

Contains the classical Motion Retargeting anonymizer model. The other anonymizer models utilized are naive adaptations from the following repos: [UNET/ResNet](https://github.com/ml-postech/Skeleton-anonymization), [Deep MR](https://github.com/ChrisWu1997/2D-Motion-Retargeting)

## Linkage Attack

Contains the SGN-based (Main) and RF (Baseline) versions of the Linkage Attack proposed in this paper.

## Skeleton Info

Contains code used to parse and process the .skeleton files provided by the NTU paper [here](https://rose1.ntu.edu.sg/dataset/actionRecognition/). Raw data is available [here](https://github.com/shahroudy/NTURGB-D)

# Setup

All contributions were written inside Jupyter Notebooks but have been exported to standalone Python files. Details on installing Jupyter can be found [Here](https://jupyter.org/install).

Specifics on how to run each file are located inside the README in each relevant directory.

# Data

All pickle files containing the preprocessed data and saved models are located inside the README file inside each relevant directory. Additionally, the parent folder can be viewed [Here](https://drive.google.com/drive/folders/1aO2MU_HQDbxHgdZy6HaMFS0REc7DXtKQ?usp=sharing).

# Acknowledgements

This work was supported in part by UNC Charlotte startup fund and NSF grant 1840080.

This code relies on multiple previous works as listed below:

NTU RGB+D ([Code](https://github.com/shahroudy/NTURGB-D)) ([Paper (60)](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shahroudy_NTU_RGBD_A_CVPR_2016_paper.pdf)) ([Paper (120)](https://arxiv.org/pdf/1905.04757.pdf))

SGN ([Code](https://github.com/microsoft/SGN)) ([Paper](https://arxiv.org/pdf/1904.01189.pdf))

Skeleton-anonymization ([Code](https://github.com/ml-postech/Skeleton-anonymization)) ([Paper](https://arxiv.org/pdf/2111.15129.pdf))

2D MR ([Code](https://github.com/ChrisWu1997/2D-Motion-Retargeting)) ([Paper](https://arxiv.org/pdf/1905.01680.pdf))

Deep MR ([Code](https://github.com/DeepMotionEditing/deep-motion-editing)) ([Paper](https://arxiv.org/pdf/2005.05732.pdf))
