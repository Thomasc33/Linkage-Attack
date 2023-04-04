# Training

Training is done inside of a jupyter notebook `SGN.ipynb`

# Testing

```bash
python sgn_attack.py --data "path/to/X.pkl"
```
X.pkl can be generated via NTU in ["../RF Attack Model/RF.ipynb"](https://github.com/Thomasc33/Motion-Privacy/blob/main/Attacking%20Models/RF%20Attack%20Model/RF.ipynb) or from my fork of [Skeleton-anomymization](https://github.com/Thomasc33/Skeleton-anonymization) *(add --source skele_anon if using X_unet/X_resnet from there)*

If using a different dataset, X should be a dict where the key is the actor identifier. The identifier should have a label similar to that seen in "./data/Genders.csv", provided with the `--labels file.csv` flag

All the pickled files made from training are available [here](https://drive.google.com/drive/folders/1lgxskKv47i1uVt-2AmwVE1VQCY0MuQCM?usp=share_link) *(nothing available atm)*


### Action Classification + Privacy testing

Add the `--action_data` flag, and provide the path to X_action.pkl. No default is set, but I had it in `data/X_action.pkl`


### Commands I used for evalutation
```bash
# Raw Data
python sgn_attack.py --data "data/test_X.pkl" --action_data "data/X_action.pkl"

# UNet
python sgn_attack.py --data "C:/Users/Carrt/OneDrive/Code/Motion Privacy/External Repositories/Skeleton-anonymization/X_unet.pkl" --action_data "C:/Users/Carrt/OneDrive/Code/Motion Privacy/External Repositories/Skeleton-anonymization/X_unet_action.pkl" --source skele_anon

# ResNet
python sgn_attack.py --data "C:/Users/Carrt/OneDrive/Code/Motion Privacy/External Repositories/Skeleton-anonymization/X_resnet.pkl" --action_data "C:/Users/Carrt/OneDrive/Code/Motion Privacy/External Repositories/Skeleton-anonymization/X_resnet_action.pkl" --source skele_anon

# Classical Motion Retargeting
python sgn_attack.py --data "C:/Users/Carrt/OneDrive/Code/Motion Privacy/Defense Models/Mean Skeleton/X_SGN_normalized.pkl" --action_data "C:/Users/Carrt/OneDrive/Code/Motion Privacy/Defense Models/Mean Skeleton/X_SGN_AR_normalized.pkl"
```