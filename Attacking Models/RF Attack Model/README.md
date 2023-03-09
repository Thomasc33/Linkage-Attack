# Training

Training is done inside of a jupyter notebook `RF.ipynb`

# Testing

```bash
python rf_attack.py --model "clf pickle name" --data "path/to/X.pkl"
```
X.pkl can be generated via NTU in ["RF.ipynb"](https://github.com/Thomasc33/Motion-Privacy/blob/main/Attacking%20Models/RF%20Attack%20Model/RF.ipynb) or from my fork of [Skeleton-anomymization](https://github.com/Thomasc33/Skeleton-anonymization) *(add --source skele_anon if using X_unet/X_resnet from there)*

If using a different dataset, X should be a dict where the key is the actor identifier. The identifier should have a label similar to that seen in "./data/Genders.csv", provided with the `--labels file.csv` flag

All the pickled files made from training are available [here](https://drive.google.com/drive/folders/1XovBZG3Qh8guIKf1-UMpfVG3l_efQt1Z?usp=sharing)