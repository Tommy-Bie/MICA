## Implementation of MICA.

### Usage
The first stage is multi-level image-concept alignment and the second stage is explainable disease diagnosis. Please set the model path in `mica.py` after the first stage training.
```
# first stage training
python run.py \
-c ./configs/[dataset_pretrain_config_sample].yaml \
--train

# second stage training
python run.py \
-c ./configs/[dataset_classification_CBM_config_sample].yaml \
--train
```

### NOTE
The sample configurations and dataset loading logics are under the configs and datasets folder, respectively. Please adapt to custom dataset by replacing the dataset name and path in configuration files (.yaml) and `constants.py`. You can create the dataset loading and processing logic by modifying `pretraining_dataset_derm.py` and `image_dataset_derm.py` under the datasets folder. The class activation vecotor (CAV) file can be created by following the work [here](https://github.com/mertyg/post-hoc-cbm). 