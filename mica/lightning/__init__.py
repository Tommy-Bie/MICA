from .pretrain_model import PretrainModel
from .classification_model import ClassificationModel, ClassificationModelCBM


LIGHTNING_MODULES = {
    "pretrain": PretrainModel,
    "classification": ClassificationModel,

    "classification_cbm": ClassificationModelCBM,
}
