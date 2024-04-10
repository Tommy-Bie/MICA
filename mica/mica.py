import os
import torch
import numpy as np
import copy
import random
import pandas as pd

from . import builder
from . import utils
from . import constants
from .models.vision_model import PretrainedImageClassifier
from typing import Union, List


np.random.seed(6)
random.seed(6)


_MODELS = {
    # NOTE: the path of pretrained model (the first stage)
    "mica_resnet50": None
}


_FEATURE_DIM = {"mica_resnet50": 2048, "mica_resnet18": 2048, "mica_ViT": 768}


def available_models() -> List[str]:
    """Returns the names of available models"""
    return list(_MODELS.keys())


def load_mica(
    name: str = "mica_resnet50",
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
):

    # warnings
    if name in _MODELS:
        ckpt_path = _MODELS[name]
    elif os.path.isfile(name):
        ckpt_path = name
    else:
        raise RuntimeError(
            f"Model {name} not found; available models = {available_models()}"
        )

    if not os.path.exists(ckpt_path):
        raise RuntimeError(
            f"Model {name} not found.\n"
            + "Please check the pretrained weights. \n"
        )

    ckpt = torch.load(ckpt_path, map_location='cpu')
    cfg = ckpt["hyper_parameters"]
    ckpt_dict = ckpt["state_dict"]

    fixed_ckpt_dict = {}
    for k, v in ckpt_dict.items():
        new_key = k.split("mica.")[-1]
        fixed_ckpt_dict[new_key] = v
    ckpt_dict = fixed_ckpt_dict

    mica_model = builder.build_mica_model(cfg).to(device)
    mica_model.load_state_dict(ckpt_dict)

    return mica_model


def load_img_classification_model(
    name: str = "mica_resnet50",
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    num_cls: int = 1,
    freeze_encoder: bool = True,
):

    # load pretrained image encoder
    mica_model = load_mica(name, device)
    image_encoder = copy.deepcopy(mica_model.img_encoder)
    del mica_model

    # create image classifier
    feature_dim = _FEATURE_DIM[name]
    img_model = PretrainedImageClassifier(
        image_encoder, num_cls, feature_dim, freeze_encoder
    )

    return img_model



def get_similarities(mica_model, imgs, txts, similarity_type="both"):

    # warnings
    if similarity_type not in ["global", "local", "both"]:
        raise RuntimeError(
            f"similarity type should be one of ['global', 'local', 'both']"
        )
    if type(txts) == str or type(txts) == list:
        raise RuntimeError(
            f"Text input not processed - please use mica_model.process_text"
        )
    if type(imgs) == str or type(imgs) == list:
        raise RuntimeError(
            f"Image input not processed - please use mica_model.process_img"
        )

    # get global and local image features
    with torch.no_grad():
        img_emb_l, img_emb_g = mica_model.image_encoder_forward(imgs)
        text_emb_l, text_emb_g, _ = mica_model.text_encoder_forward(
            txts["caption_ids"], txts["attention_mask"], txts["token_type_ids"]
        )

    # get similarities
    global_similarities = mica_model.get_global_similarities(img_emb_g, text_emb_g)
    local_similarities = mica_model.get_local_similarities(
        img_emb_l, text_emb_l, txts["cap_lens"]
    )
    similarities = (local_similarities + global_similarities) / 2

    if similarity_type == "global":
        return global_similarities.detach().cpu().numpy()
    elif similarity_type == "local":
        return local_similarities.detach().cpu().numpy()
    else:
        return similarities.detach().cpu().numpy()

