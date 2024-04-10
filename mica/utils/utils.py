"""Adapted from: https://github.com/mrlibw/ControlGAN"""

import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict


def normalize(similarities, method="norm"):

    if method == "norm":
        return (similarities - similarities.mean(axis=0)) / (similarities.std(axis=0))
    elif method == "standardize":
        return (similarities - similarities.min(axis=0)) / (
            similarities.max(axis=0) - similarities.min(axis=0)
        )
    else:
        raise Exception("normalizing method not implemented")


# CAV
class ConceptBank:
    def __init__(self, concept_dict, device):
        all_vectors, concept_names, all_intercepts = [], [], []
        all_margin_info = defaultdict(list)
        for k, (tensor, _, _, intercept, margin_info) in concept_dict.items():
            all_vectors.append(tensor)
            concept_names.append(k)
            all_intercepts.append(np.array(intercept).reshape(1, 1))
            for key, value in margin_info.items():
                if key != "train_margins":
                    all_margin_info[key].append(np.array(value).reshape(1, 1))
        for key, val_list in all_margin_info.items():
            margin_tensor = torch.tensor(np.concatenate(
                val_list, axis=0), requires_grad=False).float().to(device)
            all_margin_info[key] = margin_tensor

        self.concept_info = EasyDict()
        self.concept_info.margin_info = EasyDict(dict(all_margin_info))
        self.concept_info.vectors = torch.tensor(np.concatenate(all_vectors, axis=0), requires_grad=False).float().to(
            device)
        self.concept_info.norms = torch.norm(
            self.concept_info.vectors, p=2, dim=1, keepdim=True).detach()
        self.concept_info.intercepts = torch.tensor(np.concatenate(all_intercepts, axis=0),
                                                    requires_grad=False).float().to(device)
        self.concept_info.concept_names = concept_names
        print("Concept Bank is initialized.")

    def __getattr__(self, item):
        return self.concept_info[item]


class EasyDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__