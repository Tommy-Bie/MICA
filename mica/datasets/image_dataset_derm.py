import pandas as pd
import torch
import os
from PIL import Image
from torchvision import transforms

from .image_dataset import ImageBaseDataset
from mica.constants import *


class SampleDataset(ImageBaseDataset):
    """ Sample dataset, modify to adapt to custom dataset """

    def __init__(self, cfg, split="train", transform=None):
        self.cfg = cfg
        self.base_dir = os.path.join(DATA_FOLER, "images")
        self.df_meta = pd.read_csv(META)

        if split == "train":
            self.df_meta = pd.read_csv(META_TRAIN)
        elif split == "valid":
            self.df_meta = pd.read_csv(META_VALID)
        elif split == "test":
            self.df_meta = pd.read_csv(META_TEST)

        self.resize = transforms.Resize((self.cfg.data.image.imsize, self.cfg.data.image.imsize))
        super(SampleDataset, self).__init__(cfg, split, transform)

    def __getitem__(self, idx):
        # sample logic
        if int(self.df_meta['labels'].iloc[idx]) == 0:
            label = torch.tensor([1, 0])
        elif int(self.df_meta['labels'].iloc[idx]) == 1:
            label = torch.tensor([0, 1])
        label = label.float()
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.df_meta.iloc[idx]
        img_path = os.path.join(self.base_dir, row["img_path"])
        image = Image.open(img_path).convert('RGB')
        image = self.resize(image) 
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.df_meta)


class SampleDatasetCBM(ImageBaseDataset):

    def __init__(self, cfg, split="train", transform=None):
        self.cfg = cfg
        self.base_dir = os.path.join(DATA_FOLER, "images")

        if split == "train":
            self.df_meta = pd.read_csv(META_TRAIN)
        elif split == "valid":
            self.df_meta = pd.read_csv(META_VALID)
        elif split == "test":
            self.df_meta = pd.read_csv(META_TEST)

        self.resize = transforms.Resize((self.cfg.data.image.imsize, self.cfg.data.image.imsize))
        super(SampleDatasetCBM, self).__init__(cfg, split, transform)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.df_meta.iloc[idx]
        img_path = os.path.join(self.base_dir, row["img_path"])
        image = Image.open(img_path).convert('RGB')
        image = self.resize(image) 
        if self.transform:
            image = self.transform(image)
        label = {}
        diag_label = None
        if int(self.df_meta['labels'].iloc[idx]) == 0:
            diag_label = torch.tensor([1, 0]).float()
        elif int(self.df_meta['labels'].iloc[idx]) == 1:
            diag_label = torch.tensor([0, 1]).float()
        label["diag"] = diag_label
        label["concept"] = self.get_concept_label(idx)
        return image, label

    def __len__(self):
        return len(self.df_meta)

    def get_concept_label(self, idx):
        row = self.df_meta.iloc[idx]
        concept_label = []
        concepts = row["concept_first":"concept_final"]  # NOTE: modify to adapt to your dataset
        for concept in concepts:
            concept_label.append(concept)
        concept_label = torch.Tensor(concept_label).float()
        return concept_label
