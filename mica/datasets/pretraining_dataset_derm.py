import re
import os
import numpy as np
import pandas as pd
import cv2
import tqdm
import pickle
import numpy.random as random
import torch
import torch.utils.data as data
from torchvision import transforms

from PIL import Image
from nltk.tokenize import RegexpTokenizer
from transformers import AutoTokenizer
from mica.constants import *


class MultimodalPretrainingDataset(data.Dataset):
    """ Pretraining Dataset """

    def __init__(self, cfg, split="train", transform=None):

        if DATA_FOLER is None:
            raise RuntimeError(
                "Data path empty\n"
                + "make sure to update your data path in constants.py"
            )

        self.cfg = cfg

        if transform is not None:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.cfg.data.image.imsize, self.cfg.data.image.imsize)),
                                              ])

        self.resize = transforms.Compose([
                transforms.Resize((self.cfg.data.image.imsize, self.cfg.data.image.imsize)),
                                              ])
        self.max_word_num = self.cfg.data.text.captions_per_image

        self.meta_df = pd.read_csv(META)

        if split == "train":
            self.meta_df = pd.read_csv(META_TRAIN)

        elif split == "valid":
            self.meta_df = pd.read_csv(META_VALID)
        elif split == "test":
            self.meta_df = pd.read_csv(META_TEST)

        # load text mapping
        self.path2sent = self.load_text_data(split)

        # create BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)

    def __getitem__(self, idx):
        row = self.meta_df.iloc[idx]
        img_path = os.path.join(DATA_FOLER, row["img_path"])
        image = Image.open(img_path).convert('RGB')
        image = self.resize(image)
        if self.transform:
            image = self.transform(image)

        cap, cap_len = self.get_caption(row["img_path"])

        label = row["labels"]
        if label < 1:
            label = torch.tensor([1, 0])
        elif label >= 1:
            label = torch.tensor([0, 1])

        return image, cap, cap_len, img_path, label.float()

    def __len__(self):
        return self.meta_df.shape[0]

    def load_text_data(self, split):

        # get text mapping
        filepath = "out/captions_dataset.pickle"  # NOTE: modify dataset name
        if not os.path.isfile(filepath):
            print(f"Caption file {filepath} does not exit. Creating captions...")
            path2sent = self.create_text_mapping(
                self.meta_df, self.max_word_num
            )
            with open(filepath, "wb") as f:
                pickle.dump(path2sent, f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                print(f"Loading captions from {filepath}")
                path2sent = pickle.load(f)

        return path2sent

    def create_text_mapping(self, meta_df, max_word_num):
        """ tokenize the report in meta_df"""
        sent_lens, num_sents = [], []
        text_mapping = {}

        for idx, row in tqdm.tqdm(meta_df.iterrows(), total=meta_df.shape[0]):
            captions = ""
            captions += row["report"]

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(",") for point in captions]  # NOTE: split by comma
            captions = [sent for point in captions for sent in point]

            count = 0
            mapping_sent = []

            # create tokens from captions
            for cap in captions:
                cap = cap.replace("\ufffd\ufffd", " ")

                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                # if len(tokens) <= 1:
                #     continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                mapping_sent.append(" ".join(included_tokens))

                # check if reached maximum number of words in the sentences
                count += len(included_tokens)
                if count == max_word_num:
                    break

                sent_lens.append(len(included_tokens))
            num_sents.append(len(mapping_sent))
            text_mapping[row["derm"]] = mapping_sent

        # get report word/sentence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)
        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return text_mapping

    def get_caption(self, img_path):

        sents = self.path2sent[img_path]

        if len(sents) == 0:
            print(img_path)
            raise Exception("no sentence for path")

        if self.cfg.data.text.full_report is True:
            sent = " ".join(sents)
        else:
            sent_ix = random.randint(0, len(sents))
            sent = sents[sent_ix]

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.cfg.data.text.word_num,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len


def multimodal_collate_fn(batch):
    """ sort sequence """

    imgs, cap_len, ids, tokens, attention, paths = [], [], [], [], [], []
    labels = []

    # flatten
    for b in batch:
        img, cap, cal_l, path, label = b
        imgs.append(img)
        cap_len.append(cal_l)
        ids.append(cap["input_ids"])
        tokens.append(cap["token_type_ids"])
        attention.append(cap["attention_mask"])
        paths.append(path)
        labels.append(label)

    # stack
    imgs = torch.stack(imgs)
    ids = torch.stack(ids).squeeze()
    tokens = torch.stack(tokens).squeeze()
    attention = torch.stack(attention).squeeze()
    labels = torch.stack(labels).squeeze()

    # sort and add to dictionary
    sorted_cap_lens, sorted_cap_indices = torch.sort(torch.tensor(cap_len), 0, True)
    return_dict = {
        "caption_ids": ids[sorted_cap_indices],
        "token_type_ids": tokens[sorted_cap_indices],
        "attention_mask": attention[sorted_cap_indices],
        "imgs": imgs[sorted_cap_indices],
        "cap_lens": sorted_cap_lens,
        "path": paths,
        "labels": labels,
    }

    return return_dict
