import torch
import torch.nn as nn
import cv2
import re
import numpy as np
from sklearn import metrics
import os
import pickle

from PIL import Image
from .. import builder
from .. import loss
from .. import utils
from transformers import AutoTokenizer
from nltk.tokenize import RegexpTokenizer
from mica.constants import *


class MICA(nn.Module):
    def __init__(self, cfg):
        super(MICA, self).__init__()

        self.cfg = cfg
        self.text_encoder = builder.build_text_model(cfg)  # BioClinical BERT
        self.img_encoder = builder.build_img_model(cfg)    # ResNet-50

        self.local_loss = loss.mica_loss.local_loss
        self.global_loss = loss.mica_loss.global_loss
        self.local_loss_weight = self.cfg.model.mica.local_loss_weight
        self.global_loss_weight = self.cfg.model.mica.global_loss_weight

        # self.concept_encoder = nn.Sequential(  
        #     nn.Linear(cfg.model.text.embedding_dim * cfg.data.text.word_num, 22),
        #     nn.Tanh())

        self.concept_encoder = nn.Sequential(
            nn.Linear(cfg.model.text.embedding_dim * cfg.data.text.word_num, 2048),
            nn.Tanh())
        self.concept_loss = nn.CrossEntropyLoss()
        self.concept_loss_weight = self.cfg.model.mica.concept_loss_weight
        all_concepts = pickle.load(open(CAV_FILE, 'rb'))
        all_concept_names = list(all_concepts.keys())
        print(f"Bank path: {CAV_FILE}. {len(all_concept_names)} concepts will be used.")
        self.concept_bank = utils.ConceptBank(all_concepts, "cuda:0")  # TODO: remove device
        self.cavs = self.concept_bank.vectors
        self.intercepts = self.concept_bank.intercepts
        self.norms = self.concept_bank.norms

        self.temp1 = self.cfg.model.mica.temp1
        self.temp2 = self.cfg.model.mica.temp2
        self.temp3 = self.cfg.model.mica.temp3
        self.batch_size = self.cfg.train.batch_size

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.text.bert_type)
        self.ixtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

    def text_encoder_forward(self, caption_ids, attention_mask, token_type_ids):

        text_emb_l, text_emb_g, sents = self.text_encoder(
            caption_ids, attention_mask, token_type_ids
        )
        return text_emb_l, text_emb_g, sents

    def image_encoder_forward(self, imgs):

        img_feat_g, img_feat_l = self.img_encoder(imgs, get_local=True)
        img_emb_g, img_emb_l = self.img_encoder.generate_embeddings(
            img_feat_g, img_feat_l
        )

        return img_emb_l, img_emb_g

    def concept_encoder_forward(self, img_emb_l, text_emb_l):
        
        weighted_represent = self.get_weighted_representation(img_emb_l, text_emb_l)
        predict_concepts = self.concept_encoder(weighted_represent)

        predict_concepts = (torch.matmul(self.cavs, predict_concepts.T) + self.intercepts) / self.norms
        return predict_concepts.T

    def _calc_local_loss(self, img_emb_l, text_emb_l, sents):

        cap_lens = [
            len([w for w in sent if not w.startswith("[")]) + 1 for sent in sents
        ]
        l_loss0, l_loss1, _ = self.local_loss(
            img_emb_l,
            text_emb_l,
            cap_lens,
            temp1=self.temp1,
            temp2=self.temp2,
            temp3=self.temp3,
        )
        return l_loss0, l_loss1

    def _calc_global_loss(self, img_emb_g, text_emb_g):
        g_loss0, g_loss1 = self.global_loss(img_emb_g, text_emb_g, temp3=self.temp3)
        return g_loss0, g_loss1

    def _calc_concept_loss(self, predict_concepts, concept_labels):
        concept_loss = self.concept_loss(predict_concepts, concept_labels)
        return concept_loss

    def calc_loss(self, img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents, predict_concepts, concept_labels):

        l_loss0, l_loss1 = self._calc_local_loss(
            img_emb_l, text_emb_l, sents
        )
        g_loss0, g_loss1 = self._calc_global_loss(img_emb_g, text_emb_g)

        concept_loss = self._calc_concept_loss(predict_concepts, concept_labels)

        # weighted loss
        loss = 0
        loss += (l_loss0 + l_loss1) * self.local_loss_weight
        loss += (g_loss0 + g_loss1) * self.global_loss_weight 
        loss += concept_loss * self.concept_loss_weight # TODO: 

        return loss

    def forward(self, x):

        # img encoder branch
        img_emb_l, img_emb_g = self.image_encoder_forward(x["imgs"])

        # text encoder branch
        text_emb_l, text_emb_g, sents = self.text_encoder_forward(
            x["caption_ids"], x["attention_mask"], x["token_type_ids"]
        )

        predict_concepts = self.concept_encoder_forward(img_emb_l, text_emb_l)
        concept_labels = x["concept_labels"]

        return img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents, predict_concepts, concept_labels

    def get_global_similarities(self, img_emb_g, text_emb_g):
        img_emb_g = img_emb_g.detach().cpu().numpy()
        text_emb_g = text_emb_g.detach().cpu().numpy()
        global_similarities = metrics.pairwise.cosine_similarity(img_emb_g, text_emb_g)
        global_similarities = torch.Tensor(global_similarities)
        return global_similarities

    def get_local_similarities(self, img_emb_l, text_emb_l, cap_lens):

        batch_size = img_emb_l.shape[0]
        similarities = []

        for i in range(len(text_emb_l)):
            words_num = 14 
            word = (
                text_emb_l[i, :, 1 : words_num + 1].unsqueeze(0).contiguous()
            )  # [1, 768, 25]

            word = word.repeat(batch_size, 1, 1)  # [48, 768, 25]
            context = img_emb_l  # [48, 768, 19, 19]

            weiContext, attn = loss.mica_loss.attention_fn(
                word, context, 4.0
            )  # [48, 768, 25], [48, 25, 19, 19]

            word = word.transpose(1, 2).contiguous()  # [48, 25, 768]
            weiContext = weiContext.transpose(1, 2).contiguous()  # [48, 25, 768]

            word = word.view(batch_size * words_num, -1)  # [1200, 768]
            weiContext = weiContext.view(batch_size * words_num, -1)  # [1200, 768]
            #
            row_sim = loss.mica_loss.cosine_similarity(word, weiContext)
            row_sim = row_sim.view(batch_size, words_num)  # [48, 25]

            row_sim.mul_(5.0).exp_()
            row_sim, max_row_idx = torch.max(row_sim, dim=1, keepdim=True)  # [48, 1]

            row_sim = torch.log(row_sim)

            similarities.append(row_sim)

        local_similarities = torch.cat(similarities, 1).detach().cpu()

        return local_similarities

    def get_weighted_representation(self, img_emb_l, text_emb_l):
        batch_size = img_emb_l.shape[0]
        weighted_contexts = []

        for i in range(text_emb_l.shape[0]):
            # words_num = cap_lens[i]
            # word = text_emb_l[i, :, :words_num].unsqueeze(0).contiguous()
            word = text_emb_l[i].unsqueeze(0).contiguous()  # NOTE: fix the dimension of word (caption)
            word = word.repeat(batch_size, 1, 1)
            context = img_emb_l

            weiContext, attn = loss.mica_loss.attention_fn(
                word, context, temp1=4.0
            )  # (batch_size, 768, words_num)
            weighted_contexts.append(weiContext)

        # average the weighted contexts and get the weighted representation
        weighted_represent = torch.stack(weighted_contexts, dim=0).mean(dim=0)
        return weighted_represent.view(batch_size, text_emb_l.shape[1] * text_emb_l.shape[2])

    def process_text(self, text, device):

        if type(text) == str:
            text = [text]

        processed_text_tensors = []
        for t in text:
            # use space instead of newline
            t = t.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(t)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            all_sents = []

            for t in captions:
                t = t.replace("\ufffd\ufffd", " ")
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(t.lower())

                if len(tokens) <= 1:
                    continue

                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                all_sents.append(" ".join(included_tokens))

            t = " ".join(all_sents)

            text_tensors = self.tokenizer(
                t,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.cfg.data.text.word_num,
            )
            text_tensors["sent"] = [
                self.ixtoword[ix] for ix in text_tensors["input_ids"][0].tolist()
            ]
            processed_text_tensors.append(text_tensors)

        caption_ids = torch.stack([x["input_ids"] for x in processed_text_tensors])
        attention_mask = torch.stack(
            [x["attention_mask"] for x in processed_text_tensors]
        )
        token_type_ids = torch.stack(
            [x["token_type_ids"] for x in processed_text_tensors]
        )

        if len(text) == 1:
            caption_ids = caption_ids.squeeze(0).to(device)
            attention_mask = attention_mask.squeeze(0).to(device)
            token_type_ids = token_type_ids.squeeze(0).to(device)
        else:
            caption_ids = caption_ids.squeeze().to(device)
            attention_mask = attention_mask.squeeze().to(device)
            token_type_ids = token_type_ids.squeeze().to(device)

        cap_lens = []
        for txt in text:
            cap_lens.append(len([w for w in txt if not w.startswith("[")]))

        return {
            "caption_ids": caption_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "cap_lens": cap_lens,
        }

    def process_class_prompts(self, class_prompts, device):

        cls_2_processed_txt = {}
        for k, v in class_prompts.items():
            cls_2_processed_txt[k] = self.process_text(v, device)

        return cls_2_processed_txt
    

    def process_img(self, paths, device):

        transform = builder.build_transformation(self.cfg, split="test")

        if type(paths) == str:
            paths = [paths]

        all_imgs = []
        for p in paths:

            x = cv2.imread(str(p), 0)

            # tranform images
            x = self._resize_img(x, self.cfg.data.image.imsize)
            img = Image.fromarray(x).convert("RGB")
            img = transform(img)
            all_imgs.append(torch.tensor(img))

        all_imgs = torch.stack(all_imgs).to(device)

        return all_imgs


    def _resize_img(self, img, scale):
        """
        Args:
            img - image as numpy array (cv2)
            scale - desired output image-size as scale x scale
        Return:
            image resized to scale x scale with shortest dimension 0-padded
        """
        size = img.shape
        max_dim = max(size)
        max_ind = size.index(max_dim)

        # Resizing
        if max_ind == 0:
            # image is heigher
            wpercent = scale / float(size[0])
            hsize = int((float(size[1]) * float(wpercent)))
            desireable_size = (scale, hsize)
        else:
            # image is wider
            hpercent = scale / float(size[1])
            wsize = int((float(size[0]) * float(hpercent)))
            desireable_size = (wsize, scale)
        resized_img = cv2.resize(
            img, desireable_size[::-1], interpolation=cv2.INTER_AREA
        )  # this flips the desireable_size vector

        # Padding
        if max_ind == 0:
            # height fixed at scale, pad the width
            pad_size = scale - resized_img.shape[1]
            left = int(np.floor(pad_size / 2))
            right = int(np.ceil(pad_size / 2))
            top = int(0)
            bottom = int(0)
        else:
            # width fixed at scale, pad the height
            pad_size = scale - resized_img.shape[0]
            top = int(np.floor(pad_size / 2))
            bottom = int(np.ceil(pad_size / 2))
            left = int(0)
            right = int(0)
        resized_img = np.pad(
            resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
        )

        return resized_img
