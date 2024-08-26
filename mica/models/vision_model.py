from numpy.lib.function_base import extract
import torch
import torch.nn as nn

from . import cnn_backbones
from . import ViT
from mica.constants import *
from omegaconf import OmegaConf


class ImageEncoder(nn.Module):
    def __init__(self, cfg):
        super(ImageEncoder, self).__init__()
        self.cfg = cfg
        self.output_dim = cfg.model.text.embedding_dim

        # ViT as backbone
        if "ViT" in cfg.model.vision.model_name:
            self.model, vision_width = ViT.createViT(output_dim=768)
            self.feature_dim = vision_width
            self.extract_layer = nn.Linear(325, 361) 
            # self.model = torch.jit.load(VIT_BASE_16, map_location="cuda")
            # self.visual = self.model.visual
            # self.model.load_state_dict(state_dict)
            if cfg.model.vision.freeze_vit:
                print("Freezing ViT model")
                for param in self.model.parameters():
                    param.requires_grad = False

        # CNN as backbone
        else:
            model_function = getattr(cnn_backbones, cfg.model.vision.model_name)
            self.model, self.feature_dim, self.interm_feature_dim = model_function(
                pretrained=cfg.model.vision.pretrained
            )

            self.global_embedder = nn.Linear(self.feature_dim, self.output_dim)  # resnet: 2048 -> 768
            self.local_embedder = nn.Conv2d(
                self.interm_feature_dim,  # in_channels: 1024
                self.output_dim,          # out_channels:  768
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )

            self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

            if cfg.model.vision.freeze_cnn:
                print("Freezing CNN model")
                for param in self.model.parameters():
                    param.requires_grad = False

    def forward(self, x, get_local=False):
        # --> fixed-size input: batch x 3 x 299 x 299
        if "resnet" in self.cfg.model.vision.model_name:
            global_ft, local_ft = self.resnet_forward(x, extract_features=True)  # ft: feature
        elif "densenet" in self.cfg.model.vision.model_name:
            global_ft, local_ft = self.dense_forward(x, extract_features=True)
        elif "ViT" in self.cfg.model.vision.model_name:
            global_ft, local_ft = self.vit_forward(x)

        if get_local:
            return global_ft, local_ft
        else:
            return global_ft

    def generate_embeddings(self, global_features, local_features):

        # resnet:
        if "resnet" in self.cfg.model.vision.model_name:
            global_emb = self.global_embedder(global_features)
            local_emb = self.local_embedder(local_features)
        elif "ViT" in self.cfg.model.vision.model_name:
            global_emb = global_features
            local_emb = local_features

        return global_emb, local_emb

    def resnet_forward(self, x, extract_features=False):

        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.Upsample(size=(299, 299), mode="bilinear", align_corners=True)(x)  # (batch_size, 3, 299, 299)

        x = self.model.conv1(x)  # (batch_size, 64, 150, 150)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)  # (batch_size, 256, 75, 75)
        x = self.model.layer2(x)  # (batch_size, 512, 38, 38)
        x = self.model.layer3(x)  # (batch_size, 1024, 19, 19)
        local_features = x
        x = self.model.layer4(x)  # (batch_size, 2048, 10, 10)

        x = self.pool(x)          # (batch_size, 2048, 1, 1)
        x = x.view(x.size(0), -1)  # (batch_size, 2048)

        return x, local_features  # global_ft: (batch_size, 2048), local_ft: (batch_size, 1024, 19, 19)

    def densenet_forward(self, x, extract_features=False):
        pass

    def vit_forward(self, x):
        # input: (batch_size, 3, 299, 299)
        x = nn.Upsample(size=(299, 299), mode="bilinear", align_corners=True)(x)

        x = self.model.conv1(x)  # shape = [*, width, grid, grid]  (batch_size, 768, 18, 18)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2] (batch_size, 768, 324)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width] (batch_size, 324, 768)
        x = torch.cat(
            [self.model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width] (batch_size, 325, 768)
        x = x + self.model.positional_embedding.to(x.dtype)  # (batch_size, 325, 768)
        x = self.model.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND (325, batch_size, 768)
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD (batch_size, 325, 768)

        # NOTE: new add to extract local features
        local_ft = x.permute(0, 2, 1)  # (batch_size, 768, 325)
        local_ft = self.extract_layer(local_ft)  # (batch_size, 768, 361)
        # (batch_size, 768, 19, 19)
        local_ft = local_ft.reshape(local_ft.shape[0], local_ft.shape[1], int(local_ft.shape[2] ** 0.5), int(local_ft.shape[2] ** 0.5))

        x = self.model.ln_post(x[:, 0, :])  # (batch_size, width) (batch_size, 768)

        if self.model.proj is not None:
            x = x @ self.model.proj

        global_ft = x  # (batch_size, output_dim)

        return global_ft, local_ft  # global_ft: (batch_size, 768), local_ft: (batch_size, 768, 19, 19)

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)


class PretrainedImageClassifier(nn.Module):
    def __init__(
        self,
        image_encoder: nn.Module,
        num_cls: int,
        feature_dim: int,
        freeze_encoder: bool = True,
    ):
        super(PretrainedImageClassifier, self).__init__()
        self.img_encoder = image_encoder
        self.classifier = nn.Linear(feature_dim, num_cls)

        if freeze_encoder:
            for param in self.img_encoder.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.img_encoder(x)
        pred = self.classifier(x)
        return pred


class ImageClassifier(nn.Module):
    def __init__(self, cfg, image_encoder=None):
        super(ImageClassifier, self).__init__()

        model_function = getattr(cnn_backbones, cfg.model.vision.model_name)
        self.img_encoder, self.feature_dim, _ = model_function(
            pretrained=cfg.model.vision.pretrained
        )

        self.classifier = nn.Linear(self.feature_dim, cfg.model.vision.num_targets)

    def forward(self, x):
        x = self.img_encoder(x)
        pred = self.classifier(x)
        return pred
