import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import copy

from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, accuracy_score
from torchmetrics import F1, Accuracy, Recall, Precision, Specificity
from .. import builder
from .. import mica
from pytorch_lightning.core import LightningModule


class ClassificationModel(LightningModule):
    """Pytorch-Lightning Module"""

    def __init__(self, cfg):
        """Pass in hyperparameters to the model"""
        # initalize superclass
        super().__init__()

        self.cfg = cfg

        if self.cfg.model.vision.model_name in mica.available_models():
            if "resnet50" in self.cfg.model.vision.model_name:
                self.model = mica.load_img_classification_model(
                    self.cfg.model.vision.model_name,
                    num_cls=self.cfg.model.vision.num_targets,
                    freeze_encoder=self.cfg.model.vision.freeze_cnn,
                )
            elif "ViT" in self.cfg.model.vision.model_name:
                self.model = mica.load_img_classification_model(
                    self.cfg.model.vision.model_name,
                    num_cls=self.cfg.model.vision.num_targets,
                    freeze_encoder=self.cfg.model.vision.freeze_vit,
                )
        else:
            self.model = builder.build_img_model(cfg)

        self.loss = builder.build_loss(cfg)
        self.lr = cfg.lightning.trainer.lr
        self.dm = None

    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.lr, self.model)
        scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def training_epoch_end(self, training_step_outputs):
        return self.shared_epoch_end(training_step_outputs, "train")

    def validation_epoch_end(self, validation_step_outputs):
        return self.shared_epoch_end(validation_step_outputs, "val")

    def test_epoch_end(self, test_step_outputs):
        return self.shared_epoch_end(test_step_outputs, "test")

    def shared_step(self, batch, split):
        """Similar to training step"""

        x, y = batch

        logit = self.model(x)
        loss = self.loss(logit, y)

        log_iter_loss = True if split == "train" else False
        self.log(
            f"{split}_loss",
            loss,
            on_epoch=True,
            on_step=log_iter_loss,
            logger=True,
            prog_bar=True,
        )

        return_dict = {"loss": loss, "logit": logit, "y": y}
        return return_dict

    def shared_epoch_end(self, step_outputs, split):
        logit = torch.cat([x["logit"] for x in step_outputs])
        y = torch.cat([x["y"] for x in step_outputs])
        prob = torch.sigmoid(logit)

        y = y.detach().cpu()
        prob = prob.detach().cpu()

        y = y.numpy()
        prob = prob.numpy()
        indexes = np.argmax(prob, axis=-1)
        prob_pred = np.zeros_like(prob)
        for i in range(len(indexes)):
            prob_pred[i][indexes[i]] = 1
        f1_micro = f1_score(y, prob_pred, average="micro")
        f1_weighted = f1_score(y, prob_pred, average="weighted")
        f1_samples = f1_score(y, prob_pred, average="samples")
        f1_macro = f1_score(y, prob_pred, average="macro")

        auroc = roc_auc_score(y, prob)

        self.log(f"{split}_auroc", auroc, on_epoch=True, logger=True, prog_bar=True)
        self.log(f"{split}_f1_score_micro", f1_micro, on_epoch=True, logger=True, prog_bar=True)
        self.log(f"{split}_f1_score_weighted", f1_weighted, on_epoch=True, logger=True, prog_bar=True)
        self.log(f"{split}_f1_score_samples", f1_samples, on_epoch=True, logger=True, prog_bar=True)
        self.log(f"{split}_f1_score_macro", f1_macro, on_epoch=True, logger=True, prog_bar=True)


    def calculateTop1(self, logits, labels):
        with torch.no_grad():
            labels = labels.argmax(dim=1)
            pred = logits.argmax(dim=1)
            return torch.eq(pred, labels).sum().float().item() / labels.size(0)


class ClassificationModelCBM(LightningModule):
    """ Classification model in the manner of Concept Bottleneck Models (CBM) """

    def __init__(self, cfg):
        """Pass in hyperparameters to the model"""
        # initalize superclass
        super().__init__()

        self.cfg = cfg

        if self.cfg.model.vision.model_name in mica.available_models():
            if "resnet50" in self.cfg.model.vision.model_name:
                self.model = mica.load_img_classification_model(
                    self.cfg.model.vision.model_name,
                    num_cls=self.cfg.model.vision.num_targets,
                    freeze_encoder=self.cfg.model.vision.freeze_cnn,
                )
            elif "ViT" in self.cfg.model.vision.model_name:
                self.model = mica.load_img_classification_model(
                    self.cfg.model.vision.model_name,
                    num_cls=self.cfg.model.vision.num_targets,
                    freeze_encoder=self.cfg.model.vision.freeze_vit,
                )
        else:
            self.model = builder.build_img_model(cfg)

        self.loss = builder.build_loss(cfg)
        self.lr = cfg.lightning.trainer.lr
        self.dm = None

        # NOTE: use different classes for different datasets
        self.predictor = nn.Sequential(nn.Linear(cfg.data.concept.num, 2))

    def configure_optimizers(self):
        optimizer = builder.build_optimizer(self.cfg, self.lr, self.model)
        scheduler = builder.build_scheduler(self.cfg, optimizer, self.dm)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def training_epoch_end(self, training_step_outputs):
        return self.shared_epoch_end(training_step_outputs, "train")

    def validation_epoch_end(self, validation_step_outputs):
        return self.shared_epoch_end(validation_step_outputs, "val")

    def test_epoch_end(self, test_step_outputs):
        return self.shared_epoch_end(test_step_outputs, "test")

    def shared_step(self, batch, split):
        """Similar to training step"""

        x, y = batch
        concept_labels = y["concept"]
        diag_labels = y["diag"]

        concept_logit = self.model(x)  

        diag_logit = self.predictor(concept_logit) 
        diag_weight = 1 # NOTE:
        loss = self.loss(concept_logit, concept_labels) + diag_weight * self.loss(diag_logit, diag_labels)

        log_iter_loss = True if split == "train" else False
        self.log(
            f"{split}_loss",
            loss,
            on_epoch=True,
            on_step=log_iter_loss,
            logger=True,
            prog_bar=True,
        )

        return_dict = {"loss": loss, "concept_logit": concept_logit, "diag_logit": diag_logit,
                       "concept_labels": concept_labels, "diag_labels": diag_labels}
        return return_dict

    def shared_epoch_end(self, step_outputs, split):
        concept_logit = torch.cat([x["concept_logit"] for x in step_outputs])
        diag_logit = torch.cat([x["diag_logit"] for x in step_outputs])
        concept_labels = torch.cat([x["concept_labels"] for x in step_outputs])
        diag_labels = torch.cat([x["diag_labels"] for x in step_outputs])

        diag_prob = torch.sigmoid(diag_logit)
        diag_labels = diag_labels.detach().cpu().numpy()
        diag_prob = diag_prob.detach().cpu().numpy()

        # diagnosis 
        indexes = np.argmax(diag_prob, axis=-1)
        prob_pred = np.zeros_like(diag_prob)
        for i in range(len(indexes)):
            prob_pred[i][indexes[i]] = 1
        f1_micro = f1_score(diag_labels, prob_pred, average="micro")
        f1_weighted = f1_score(diag_labels, prob_pred, average="weighted")
        f1_samples = f1_score(diag_labels, prob_pred, average="samples")
        f1_macro = f1_score(diag_labels, prob_pred, average="macro")

        diag_epoch_auc = roc_auc_score(diag_labels, diag_prob)
        self.log(f"{split}_diag_auroc", diag_epoch_auc, on_epoch=True, logger=True, prog_bar=True)

        if split == "val":
            self.log(f"{split}_auroc", diag_epoch_auc, on_epoch=True, logger=True, prog_bar=True)
            self.log(f"{split}_f1_score_micro", f1_micro, on_epoch=True, logger=True, prog_bar=True)
            self.log(f"{split}_f1_score_weighted", f1_weighted, on_epoch=True, logger=True, prog_bar=True)
            self.log(f"{split}_f1_score_samples", f1_samples, on_epoch=True, logger=True, prog_bar=True)
            self.log(f"{split}_f1_score_macro", f1_macro, on_epoch=True, logger=True, prog_bar=True)


    def calculateTop1(self, logits, labels):
        with torch.no_grad():
            labels = labels.argmax(dim=1)
            pred = logits.argmax(dim=1)
            return torch.eq(pred, labels).sum().float().item() / labels.size(0)
