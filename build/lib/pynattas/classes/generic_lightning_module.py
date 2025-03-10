import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import pytorch_lightning as pl
from .generic_network import GenericNetwork
from .losses import CategoricalCrossEntropyLoss, FocalLoss
from .custom_iou import calculate_iou
#from .mean_squared_error import MeanSquaredError 
from torchmetrics import MeanSquaredError
from datetime import datetime
import matplotlib.pyplot as plt


class GenericLightningNetwork(pl.LightningModule):
    def __init__(self, parsed_layers, input_channels, num_classes, learning_rate=1e-3):
        super(GenericLightningNetwork, self).__init__()
        self.lr = learning_rate
        self.model = GenericNetwork(
            parsed_layers=parsed_layers,
            input_channels=input_channels,
            num_classes=num_classes,
        )

        # Metrics
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.classification.BinaryAccuracy()
        self.f1_score = torchmetrics.classification.BinaryF1Score()
        self.mcc = torchmetrics.classification.matthews_corrcoef.BinaryMatthewsCorrCoef()
        self.conf_matrix = torchmetrics.classification.BinaryConfusionMatrix()
        self.conf_matrix_pred = torchmetrics.classification.BinaryConfusionMatrix()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(torch.argmax(scores, dim=1), y)
        f1_score = self.f1_score(torch.argmax(scores, dim=1), y)
        mcc = self.mcc(torch.argmax(scores, dim=1), y)
        self.log_dict({
            'train_loss': loss,
            'train_accuracy': accuracy,
            'train_f1_score': f1_score,
            'train_mcc': mcc.float(),
        },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, scores, y = self._common_step(batch, batch_idx)
        self.accuracy.update(torch.argmax(scores, dim=1), y)
        self.f1_score.update(torch.argmax(scores, dim=1), y)
        self.mcc.update(torch.argmax(scores, dim=1), y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(torch.argmax(scores, dim=1), y)
        f1_score = self.f1_score(torch.argmax(scores, dim=1), y)
        mcc = self.mcc(torch.argmax(scores, dim=1), y)
        self.conf_matrix.update(torch.argmax(scores, dim=1), y)
        self.conf_matrix.compute()
        self.log_dict({
            'test_loss': loss,
            'test_accuracy': accuracy,
            'test_f1_score': f1_score,
            'test_mcc': mcc.float(),
        },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        return loss

    def on_test_end(self):
        fig_, ax_ = self.conf_matrix.plot()  # to plot and save confusion matrix
        plt.xlabel('Prediction')
        plt.ylabel('Class')
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(rf"./logs/tb_logs/confusion_matrix_{current_datetime}.png")
        # plt.show()

    def _common_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        loss = self.loss_fn(scores, y)
        return loss, scores, y

    def predict_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        accuracy = self.accuracy(preds, y)
        f1_score = self.f1_score(preds, y)
        mcc = self.mcc(preds, y)
        self.conf_matrix_pred.update(preds, y)
        self.conf_matrix_pred.compute()

        print(f"Accuracy: {accuracy:.3f}")   
        print(f"F1-score: {f1_score:.3f}")
        print(f"MCC: {mcc:.3f} ")
        return preds

    """
    def on_predict_end(self):
        fig_, ax_ = self.conf_matrix_pred.plot()
        plt.xlabel('Prediction')
        plt.ylabel('Class')
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(rf"./logs/tb_logs/confusion_matrix_predictions_{current_datetime}.png")
        plt.show()  # test block=False
    """

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)  # 1e-3 is a sane default value for lr
        return optimizer

    
'''
class GenericLightningSegmentationNetwork(pl.LightningModule):
    def __init__(self, parsed_layers, input_channels, num_classes, learning_rate=1e-3):
        super(GenericLightningSegmentationNetwork, self).__init__()
        self.lr = learning_rate
        self.model = GenericNetwork(
            parsed_layers=parsed_layers,
            input_channels=input_channels,
            num_classes=num_classes,
        )
        self.loss_fn = CategoricalCrossEntropyLoss()
        self.mse = MeanSquaredError()
        self.iou = calculate_iou

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        mse = self.mse(logits, y)
        iou = self.iou(logits, y)
        self.log('train_loss', loss)
        self.log('train_mse', mse)
        self.log('train_iou', iou)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, mse, iou = self._common_step(batch, batch_idx)
        self.log('val_loss', loss)
        self.log('val_mse', mse)
        self.log('val_iou', iou)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss, mse, iou = self._common_step(batch, batch_idx)
        self.log('test_loss', loss)
        self.log('test_mse', mse)
        self.log('test_iou', iou)
        return loss

    def _common_step(self, batch, batch_idx):
        x, y = batch
        print(f"y shape is: {y.shape}")
        logits = self(x)
        print(f"Logits shape is: {logits.shape}")
        loss = self.loss_fn(logits, y)
        mse = self.mse(logits, y)
        iou = self.iou(logits, y)
        return loss, mse, iou

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch
        logits = self(x)
        print(f"Predictions for batch {batch_idx}: {logits}")
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
'''

class GenericLightningSegmentationNetwork(pl.LightningModule):
    def __init__(self, parsed_layers, input_channels, num_classes, learning_rate=1e-3):
        super(GenericLightningSegmentationNetwork, self).__init__()
        self.lr = learning_rate
        self.model = GenericNetwork(
            parsed_layers=parsed_layers,
            input_channels=input_channels,
            num_classes=num_classes,
        )
        
        if not self.model.is_valid_architecture:  
            raise ValueError("The architecture is invalid due to exceeding the parameter limit.")
        
        #self.loss_fn = CategoricalCrossEntropyLoss()
        self.loss_fn = FocalLoss()
        self.mse = MeanSquaredError()
        self.iou = calculate_iou

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        mse = self.mse(logits, y)
        iou = self.iou(logits, y).mean()  # Compute mean IoU for logging
        return loss, mse, iou

    def training_step(self, batch, batch_idx):
        loss, mse, iou = self._common_step(batch, batch_idx)
        self.log('train_loss', loss)
        self.log('train_mse', mse)
        self.log('train_iou', iou)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, mse, iou = self._common_step(batch, batch_idx)
        self.log('val_loss', loss)
        self.log('val_mse', mse)
        self.log('val_iou', iou)
        return loss

    def test_step(self, batch, batch_idx):
        loss, mse, iou = self._common_step(batch, batch_idx)
        self.log('test_loss', loss)
        self.log('test_mse', mse)
        self.log('test_iou', iou)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch
        logits = self(x)
        return logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class GenericLightningNetwork_Custom(pl.LightningModule):
    def __init__(self, parsed_layers, model_parameters, input_channels, num_classes, learning_rate=1e-3):
        super(GenericLightningNetwork_Custom, self).__init__()
        self.lr = learning_rate
        self.model = GenericNetwork(
            parsed_layers=parsed_layers,
            model_parameters=model_parameters,
            input_channels=input_channels,
            num_classes=num_classes,
        )
        self.class_weights = None  # Initialize with a default value

        # Metrics
        self.loss_fn = ce_loss  # Use custom loss function
        self.accuracy = torchmetrics.classification.BinaryAccuracy()
        self.f1_score = torchmetrics.classification.BinaryF1Score()
        self.mcc = torchmetrics.classification.matthews_corrcoef.BinaryMatthewsCorrCoef()
        self.conf_matrix = torchmetrics.classification.BinaryConfusionMatrix()
        self.conf_matrix_pred = torchmetrics.classification.BinaryConfusionMatrix()

    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        # Ensure the datamodule is attached and has class_weights
        if hasattr(self.trainer, 'datamodule') and hasattr(self.trainer.datamodule, 'class_weights'):
            self.class_weights = self.trainer.datamodule.class_weights.to(self.device)
            print(f"GenericLightningNetwork class_weights set: {self.class_weights}")
        else:
            print("GenericLightningNetwork class_weights NOT set")

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(torch.argmax(scores, dim=1), y)
        f1_score = self.f1_score(torch.argmax(scores, dim=1), y)
        mcc = self.mcc(torch.argmax(scores, dim=1), y)
        self.log_dict({
            'train_loss': loss,
            'train_accuracy': accuracy,
            'train_f1_score': f1_score,
            'train_mcc': mcc.float(),
        },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, scores, y = self._common_step(batch, batch_idx)
        self.accuracy.update(torch.argmax(scores, dim=1), y)
        self.f1_score.update(torch.argmax(scores, dim=1), y)
        self.mcc.update(torch.argmax(scores, dim=1), y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss, scores, y = self._common_step(batch, batch_idx)
        accuracy = self.accuracy(torch.argmax(scores, dim=1), y)
        f1_score = self.f1_score(torch.argmax(scores, dim=1), y)
        mcc = self.mcc(torch.argmax(scores, dim=1), y)
        self.conf_matrix.update(torch.argmax(scores, dim=1), y)
        self.conf_matrix.compute()
        self.log_dict({
            'test_loss': loss,
            'test_accuracy': accuracy,
            'test_f1_score': f1_score,
            'test_mcc': mcc.float(),
        },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True
        )
        return loss

    def on_test_end(self):
        fig_, ax_ = self.conf_matrix.plot()  # to plot and save confusion matrix
        plt.xlabel('Prediction')
        plt.ylabel('Class')
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(rf"./logs/tb_logs/confusion_matrix_{current_datetime}.png")
        # plt.show()

    def _common_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        if self.class_weights is not None:
            loss = self.loss_fn(logits=scores, targets=y, weight=self.class_weights, use_hard_labels=True)
        else:
            loss = self.loss_fn(logits=scores, targets=y, use_hard_labels=True)

        loss = loss.mean()
        return loss, scores, y

    def predict_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        preds = torch.argmax(scores, dim=1)
        accuracy = self.accuracy(preds, y)
        f1_score = self.f1_score(preds, y)
        mcc = self.mcc(preds, y)
        self.conf_matrix_pred.update(preds, y)
        self.conf_matrix_pred.compute()

        print(f"Accuracy: {accuracy:.3f}")
        print(f"F1-score: {f1_score:.3f}")
        print(f"MCC: {mcc:.3f} ")
        return preds

    """
    def on_predict_end(self):
        fig_, ax_ = self.conf_matrix_pred.plot()
        plt.xlabel('Prediction')
        plt.ylabel('Class')
        current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(rf"./logs/tb_logs/confusion_matrix_predictions_{current_datetime}.png")
        plt.show()  # test block=False
    """

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)  # 1e-3 is a sane default value for lr
        return optimizer


def ce_loss(logits, targets, weight=None, use_hard_labels=True, reduction="none"):
    """
    wrapper for cross entropy loss in pytorch.

    Args
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        weight: weights for loss if hard labels are used.
        use_hard_labels: If True, targets have [Batch size] shape with int values.
                         If False, the target is vector. Default to True.
    """
    if use_hard_labels:
        if weight is not None:
            return F.cross_entropy(
                logits, targets.long(), weight=weight, reduction=reduction
            )
        else:
            return F.cross_entropy(logits, targets.long(), reduction=reduction)
    else:
        assert logits.shape == targets.shape
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        return nll_loss

