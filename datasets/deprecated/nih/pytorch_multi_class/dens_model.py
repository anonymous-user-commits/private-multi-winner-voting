import torch
import torchvision
import torch.nn as nn
from datasets.deprecated.nih.pytorch_multi_class.loss_function import \
    weighted_loss

from pytorch_lightning.core.lightning import LightningModule


class DensModel(LightningModule):

    def __init__(self, pos_weights, neg_weights):
        super().__init__()
        self.pos_weights = pos_weights
        self.neg_weights = neg_weights

        # self.metric = Accuracy()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)

        feature_extracting = True
        self.set_parameter_requires_grad(self.densenet121, feature_extracting)

        num_ftrs = self.densenet121.classifier.in_features

        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 14),
            nn.Sigmoid()
        )

        self.model = self.densenet121

    def forward(self, x):
        x = self.densenet121(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.001,
            betas=(0.9, 0.999), eps=1e-08,
            weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=0.1,
            patience=5,
            mode='min')

        return [optimizer], [scheduler]

    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        #         loss = F.binary_cross_entropy(y_hat, y, size_average = True)
        loss = weighted_loss(
            y, y_hat, self.pos_weights, self.neg_weights, epsilon=1e-7)

        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        # loss = F.binary_cross_entropy(y_hat, y, size_average = True)
        loss = weighted_loss(
            y, y_hat, self.pos_weights, self.neg_weights, epsilon=1e-7)
        # acc = self.metric(y_hat, y)

        return {'val_loss': loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        # loss = F.binary_cross_entropy(y_hat, y, size_average = True)

        loss = weighted_loss(
            y, y_hat, self.pos_weights, self.neg_weights, epsilon=1e-7)

        return {'test_loss': loss}

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}
