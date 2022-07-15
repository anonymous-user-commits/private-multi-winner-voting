import torch
from torch import nn as nn
import torch.nn.functional as F
import pretrainedmodels
import torchvision.models as models


class resnetpre(torch.nn.Module):
    def __init__(self):
        super(resnetpre, self).__init__()
        self.model = pretrainedmodels.__dict__["resnet50"](pretrained="imagenet")
        # self.model = models.resnet50(pretrained= True)
        # modules = list(self.model.children())[:-1]
        # self.model = nn.Sequential(*modules)
        for param in self.model.parameters():
            param.requires_grad = False
        #print(self.model.__dict__)
        #self.model.features.train(False)
        # self.classifier_layer = nn.Sequential(
        #     nn.Linear(1024, 256),
        #     nn.BatchNorm1d(256),
        #     nn.Dropout(0.2),
        #     nn.Linear(256, 128),
        #     nn.Linear(128, 5)
        # )
        self.classifier_layer = nn.Linear(2048, 20)
        self.classifier_layer.train(True)
        # self.classifier_layer = nn.Sequential(
        #     nn.Linear(128, 5)
        # )

    def forward(self, x):
        # batch_size, _, _, _ = x.shape
        out = self.model.features(x)
        # out = F.avg_pool2d(out, 1).reshape(256, -1) #
        # print("shape out.", out.size())
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier_layer(out)
        return out


def test():
    net = resnetpre()
    y = net(torch.randn(1, 3, 224, 224))
    print("y size: ", y.size())


if __name__ == "__main__":
    test()
