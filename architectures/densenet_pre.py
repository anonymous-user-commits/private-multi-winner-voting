import torch
from torch import nn as nn
import torch.nn.functional as F
import pretrainedmodels


class densenetpre(torch.nn.Module):
    def __init__(self, num_outputs=5):
        super(densenetpre, self).__init__()
        self.model = pretrainedmodels.__dict__["densenet121"](pretrained="imagenet")
        self.model.features.train(False)
        # self.classifier_layer = nn.Sequential(
        #     nn.Linear(1024, 256),
        #     nn.BatchNorm1d(256),
        #     nn.Dropout(0.2),
        #     nn.Linear(256, 128),
        #     nn.Linear(128, 5)
        # )
        self.classifier_layer = nn.Linear(1024, num_outputs)
        self.classifier_layer.train(True)
        self.num_outputs = num_outputs
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
        # if self.num_outputs == 1:
        #     out = torch.sigmoid(out)
        return out


def test():
    net = densenetpre(num_outputs=1)
    y = net(torch.randn(1, 3, 224, 224))
    print("y", y)
    #print("softmax", torch.sigmoid(y))
    print("y size: ", y.size())


if __name__ == "__main__":
    test()
