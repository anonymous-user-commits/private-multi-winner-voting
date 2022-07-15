from torch import nn

import torch.nn.functional as F
from datasets.deprecated.chexpert.model.backbone.vgg import (
    vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn)
from datasets.deprecated.chexpert.model.backbone.resnet import (
    resnet18, resnet34, resnet50, resnet101, wide_resnet50_2,
)
from datasets.deprecated.chexpert.model.backbone.densenet import (
    densenet121, densenet169, densenet201)
from datasets.deprecated.chexpert.model.backbone.inception import (inception_v3)
from datasets.deprecated.chexpert.model.global_pool import GlobalPool
from datasets.deprecated.chexpert.model.attention_map import AttentionMap

BACKBONES = {
    'vgg11': vgg11,
    'vgg11_bn': vgg11_bn,
    'vgg13': vgg13,
    'vgg13_bn': vgg13_bn,
    'vgg16': vgg16,
    'vgg16_bn': vgg16_bn,
    'vgg19': vgg19,
    'vgg19_bn': vgg19_bn,
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'wide_resnet50_2': wide_resnet50_2,
    'densenet121': densenet121,
    'densenet169': densenet169,
    'densenet201': densenet201,
    'inception_v3': inception_v3,
}

BACKBONES_TYPES = {
    'vgg11': 'vgg',
    'vgg11_bn': 'vgg',
    'vgg13': 'vgg',
    'vgg13_bn': 'vgg',
    'vgg16': 'vgg',
    'vgg16_bn': 'vgg',
    'vgg19': 'vgg',
    'vgg19_bn': 'vgg',
    'resnet18': 'resnet',
    'resnet34': 'resnet',
    'resnet50': 'resnet',
    'resnet101': 'resnet',
    'wide_resnet50_2': 'resnet',
    'densenet121': 'densenet',
    'densenet169': 'densenet',
    'densenet201': 'densenet',
    'inception_v3': 'inception',
}


class Classifier(nn.Module):

    def __init__(self, name, args):
        super(Classifier, self).__init__()
        self.args = args
        cfg = args.cfg
        self.cfg = cfg
        self.name = name
        self.backbone_name = args.architecture.split('-')[1]
        self.backbone = BACKBONES[self.backbone_name](cfg)
        self.global_pool = GlobalPool(cfg)
        self.expand = 1
        if cfg.global_pool == 'AVG_MAX':
            self.expand = 2
        elif cfg.global_pool == 'AVG_MAX_LSE':
            self.expand = 3
        self._init_classifier()
        self._init_bn()
        self._init_attention_map()

    def _init_classifier(self):
        for index, num_class in enumerate(self.cfg.num_classes):
            if BACKBONES_TYPES[self.backbone_name] == 'vgg':
                setattr(
                    self,
                    "fc_" + str(index),
                    nn.Conv2d(
                        512 * self.expand,
                        num_class,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True))
            elif BACKBONES_TYPES[self.backbone_name] == 'resnet':
                setattr(
                    self,
                    "fc_" + str(index),
                    nn.Conv2d(
                        512 * self.expand,
                        num_class,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True))
            elif BACKBONES_TYPES[self.backbone_name] == 'densenet':
                setattr(
                    self,
                    "fc_" + str(index),
                    nn.Conv2d(
                        self.backbone.num_features *
                        self.expand,
                        num_class,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True))
            elif BACKBONES_TYPES[self.backbone_name] == 'inception':
                setattr(
                    self,
                    "fc_" + str(index),
                    nn.Conv2d(
                        2048 * self.expand,
                        num_class,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=True))
            else:
                raise Exception(
                    'Unknown backbone_name: {}'.format(self.backbone_name)
                )

            classifier = getattr(self, "fc_" + str(index))
            if isinstance(classifier, nn.Conv2d):
                classifier.weight.data.normal_(0, 0.01)
                classifier.bias.data.zero_()

    def _init_bn(self):
        for index, num_class in enumerate(self.cfg.num_classes):
            if BACKBONES_TYPES[self.backbone_name] == 'vgg':
                setattr(self, "bn_" + str(index),
                        nn.BatchNorm2d(512 * self.expand))
            elif BACKBONES_TYPES[self.backbone_name] == 'densenet':
                setattr(
                    self,
                    "bn_" +
                    str(index),
                    nn.BatchNorm2d(
                        self.backbone.num_features *
                        self.expand))
            elif BACKBONES_TYPES[self.backbone_name] == 'inception':
                setattr(self, "bn_" + str(index),
                        nn.BatchNorm2d(2048 * self.expand))
            else:
                raise Exception(
                    'Unknown backbone_name: {}'.format(self.backbone_name)
                )

    def _init_attention_map(self):
        if BACKBONES_TYPES[self.backbone_name] == 'vgg':
            setattr(self, "attention_map", AttentionMap(self.cfg, 512))
        elif BACKBONES_TYPES[self.backbone_name] == 'densenet':
            setattr(
                self,
                "attention_map",
                AttentionMap(
                    self.cfg,
                    self.backbone.num_features))
        elif BACKBONES_TYPES[self.backbone_name] == 'inception':
            setattr(self, "attention_map", AttentionMap(self.cfg, 2048))
        else:
            raise Exception(
                'Unknown backbone_name: {}'.format(self.backbone_name)
            )

    def cuda(self, device=None):
        return self._apply(lambda t: t.cuda(device))

    def forward(self, x):
        # (N, C, H, W)
        feat_map = self.backbone(x)
        # [(N, 1), (N,1),...]
        logits = list()
        # [(N, H, W), (N, H, W),...]
        logit_maps = list()
        for index, num_class in enumerate(self.cfg.num_classes):
            if self.cfg.attention_map != "None":
                feat_map = self.attention_map(feat_map)

            classifier = getattr(self, "fc_" + str(index))
            # (N, 1, H, W)
            logit_map = None
            if not (self.cfg.global_pool == 'AVG_MAX' or
                    self.cfg.global_pool == 'AVG_MAX_LSE'):
                logit_map = classifier(feat_map)
                logit_maps.append(logit_map.squeeze())
            # (N, C, 1, 1)
            feat = self.global_pool(feat_map, logit_map)

            if self.cfg.fc_bn:
                bn = getattr(self, "bn_" + str(index))
                feat = bn(feat)
            feat = F.dropout(feat, p=self.cfg.fc_drop, training=self.training)
            # (N, num_class, 1, 1)

            logit = classifier(feat)
            # (N, num_class)
            logit = logit.squeeze(-1).squeeze(-1)
            logits.append(logit)

        if self.args.chexpert_dataset_type == 'pos':
            # We only have a single class/task and a binary classification.
            logits = logits[0]
            logits = logits.squeeze()
        return logits


class SingleClassClassifier(Classifier):

    def __init__(self, name, args):
        super(SingleClassClassifier, self).__init__(name=name, args=args)
        if args.chexpert_dataset_type == 'pos':
            # Extract only a single label from the binary positive or negative
            # classication (positive - disease present, negtive - disease absent).
            self.class_index = 0
        else:
            self.class_index = -1

    def forward(self, x):
        logits = super(SingleClassClassifier, self).forward(x)
        return logits
