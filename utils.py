from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import getpass
import json
import os
import sys
import time
from datetime import datetime
from enum import Enum
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ax.service.managed_loop import optimize  # pip install ax-platform
from easydict import EasyDict as edict
from numba import njit
from sklearn import metrics
from torch import Tensor
from torch.nn import DataParallel
from torch.optim import SGD, Adadelta, Adagrad, Adam, RMSprop
from torch.optim.lr_scheduler import MultiStepLR
# from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms

from datasets.celeba.celeba_utils import celeba_load_private_data_and_qap
from datasets.celeba.celeba_utils import get_celeba_dev_set
from datasets.celeba.celeba_utils import get_celeba_private_data
from datasets.celeba.celeba_utils import get_celeba_test_set
from datasets.celeba.celeba_utils import get_celeba_train_set
from datasets.cifar.cifar_utils import get_cifar_private_data
from datasets.cifar.tinyimage500k import get_extra_cifar10_data_from_ti
from datasets.deprecated.chexpert.chexpert_utils import (
    get_chexpert_all_private_datasets,
)
from datasets.deprecated.chexpert.chexpert_utils import \
    get_chexpert_dataset_type
from datasets.deprecated.chexpert.chexpert_utils import get_chexpert_dev_set
from datasets.deprecated.chexpert.chexpert_utils import get_chexpert_train_set
from datasets.deprecated.chexpert.chexpert_utils import \
    get_chexpert_private_data
from datasets.deprecated.chexpert.chexpert_utils import get_chexpert_test_set
from datasets.deprecated.coco.coco_data_utils import get_coco_private_data
from datasets.deprecated.coco.coco_data_utils import get_coco_test_set
from datasets.deprecated.coco.coco_data_utils import get_coco_train_set
from datasets.deprecated.coco.helper_functions.helper_functions import \
    average_precision
from datasets.deprecated.coco.helper_functions.helper_functions import mAP
from datasets.deprecated.coco.loss_functions.losses import \
    AsymmetricLossOptimized
from datasets.deprecated.coco.models.utils.tresnet_params import \
    get_tresnet_params
from datasets.deprecated.retinopathy.retinopathy_utils import (
    get_retinopathy_private_data,
)
from datasets.deprecated.retinopathy.retinopathy_utils import \
    get_retinopathy_train_data
from datasets.deprecated.retinopathy.retinopathy_utils import \
    get_retinopathy_transform
from datasets.mnist.mnist_utils import get_mnist_dataset, get_mnist_private_data
from datasets.mnist.mnist_utils import get_mnist_dataset_by_name
from datasets.mnist.mnist_utils import get_mnist_transforms
from datasets.pascal.PascalLoader import DataLoader as PascalDataLoader
from datasets.pascal.Pascal_utils import \
    get_dataloaders as get_pascal_private_data
from datasets.svhn.svhn_utils import FromSVHNtoMNIST
from datasets.svhn.svhn_utils import get_svhn_private_data
from datasets.xray.dataset_pathologies import default_pathologies
from datasets.xray.dataset_pathologies import get_indexes
from datasets.xray.xray_datasets import SubsetDataset
from datasets.xray.xray_utils import get_xray_private_dataloaders
from datasets.xray.xray_utils import get_xray_train_data
# from datasets.cxpert.data import get_cxpert_debug_dataloaders
from datasets.xray.xray_utils import get_xray_test_data
from datasets.xray.xray_utils import get_xray_unlabeled_data
from datasets.xray.xray_utils import get_xray_unlabeled_dataloaders
from datasets.xray.xray_utils import load_ordered_unlabeled_xray
from datasets.xray.xray_utils import xray_load_private_data_and_qap
from general_utils.save_load import save_obj
from models.private_model import get_private_model_by_id
from queryset import QuerySet
from queryset import get_aggregated_labels_filename
from queryset import get_queries_filename
from queryset import get_raw_queries_filename
from queryset import get_targets_filename


class metric(Enum):
    """
    Evaluation metrics for the models.
    """

    acc = "acc"
    acc_detailed = "acc_detailed"
    acc_detailed_avg = "acc_detailed_avg"
    balanced_acc = "balanced_acc"
    balanced_acc_detailed = "balanced_acc_detailed"
    auc = "auc"
    auc_detailed = "auc_detailed"
    f1_score = "f1_score"
    f1_score_detailed = "f1_score_detailed"
    loss = "loss"
    test_loss = "test_loss"
    train_loss = "train_loss"
    map = "map"
    map_detailed = "map_detailed"
    gaps_mean = "gaps_mean"
    gaps_detailed = "gaps_detailed"
    pc = "pc"
    rc = "rc"
    fc = "fc"
    po = "po"
    ro = "ro"
    fo = "fo"

    def __str__(self):
        return self.name


class result(Enum):
    """
    Properties of the results.
    """

    aggregated_labels = "aggregated_labels"
    indices_answered = "indices_answered"
    predictions = "predictions"
    labels_answered = "labels_answered"
    count_answered = "count_answered"
    confidence_scores = "confidence_scores"

    def __str__(self):
        return self.name


def get_device(args):
    num_devices = torch.cuda.device_count()
    device_ids = args.device_ids
    if not torch.cuda.is_available():
        return torch.device("cpu"), []
    if num_devices < len(device_ids):
        raise Exception(
            "#available gpu : {} < --device_ids : {}".format(
                num_devices, len(device_ids)
            )
        )
    if args.cuda:
        device = torch.device("cuda:{}".format(device_ids[0]))
    else:
        device = torch.device("cpu")
    return device, device_ids


def get_auc(classification_type, y_true, y_pred, num_classes=None):
    """
    Compute the AUC (Area Under the receiver operator Curve).
    :param classification_type: the type of classification.
    :param y_true: the true labels.
    :param y_pred: the scores or predicted labels.
    :return: AUC score.
    """
    if classification_type == "binary":
        # fpr, tpr, thresholds = metrics.roc_curve(
        #     y_true, y_pred, pos_label=1)
        # auc = metrics.auc(fpr, tpr)
        auc = metrics.roc_auc_score(y_true=y_true, y_score=y_pred,
                                    average="weighted")
    elif classification_type == "multiclass":
        auc = metrics.roc_auc_score(
            y_true=y_true,
            y_score=y_pred,
            # one-vs-one, insensitive to class imbalances when average==macro
            multi_class="ovo",
            average="macro",
            labels=[x for x in range(num_classes)],
        )
    elif classification_type in ["multilabel", "multilabel_counting"]:
        # fpr, tpr, thresholds = metrics.roc_curve(
        #     y_true, y_pred, pos_label=1)
        # auc = metrics.auc(fpr, tpr)
        auc = metrics.roc_auc_score(y_true=y_true, y_score=y_pred,
                                    average="weighted")
    else:
        raise Exception(
            f"Unexpected classification_type: {classification_type}.")
    return auc


def get_prediction(args, model, unlabeled_dataloader):
    initialized = False
    with torch.no_grad():
        for data, _ in unlabeled_dataloader:
            if args.cuda:
                data = data.cuda()
            output = model(data)
            if not initialized:
                result = output
                initialized = True
            else:
                result = torch.cat((result, output), 0)
    return result


def count_samples_per_class(dataloader):
    steps = len(dataloader)
    dataiter = iter(dataloader)
    targets = []
    for step in range(steps):
        _, target = next(dataiter)
        if isinstance(target, (int, float)):
            targets.append(target)
        else:
            if isinstance(target, torch.Tensor):
                target = target.detach().cpu().squeeze().squeeze().numpy()
            targets += list(target)
    targets = np.array(targets)
    uniques = np.unique(targets)
    counts = {u: 0 for u in uniques}
    for u in targets:
        counts[u] += 1
    return counts


def get_timestamp():
    dateTimeObj = datetime.now()
    # timestampStr = dateTimeObj.strftime("%Y-%B-%d-(%H:%M:%S.%f)")
    timestampStr = dateTimeObj.strftime("%Y-%m-%d-%H-%M-%S-%f")
    return timestampStr


def get_cfg(cfg_path):
    with open(cfg_path) as f:
        cfg = edict(json.load(f))

    user = getpass.getuser()
    for k, v in cfg.items():
        if "{user}" in str(v):
            cfg[k] = v.replace("{user}", user)
    return cfg


def lr_schedule(lr, lr_factor, epoch_now, lr_epochs):
    """
    Learning rate schedule with respect to epoch
    lr: float, initial learning rate
    lr_factor: float, decreasing factor every epoch_lr
    epoch_now: int, the current epoch
    lr_epochs: list of int, decreasing every epoch in lr_epochs
    return: lr, float, scheduled learning rate.
    """
    count = 0
    for epoch in lr_epochs:
        if epoch_now >= epoch:
            count += 1
            continue

        break

    return lr * np.power(lr_factor, count)


def class_wise_loss_reweighting(beta, samples_per_cls):
    """
     https://towardsdatascience.com/handling-class-imbalanced-data-using-a-loss-specifically-made-for-it-6e58fd65ffab

    :param samples_per_cls: number of samples per class
    :return: weights per class for the loss function
    """
    num_classes = len(samples_per_cls)
    effective_sample_count = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_sample_count)
    # normalize the weights
    weights = weights / np.sum(weights) * num_classes
    return weights


def load_private_data_and_qap(args):
    """Load labeled private data and query-answer pairs for retraining private models."""
    kwargs = get_kwargs(args=args)
    args.kwargs = kwargs
    if "mnist" in args.dataset:
        all_private_datasets = get_mnist_dataset(args=args, train=True)
        private_dataset_size = len(all_private_datasets) // args.num_models
        all_augmented_dataloaders = []
        for i in range(args.num_querying_parties):
            begin = i * private_dataset_size
            if i == args.num_models - 1:
                end = len(all_private_datasets)
            else:
                end = (i + 1) * private_dataset_size
            indices = list(range(begin, end))
            private_dataset = Subset(all_private_datasets, indices)
            query_dataset = QuerySet(
                args, transform=get_mnist_transforms(args=args), id=i
            )
            augmented_dataset = ConcatDataset([private_dataset, query_dataset])
            augmented_dataloader = DataLoader(
                augmented_dataset, batch_size=args.batch_size, shuffle=True,
                **kwargs
            )
            all_augmented_dataloaders.append(augmented_dataloader)
        return all_augmented_dataloaders
    elif args.dataset == "pascal":
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        val_transform = transforms.Compose(
            [
                # transforms.Scale(256),
                # transforms.CenterCrop(227),
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        # args.pascal_path = '/jmain01/home/JAD007/txk02/axs14-txk02/multi-label/VOC2012/'
        args.batch = 32
        args.pascal_path = os.path.join(args.data_dir, "VOC2012/")
        all_augmented_dataloaders = []
        for i in range(args.num_querying_parties):
            private_dataset = PascalDataLoader(
                args.pascal_path, "small_train_{}".format(i),
                transform=val_transform
            )

            query_dataset = QuerySet(args, transform=val_transform, id=i)
            augmented_dataset = ConcatDataset([private_dataset, query_dataset])
            augmented_dataloader = DataLoader(
                augmented_dataset, batch_size=args.batch_size, shuffle=True,
                **kwargs
            )
            # x, y = private_dataset.__getitem__(3)
            # print(x, y)
            # x,y = query_dataset.__getitem__(3)
            # print(x,y)
            a1 = DataLoader(private_dataset, batch_size=args.batch_size)
            a2 = DataLoader(query_dataset, batch_size=args.batch_size)
            # print(a1.dataset.__dict__)
            # print(a2.dataset.__dict__)
            # print("labels1", a1.dataset.labels)
            # print("labels2", a2.dataset.labels)
            # print("comb", augmented_dataloader.dataset.labels)

            all_augmented_dataloaders.append(
                a2
            )  # augmented_dataloader) find another way of comibing the data
        return all_augmented_dataloaders

    elif args.dataset == "svhn":
        trainset = datasets.SVHN(
            root=args.dataset_path,
            split="train",
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.43768212, 0.44376972, 0.47280444),
                        (0.19803013, 0.20101563, 0.19703615),
                    ),
                ]
            ),
            download=True,
        )
        extraset = datasets.SVHN(
            root=args.dataset_path,
            split="extra",
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.42997558, 0.4283771, 0.44269393),
                        (0.19630221, 0.1978732, 0.19947216),
                    ),
                ]
            ),
            download=True,
        )
        private_trainset_size = len(trainset) // args.num_models
        private_extraset_size = len(extraset) // args.num_models
        all_augmented_dataloaders = []
        for i in range(args.num_querying_parties):
            train_begin = i * private_trainset_size
            extra_begin = i * private_extraset_size
            if i == args.num_models - 1:
                train_end = len(trainset)
            else:
                train_end = (i + 1) * private_trainset_size
            if i == args.num_models - 1:
                extra_end = len(extraset)
            else:
                extra_end = (i + 1) * private_extraset_size
            train_indices = list(range(train_begin, train_end))
            extra_indices = list(range(extra_begin, extra_end))
            private_trainset = Subset(trainset, train_indices)
            private_extraset = Subset(extraset, extra_indices)
            query_dataset = QuerySet(
                args,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.45242317, 0.45249586, 0.46897715),
                            (0.21943446, 0.22656967, 0.22850613),
                        ),
                    ]
                ),
                id=i,
            )
            augmented_dataset = ConcatDataset(
                [private_trainset, private_extraset, query_dataset]
            )
            augmented_dataloader = DataLoader(
                augmented_dataset, batch_size=args.batch_size, shuffle=True,
                **kwargs
            )
            all_augmented_dataloaders.append(augmented_dataloader)
        return all_augmented_dataloaders
    elif args.dataset.startswith("cifar"):
        if args.dataset == "cifar10":
            datasets_cifar = datasets.CIFAR10
        elif args.dataset == "cifar100":
            datasets_cifar = datasets.CIFAR100
        else:
            raise Exception(args.datasets_exception)
        all_private_datasets = datasets_cifar(
            args.dataset_path,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.Pad(4),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.49139969, 0.48215842, 0.44653093),
                        (0.24703223, 0.24348513, 0.26158784),
                    ),
                ]
            ),
            download=True,
        )
        private_dataset_size = len(all_private_datasets) // args.num_models
        all_augmented_dataloaders = []
        for i in range(args.num_querying_parties):
            begin = i * private_dataset_size
            if i == args.num_models - 1:
                end = len(all_private_datasets)
            else:
                end = (i + 1) * private_dataset_size
            indices = list(range(begin, end))
            private_dataset = Subset(all_private_datasets, indices)
            query_dataset = QuerySet(
                args,
                transform=transforms.Compose(
                    [
                        transforms.Pad(4),
                        transforms.RandomCrop(32),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.49421429, 0.4851314, 0.45040911),
                            (0.24665252, 0.24289226, 0.26159238),
                        ),
                    ]
                ),
                id=i,
            )
            augmented_dataset = ConcatDataset([private_dataset, query_dataset])
            augmented_dataloader = DataLoader(
                augmented_dataset, batch_size=args.batch_size, shuffle=True,
                **kwargs
            )
            all_augmented_dataloaders.append(augmented_dataloader)
        return all_augmented_dataloaders

    elif args.dataset.startswith("chexpert"):
        all_private_datasets, private_dataset_size = get_chexpert_all_private_datasets(
            args=args
        )
        all_augmented_dataloaders = []
        for i in range(1):  # CHANGED HERE. WAS args.num_querying_parties
            begin = i * private_dataset_size
            if i == args.num_models - 1:
                end = len(all_private_datasets)
            else:
                end = (i + 1) * private_dataset_size
            indices = list(range(begin, end))
            private_dataset = Subset(all_private_datasets, indices)
            query_dataset = QuerySet(
                args, transform=transforms.Compose([transforms.ToTensor()]),
                id=i
            )
            augmented_dataset = ConcatDataset([private_dataset, query_dataset])
            augmented_dataloader = DataLoader(
                augmented_dataset, batch_size=args.batch_size, shuffle=True,
                **kwargs
            )
            all_augmented_dataloaders.append(augmented_dataloader)
        return all_augmented_dataloaders
    elif args.dataset == "retinopathy":
        all_private_datasets, private_dataset_size = get_retinopathy_train_data(
            args, mode="train"
        )
        all_augmented_dataloaders = []
        for i in range(args.num_querying_parties):
            begin = i * private_dataset_size
            if i == args.num_models - 1:
                end = len(all_private_datasets)
            else:
                end = (i + 1) * private_dataset_size
            indices = list(range(begin, end))
            private_dataset = Subset(all_private_datasets, indices)
            query_dataset = QuerySet(
                args, transform=get_retinopathy_transform(args=args), id=i
            )
            augmented_dataset = ConcatDataset([private_dataset, query_dataset])
            augmented_dataloader = DataLoader(
                augmented_dataset, batch_size=args.batch_size, shuffle=True,
                **kwargs
            )
            all_augmented_dataloaders.append(augmented_dataloader)
        return all_augmented_dataloaders
    elif args.dataset == "celeba":
        return celeba_load_private_data_and_qap(args=args)
    elif args.dataset in args.xray_datasets:
        return xray_load_private_data_and_qap(args=args)
    else:
        raise Exception(args.datasets_exception)


def get_data_subset(args, dataset, indices):
    """
    The Subset function differs between datasets, unfortunately.

    :param args: program params
    :param dataset: extract subset of the data from this dataset
    :param indices: the indices in the dataset to be accessed
    :return: the subset
    """
    if args.dataset in args.xray_datasets:
        return SubsetDataset(dataset=dataset, idxs=indices)
    else:
        return Subset(dataset=dataset, indices=indices)


def save_raw_queries_targets(args, dataset, indices, name):
    kwargs = get_kwargs(args=args)
    query_dataset = get_data_subset(args=args, dataset=dataset, indices=indices)

    queryloader = DataLoader(
        query_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
    )
    all_samples = []
    all_targets = []
    for data, targets in queryloader:
        all_samples.append(data.numpy())
        all_targets.append(targets.numpy())
    all_samples = np.concatenate(all_samples, axis=0).transpose(0, 2, 3, 1)
    assert len(all_samples.shape) == 4 and all_samples.shape[0] == len(indices)
    all_samples = (all_samples * 255).astype(np.uint8)

    if "mnist" in args.dataset:
        all_samples = np.squeeze(all_samples)
        shape_len = 3
    else:
        shape_len = 4
    assert len(all_samples.shape) == shape_len

    filename = get_raw_queries_filename(name=name, args=args)
    filepath = os.path.join(args.ensemble_model_path, filename)
    np.save(filepath, all_samples)

    save_targets(name=name, args=args, targets=all_targets)


def save_targets(args, name, targets):
    targets = np.concatenate(targets, axis=0)
    filename = get_targets_filename(name=name, args=args)
    filepath = os.path.join(args.ensemble_model_path, filename)
    np.save(filepath, targets)


def save_queries(args, dataset, indices, name):
    # Select the query items (data points that) given by indices.
    query_dataset = get_data_subset(args=args, dataset=dataset, indices=indices)

    kwargs = get_kwargs(args=args)
    queryloader = DataLoader(
        query_dataset, batch_size=args.batch_size, shuffle=False, **kwargs
    )
    all_samples = []
    all_targets = []
    for data, targets in queryloader:
        all_samples.append(data.numpy())
        all_targets.append(targets.numpy())
    all_samples = np.concatenate(all_samples, axis=0)

    assert len(all_samples.shape) == 4 and all_samples.shape[0] == len(indices)
    if "mnist" in args.dataset:
        all_samples = np.squeeze(all_samples)
        shape_len = 3
    else:
        shape_len = 4

    assert len(all_samples.shape) == shape_len

    filename = get_queries_filename(name=name, args=args)
    filepath = os.path.join(args.ensemble_model_path, filename)
    np.save(filepath, all_samples)

    save_targets(name=name, args=args, targets=all_targets)


def get_all_targets(dataloader) -> Optional[Tensor]:
    dataset = dataloader.dataset
    dataset_len = len(dataset)
    all_targets = None
    with torch.no_grad():
        end = 0
        for _, targets in dataloader:
            batch_size = targets.shape[0]
            begin = end
            end = begin + batch_size
            if all_targets is None:
                if len(targets.shape) == 1:
                    all_targets = torch.zeros(dataset_len)
                elif len(targets.shape) == 2:
                    num_labels = targets.shape[1]
                    all_targets = torch.zeros((dataset_len, num_labels))
                else:
                    raise Exception(
                        f"Unknown setting with the shape of "
                        f"targets: {targets.shape}."
                    )
            all_targets[begin:end] += targets

    return all_targets


def get_all_targets_numpy(dataloader, args) -> Optional[np.ndarray]:
    all_targets = get_all_targets(dataloader=dataloader)
    if all_targets is not None:
        all_targets = all_targets.numpy()
        if args.pick_labels is not None and args.pick_labels != [-1]:
            all_targets = retain_labels_cols(
                target_labels_index=args.pick_labels, labels=all_targets
            )
    return all_targets


def retain_labels_general(labels, args):
    if args.pick_labels != None and args.pick_labels != [-1]:
        labels = retain_labels_cols(target_labels_index=args.pick_labels,
                                    labels=labels)
    elif args.dataset in args.xray_datasets:
        indexes = get_indexes(dataset=args.dataset)
        labels = retain_labels_cols(target_labels_index=indexes, labels=labels)
    return labels


def pick_labels_general(labels: np.ndarray, args, axis=None) -> np.ndarray:
    if args.pick_labels != None and args.pick_labels != [-1]:
        labels = pick_labels_cols(target_labels_index=args.pick_labels,
                                  labels=labels, axis=axis)
    elif args.dataset in args.xray_datasets:
        indexes = get_indexes(dataset=args.dataset)
        labels = pick_labels_cols(target_labels_index=indexes, labels=labels,
                                  axis=axis)
    return labels


def pick_labels_torch(labels: torch.tensor, args):
    if args.pick_labels is not None and args.pick_labels != [-1]:
        labels = pick_labels_cols_torch(target_labels_index=args.pick_labels,
                                        labels=labels)
    elif args.dataset in args.xray_datasets:
        indexes = get_indexes(dataset=args.dataset)
        labels = pick_labels_cols_torch(target_labels_index=indexes,
                                        labels=labels)
    return labels


def save_labels(name, args, labels):
    if np.any(labels == -1):
        not_answered = labels == -1
        labels = labels.astype(np.float32)
        labels[not_answered] = np.nan
    else:
        labels = labels.astype(np.float32)

    labels = retain_labels_general(labels=labels, args=args)

    filename = get_aggregated_labels_filename(name=name, args=args)
    filepath = os.path.join(args.ensemble_model_path, filename)
    np.save(filepath, labels)


def load_private_data_and_qap_imbalanced(args):
    """Load labeled private data (imbalanced) and query-answer pairs for
    retraining private models."""
    kwargs = get_kwargs(args=args)
    if "mnist" in args.dataset:
        all_private_datasets = get_mnist_dataset(args=args, train=True)
        class_indices = get_class_indices(all_private_datasets, args)
        data_indices = [[] for _ in range(args.num_models)]
        for c in range(args.num_classes):
            class_size = len(class_indices[c])
            if c in args.weak_classes:
                samples_per_model = class_size / args.num_models
                size_weak = int(args.weak_class_ratio * samples_per_model)
                total_size_weak = size_weak * args.num_querying_parties
                total_size_strong = class_size - total_size_weak
                num_strong_parties = args.num_models - args.num_querying_parties
                size_strong = total_size_strong // num_strong_parties
                for i in range(args.num_models):
                    if i < args.num_querying_parties:
                        begin = i * size_weak
                        end = (i + 1) * size_weak
                    else:
                        begin = (
                                total_size_weak
                                + (i - args.num_querying_parties) * size_strong
                        )
                        if i == args.num_models - 1:
                            end = class_size
                        else:
                            end = (
                                    total_size_weak
                                    + (
                                            i - args.num_querying_parties + 1) * size_strong
                            )
                    data_indices[i].append(class_indices[c][begin:end])
            else:
                size = class_size // args.num_models
                for i in range(args.num_models):
                    begin = i * size
                    if i == args.num_models - 1:
                        end = class_size
                    else:
                        end = (i + 1) * size
                    data_indices[i].append(class_indices[c][begin:end])
        data_indices = [
            np.concatenate(data_indices[i], axis=0) for i in
            range(args.num_models)
        ]
        assert sum(
            [len(data_indices[i]) for i in range(args.num_models)]) == len(
            all_private_datasets
        )
        assert len(set(np.concatenate(data_indices, axis=0))) == len(
            all_private_datasets
        )
        all_augmented_dataloaders = []
        for i in range(args.num_querying_parties):
            private_dataset = Subset(all_private_datasets, data_indices[i])
            query_dataset = QuerySet(
                args, transform=get_mnist_transforms(args=args), id=i
            )
            augmented_dataset = ConcatDataset([private_dataset, query_dataset])
            augmented_dataloader = DataLoader(
                augmented_dataset, batch_size=args.batch_size, shuffle=True,
                **kwargs
            )
            all_augmented_dataloaders.append(augmented_dataloader)
        return all_augmented_dataloaders

    elif args.dataset == "svhn":
        trainset = datasets.SVHN(
            root=args.dataset_path,
            split="train",
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.43768212, 0.44376972, 0.47280444),
                        (0.19803013, 0.20101563, 0.19703615),
                    ),
                ]
            ),
            download=True,
        )
        extraset = datasets.SVHN(
            root=args.dataset_path,
            split="extra",
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.42997558, 0.4283771, 0.44269393),
                        (0.19630221, 0.1978732, 0.19947216),
                    ),
                ]
            ),
            download=True,
        )
        trainset_class_indices = get_class_indices(trainset, args)
        trainset_data_indices = [[] for _ in range(args.num_models)]
        for c in range(args.num_classes):
            if c in args.weak_classes:
                size_weak = int(
                    args.weak_class_ratio
                    * (len(trainset_class_indices[c]) / args.num_models)
                )
                size_strong = (
                                      len(trainset_class_indices[c])
                                      - size_weak * args.num_querying_parties
                              ) // (args.num_models - args.num_querying_parties)
                for i in range(args.num_models):
                    if i < args.num_querying_parties:
                        begin = i * size_weak
                        end = (i + 1) * size_weak
                        trainset_data_indices[i].append(
                            trainset_class_indices[c][begin:end]
                        )
                    else:
                        begin = (
                                size_weak * args.num_querying_parties
                                + (i - args.num_querying_parties) * size_strong
                        )
                        end = (
                            len(trainset_class_indices[c])
                            if i == args.num_models - 1
                            else size_weak * args.num_querying_parties
                                 + (
                                         i + 1 - args.num_querying_parties) * size_strong
                        )
                        trainset_data_indices[i].append(
                            trainset_class_indices[c][begin:end]
                        )
            else:
                size = len(trainset_class_indices[c]) // args.num_models
                for i in range(args.num_models):
                    begin = i * size
                    end = (
                        len(trainset_class_indices[c])
                        if i == args.num_models - 1
                        else (i + 1) * size
                    )
                    trainset_data_indices[i].append(
                        trainset_class_indices[c][begin:end]
                    )
        trainset_data_indices = [
            np.concatenate(trainset_data_indices[i], axis=0)
            for i in range(args.num_models)
        ]
        assert sum(
            [len(trainset_data_indices[i]) for i in range(args.num_models)]
        ) == len(trainset)
        assert len(set(np.concatenate(trainset_data_indices, axis=0))) == len(
            trainset)
        extraset_class_indices = get_class_indices(extraset, args)
        extraset_data_indices = [[] for _ in range(args.num_models)]
        for c in range(args.num_classes):
            if c in args.weak_classes:
                size_weak = int(
                    args.weak_class_ratio
                    * (len(extraset_class_indices[c]) / args.num_models)
                )
                size_strong = (
                                      len(extraset_class_indices[c])
                                      - size_weak * args.num_querying_parties
                              ) // (args.num_models - args.num_querying_parties)
                for i in range(args.num_models):
                    if i < args.num_querying_parties:
                        begin = i * size_weak
                        end = (i + 1) * size_weak
                        extraset_data_indices[i].append(
                            extraset_class_indices[c][begin:end]
                        )
                    else:
                        begin = (
                                size_weak * args.num_querying_parties
                                + (i - args.num_querying_parties) * size_strong
                        )
                        end = (
                            len(extraset_class_indices[c])
                            if i == args.num_models - 1
                            else size_weak * args.num_querying_parties
                                 + (
                                         i + 1 - args.num_querying_parties) * size_strong
                        )
                        extraset_data_indices[i].append(
                            extraset_class_indices[c][begin:end]
                        )
            else:
                size = len(extraset_class_indices[c]) // args.num_models
                for i in range(args.num_models):
                    begin = i * size
                    end = (
                        len(extraset_class_indices[c])
                        if i == args.num_models - 1
                        else (i + 1) * size
                    )
                    extraset_data_indices[i].append(
                        extraset_class_indices[c][begin:end]
                    )
        extraset_data_indices = [
            np.concatenate(extraset_data_indices[i], axis=0)
            for i in range(args.num_models)
        ]
        assert sum(
            [len(extraset_data_indices[i]) for i in range(args.num_models)]
        ) == len(extraset)
        assert len(set(np.concatenate(extraset_data_indices, axis=0))) == len(
            extraset)
        all_augmented_dataloaders = []
        for i in range(args.num_querying_parties):
            private_trainset = Subset(trainset, trainset_data_indices[i])
            private_extraset = Subset(extraset, extraset_data_indices[i])
            query_dataset = QuerySet(
                args,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.45242317, 0.45249586, 0.46897715),
                            (0.21943446, 0.22656967, 0.22850613),
                        ),
                    ]
                ),
                id=i,
            )
            augmented_dataset = ConcatDataset(
                [private_trainset, private_extraset, query_dataset]
            )
            augmented_dataloader = DataLoader(
                augmented_dataset, batch_size=args.batch_size, shuffle=True,
                **kwargs
            )
            all_augmented_dataloaders.append(augmented_dataloader)
        return all_augmented_dataloaders
    elif args.dataset.startswith("cifar"):
        if args.dataset == "cifar10":
            datasets_cifar = datasets.CIFAR10
        elif args.dataset == "cifar100":
            datasets_cifar = datasets.CIFAR100
        else:
            raise Exception(args.datasets_exception)
        all_private_datasets = datasets_cifar(
            args.dataset_path,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.Pad(4),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.49139969, 0.48215842, 0.44653093),
                        (0.24703223, 0.24348513, 0.26158784),
                    ),
                ]
            ),
            download=True,
        )
        class_indices = get_class_indices(all_private_datasets, args)
        data_indices = [[] for i in range(args.num_models)]
        for c in range(args.num_classes):
            if c in args.weak_classes:
                size_weak = int(
                    args.weak_class_ratio * (
                            len(class_indices[c]) / args.num_models)
                )
                size_strong = (
                                      len(class_indices[
                                              c]) - size_weak * args.num_querying_parties
                              ) // (args.num_models - args.num_querying_parties)
                for i in range(args.num_models):
                    if i < args.num_querying_parties:
                        begin = i * size_weak
                        end = (i + 1) * size_weak
                        data_indices[i].append(class_indices[c][begin:end])
                    else:
                        begin = (
                                size_weak * args.num_querying_parties
                                + (i - args.num_querying_parties) * size_strong
                        )
                        end = (
                            len(class_indices[c])
                            if i == args.num_models - 1
                            else size_weak * args.num_querying_parties
                                 + (
                                         i + 1 - args.num_querying_parties) * size_strong
                        )
                        data_indices[i].append(class_indices[c][begin:end])
            else:
                size = len(class_indices[c]) // args.num_models
                for i in range(args.num_models):
                    begin = i * size
                    end = (
                        len(class_indices[c])
                        if i == args.num_models - 1
                        else (i + 1) * size
                    )
                    data_indices[i].append(class_indices[c][begin:end])
        data_indices = [
            np.concatenate(data_indices[i], axis=0) for i in
            range(args.num_models)
        ]
        assert sum(
            [len(data_indices[i]) for i in range(args.num_models)]) == len(
            all_private_datasets
        )
        assert len(set(np.concatenate(data_indices, axis=0))) == len(
            all_private_datasets
        )
        all_augmented_dataloaders = []
        for i in range(args.num_querying_parties):
            private_dataset = Subset(all_private_datasets, data_indices[i])
            query_dataset = QuerySet(
                args,
                transform=transforms.Compose(
                    [
                        transforms.Pad(4),
                        transforms.RandomCrop(32),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.49421429, 0.4851314, 0.45040911),
                            (0.24665252, 0.24289226, 0.26159238),
                        ),
                    ]
                ),
                id=i,
            )
            augmented_dataset = ConcatDataset([private_dataset, query_dataset])
            augmented_dataloader = DataLoader(
                augmented_dataset, batch_size=args.batch_size, shuffle=True,
                **kwargs
            )
            all_augmented_dataloaders.append(augmented_dataloader)
        return all_augmented_dataloaders

    elif args.dataset.startswith("chexpert"):
        all_private_datasets, _ = get_chexpert_all_private_datasets(args=args)
        class_indices = get_class_indices(all_private_datasets, args)
        data_indices = [[] for _ in range(args.num_models)]
        for c in range(args.num_classes):
            if c in args.weak_classes:
                size_weak = int(
                    args.weak_class_ratio * (
                            len(class_indices[c]) / args.num_models)
                )
                size_strong = (
                                      len(class_indices[
                                              c]) - size_weak * args.num_querying_parties
                              ) // (args.num_models - args.num_querying_parties)
                for i in range(args.num_models):
                    if i < args.num_querying_parties:
                        begin = i * size_weak
                        end = (i + 1) * size_weak
                    else:
                        begin = (
                                size_weak * args.num_querying_parties
                                + (i - args.num_querying_parties) * size_strong
                        )
                        if i == args.num_models - 1:
                            end = len(class_indices[c])
                        else:
                            end = (
                                    size_weak * args.num_querying_parties
                                    + (
                                            i + 1 - args.num_querying_parties) * size_strong
                            )
                    data_indices[i].append(class_indices[c][begin:end])
            else:
                size = len(class_indices[c]) // args.num_models
                for i in range(args.num_models):
                    begin = i * size
                    if i == args.num_models - 1:
                        end = len(class_indices[c])
                    else:
                        end = (i + 1) * size
                    data_indices[i].append(class_indices[c][begin:end])
        data_indices = [
            np.concatenate(data_indices[i], axis=0) for i in
            range(args.num_models)
        ]

        # Check that all the data items were distributed to all models.
        assert sum(
            [len(data_indices[i]) for i in range(args.num_models)]) == len(
            all_private_datasets
        )
        # Check that all data indices were used once.
        assert len(set(np.concatenate(data_indices, axis=0))) == len(
            all_private_datasets
        )

        all_augmented_dataloaders = []
        for i in range(args.num_querying_parties):
            private_dataset = Subset(all_private_datasets, data_indices[i])
            query_dataset = QuerySet(
                args,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                    ]
                ),
                id=i,
            )
            augmented_dataset = ConcatDataset([private_dataset, query_dataset])
            augmented_dataloader = DataLoader(
                augmented_dataset, batch_size=args.batch_size, shuffle=True,
                **kwargs
            )
            all_augmented_dataloaders.append(augmented_dataloader)
        return all_augmented_dataloaders
    elif args.dataset == "retinopathy":
        all_private_datasets, _ = get_retinopathy_train_data(args, mode="train")
        class_indices = get_class_indices(all_private_datasets, args)
        data_indices = [[] for _ in range(args.num_models)]
        for c in range(args.num_classes):
            if c in args.weak_classes:
                size_weak = int(
                    args.weak_class_ratio * (
                            len(class_indices[c]) / args.num_models)
                )
                size_strong = (
                                      len(class_indices[
                                              c]) - size_weak * args.num_querying_parties
                              ) // (args.num_models - args.num_querying_parties)
                for i in range(args.num_models):
                    if i < args.num_querying_parties:
                        begin = i * size_weak
                        end = (i + 1) * size_weak
                    else:
                        begin = (
                                size_weak * args.num_querying_parties
                                + (i - args.num_querying_parties) * size_strong
                        )
                        if i == args.num_models - 1:
                            end = len(class_indices[c])
                        else:
                            end = (
                                    size_weak * args.num_querying_parties
                                    + (
                                            i + 1 - args.num_querying_parties) * size_strong
                            )
                    data_indices[i].append(class_indices[c][begin:end])
            else:
                size = len(class_indices[c]) // args.num_models
                for i in range(args.num_models):
                    begin = i * size
                    if i == args.num_models - 1:
                        end = len(class_indices[c])
                    else:
                        end = (i + 1) * size
                    data_indices[i].append(class_indices[c][begin:end])
        data_indices = [
            np.concatenate(data_indices[i], axis=0) for i in
            range(args.num_models)
        ]
        assert sum(
            [len(data_indices[i]) for i in range(args.num_models)]) == len(
            all_private_datasets
        )
        assert len(set(np.concatenate(data_indices, axis=0))) == len(
            all_private_datasets
        )
        all_augmented_dataloaders = []
        for i in range(args.num_querying_parties):
            private_dataset = Subset(all_private_datasets, data_indices[i])
            query_dataset = QuerySet(
                args, transform=get_retinopathy_transform(args=args), id=i
            )
            augmented_dataset = ConcatDataset([private_dataset, query_dataset])
            augmented_dataloader = DataLoader(
                augmented_dataset, batch_size=args.batch_size, shuffle=True,
                **kwargs
            )
            all_augmented_dataloaders.append(augmented_dataloader)
        return all_augmented_dataloaders
    else:
        raise Exception(args.datasets_exception)


def load_private_data_imbalanced(args):
    """Load labeled private data for training private models in an imbalanced way."""
    kwargs = get_kwargs(args=args)
    if "mnist" in args.dataset:
        all_private_datasets = get_mnist_dataset(args=args, train=True)
        class_indices = get_class_indices(all_private_datasets, args)
        data_indices = [[] for _ in range(args.num_models)]
        for c in range(args.num_classes):
            if c in args.weak_classes:
                size_weak = int(
                    args.weak_class_ratio * (
                            len(class_indices[c]) / args.num_models)
                )
                size_strong = (
                                      len(class_indices[
                                              c]) - size_weak * args.num_querying_parties
                              ) // (args.num_models - args.num_querying_parties)
                for i in range(args.num_models):
                    if i < args.num_querying_parties:
                        begin = i * size_weak
                        end = (i + 1) * size_weak
                        data_indices[i].append(class_indices[c][begin:end])
                    else:
                        begin = (
                                size_weak * args.num_querying_parties
                                + (i - args.num_querying_parties) * size_strong
                        )
                        if i == args.num_models - 1:
                            end = len(class_indices[c])
                        else:
                            end = (
                                    size_weak * args.num_querying_parties
                                    + (
                                            i + 1 - args.num_querying_parties) * size_strong
                            )
                        data_indices[i].append(class_indices[c][begin:end])
            else:
                size = len(class_indices[c]) // args.num_models
                for i in range(args.num_models):
                    begin = i * size
                    if i == args.num_models - 1:
                        end = len(class_indices[c])
                    else:
                        end = (i + 1) * size
                    data_indices[i].append(class_indices[c][begin:end])

        data_indices = [
            np.concatenate(data_indices[i], axis=0) for i in
            range(args.num_models)
        ]

        assert sum(
            [len(data_indices[i]) for i in range(args.num_models)]) == len(
            all_private_datasets
        )
        assert len(set(np.concatenate(data_indices, axis=0))) == len(
            all_private_datasets
        )

        all_private_trainloaders = []
        for i in range(args.num_models):
            private_dataset = Subset(all_private_datasets, data_indices[i])
            private_trainloader = DataLoader(
                private_dataset, batch_size=args.batch_size, shuffle=True,
                **kwargs
            )
            all_private_trainloaders.append(private_trainloader)
        return all_private_trainloaders

    elif args.dataset == "svhn":
        trainset = datasets.SVHN(
            root=args.dataset_path,
            split="train",
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.43768212, 0.44376972, 0.47280444),
                        (0.19803013, 0.20101563, 0.19703615),
                    ),
                ]
            ),
            download=True,
        )
        extraset = datasets.SVHN(
            root=args.dataset_path,
            split="extra",
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.42997558, 0.4283771, 0.44269393),
                        (0.19630221, 0.1978732, 0.19947216),
                    ),
                ]
            ),
            download=True,
        )
        trainset_class_indices = get_class_indices(trainset, args)
        trainset_data_indices = [[] for _ in range(args.num_models)]
        for c in range(args.num_classes):
            if c in args.weak_classes:
                size_weak = int(
                    args.weak_class_ratio
                    * (len(trainset_class_indices[c]) / args.num_models)
                )
                size_strong = (
                                      len(trainset_class_indices[c])
                                      - size_weak * args.num_querying_parties
                              ) // (args.num_models - args.num_querying_parties)
                for i in range(args.num_models):
                    if i < args.num_querying_parties:
                        begin = i * size_weak
                        end = (i + 1) * size_weak
                        trainset_data_indices[i].append(
                            trainset_class_indices[c][begin:end]
                        )
                    else:
                        begin = (
                                size_weak * args.num_querying_parties
                                + (i - args.num_querying_parties) * size_strong
                        )
                        end = (
                            len(trainset_class_indices[c])
                            if i == args.num_models - 1
                            else size_weak * args.num_querying_parties
                                 + (
                                         i + 1 - args.num_querying_parties) * size_strong
                        )
                        trainset_data_indices[i].append(
                            trainset_class_indices[c][begin:end]
                        )
            else:
                size = len(trainset_class_indices[c]) // args.num_models
                for i in range(args.num_models):
                    begin = i * size
                    end = (
                        len(trainset_class_indices[c])
                        if i == args.num_models - 1
                        else (i + 1) * size
                    )
                    trainset_data_indices[i].append(
                        trainset_class_indices[c][begin:end]
                    )
        trainset_data_indices = [
            np.concatenate(trainset_data_indices[i], axis=0)
            for i in range(args.num_models)
        ]
        assert sum(
            [len(trainset_data_indices[i]) for i in range(args.num_models)]
        ) == len(trainset)
        assert len(set(np.concatenate(trainset_data_indices, axis=0))) == len(
            trainset)
        extraset_class_indices = get_class_indices(extraset, args)
        extraset_data_indices = [[] for i in range(args.num_models)]
        for c in range(args.num_classes):
            if c in args.weak_classes:
                size_weak = int(
                    args.weak_class_ratio
                    * (len(extraset_class_indices[c]) / args.num_models)
                )
                size_strong = (
                                      len(extraset_class_indices[c])
                                      - size_weak * args.num_querying_parties
                              ) // (args.num_models - args.num_querying_parties)
                for i in range(args.num_models):
                    if i < args.num_querying_parties:
                        begin = i * size_weak
                        end = (i + 1) * size_weak
                        extraset_data_indices[i].append(
                            extraset_class_indices[c][begin:end]
                        )
                    else:
                        begin = (
                                size_weak * args.num_querying_parties
                                + (i - args.num_querying_parties) * size_strong
                        )
                        if i == args.num_models - 1:
                            end = len(extraset_class_indices[c])
                        else:
                            end = (
                                    size_weak * args.num_querying_parties
                                    + (
                                            i + 1 - args.num_querying_parties) * size_strong
                            )
                        extraset_data_indices[i].append(
                            extraset_class_indices[c][begin:end]
                        )
            else:
                size = len(extraset_class_indices[c]) // args.num_models
                for i in range(args.num_models):
                    begin = i * size
                    if i == args.num_models - 1:
                        end = len(extraset_class_indices[c])
                    else:
                        end = (i + 1) * size
                    extraset_data_indices[i].append(
                        extraset_class_indices[c][begin:end]
                    )
        extraset_data_indices = [
            np.concatenate(extraset_data_indices[i], axis=0)
            for i in range(args.num_models)
        ]
        assert sum(
            [len(extraset_data_indices[i]) for i in range(args.num_models)]
        ) == len(extraset)
        assert len(set(np.concatenate(extraset_data_indices, axis=0))) == len(
            extraset)
        all_private_trainloaders = []
        for i in range(args.num_models):
            private_dataset = ConcatDataset(
                [
                    Subset(trainset, trainset_data_indices[i]),
                    Subset(extraset, extraset_data_indices[i]),
                ]
            )
            private_trainloader = DataLoader(
                private_dataset, batch_size=args.batch_size, shuffle=True,
                **kwargs
            )
            all_private_trainloaders.append(private_trainloader)
        return all_private_trainloaders

    elif args.dataset.startswith("cifar"):
        if args.dataset == "cifar10":
            datasets_cifar = datasets.CIFAR10
        elif args.dataset == "cifar100":
            datasets_cifar = datasets.CIFAR100
        else:
            raise Exception(args.datasets_exception)
        all_private_datasets = datasets_cifar(
            args.dataset_path,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.Pad(4),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.49139969, 0.48215842, 0.44653093),
                        (0.24703223, 0.24348513, 0.26158784),
                    ),
                ]
            ),
            download=True,
        )
        class_indices = get_class_indices(all_private_datasets, args)
        data_indices = [[] for _ in range(args.num_models)]
        for c in range(args.num_classes):
            if c in args.weak_classes:
                size_weak = int(
                    args.weak_class_ratio * (
                            len(class_indices[c]) / args.num_models)
                )
                size_strong = (
                                      len(class_indices[
                                              c]) - size_weak * args.num_querying_parties
                              ) // (args.num_models - args.num_querying_parties)
                for i in range(args.num_models):
                    if i < args.num_querying_parties:
                        begin = i * size_weak
                        end = (i + 1) * size_weak
                        data_indices[i].append(class_indices[c][begin:end])
                    else:
                        begin = (
                                size_weak * args.num_querying_parties
                                + (i - args.num_querying_parties) * size_strong
                        )
                        if i == args.num_models - 1:
                            end = len(class_indices[c])
                        else:
                            end = (
                                    size_weak * args.num_querying_parties
                                    + (
                                            i + 1 - args.num_querying_parties) * size_strong
                            )
                        data_indices[i].append(class_indices[c][begin:end])
            else:
                size = len(class_indices[c]) // args.num_models
                for i in range(args.num_models):
                    begin = i * size

                    if i == args.num_models - 1:
                        end = len(class_indices[c])
                    else:
                        end = (i + 1) * size
                    data_indices[i].append(class_indices[c][begin:end])
        data_indices = [
            np.concatenate(data_indices[i], axis=0) for i in
            range(args.num_models)
        ]
        assert sum(
            [len(data_indices[i]) for i in range(args.num_models)]) == len(
            all_private_datasets
        )
        assert len(set(np.concatenate(data_indices, axis=0))) == len(
            all_private_datasets
        )
        all_private_trainloaders = []
        for i in range(args.num_models):
            private_dataset = Subset(all_private_datasets, data_indices[i])
            private_trainloader = DataLoader(
                private_dataset, batch_size=args.batch_size, shuffle=True,
                **kwargs
            )
            all_private_trainloaders.append(private_trainloader)
        return all_private_trainloaders

    elif args.dataset.startswith("chexpert"):
        # The CheXpert dataset is already imbalanced.
        # return get_chexpert_all_private_trainloaders(args=args, kwargs=kwargs)
        all_private_datasets, _ = get_chexpert_all_private_datasets(args=args)
        class_indices = get_class_indices(all_private_datasets, args)
        data_indices = [[] for _ in range(args.num_models)]
        for c in range(args.num_classes):
            if c in args.weak_classes:
                size_weak = int(
                    args.weak_class_ratio * (
                            len(class_indices[c]) / args.num_models)
                )
                size_strong = (
                                      len(class_indices[
                                              c]) - size_weak * args.num_querying_parties
                              ) // (args.num_models - args.num_querying_parties)
                for i in range(args.num_models):
                    if i < args.num_querying_parties:
                        begin = i * size_weak
                        end = (i + 1) * size_weak
                    else:
                        begin = (
                                size_weak * args.num_querying_parties
                                + (i - args.num_querying_parties) * size_strong
                        )
                        if i == args.num_models - 1:
                            end = len(class_indices[c])
                        else:
                            end = (
                                    size_weak * args.num_querying_parties
                                    + (
                                            i + 1 - args.num_querying_parties) * size_strong
                            )
                    data_indices[i].append(class_indices[c][begin:end])
            else:
                size = len(class_indices[c]) // args.num_models
                for i in range(args.num_models):
                    begin = i * size
                    if i == args.num_models - 1:
                        end = len(class_indices[c])
                    else:
                        end = (i + 1) * size
                    data_indices[i].append(class_indices[c][begin:end])

        data_indices = [
            np.concatenate(data_indices[i], axis=0) for i in
            range(args.num_models)
        ]

        assert sum(
            [len(data_indices[i]) for i in range(args.num_models)]) == len(
            all_private_datasets
        )
        assert len(set(np.concatenate(data_indices, axis=0))) == len(
            all_private_datasets
        )

        all_private_trainloaders = []
        for i in range(args.num_models):
            private_dataset = Subset(all_private_datasets, data_indices[i])
            private_trainloader = DataLoader(
                private_dataset, batch_size=args.batch_size, shuffle=True,
                **kwargs
            )
            all_private_trainloaders.append(private_trainloader)
        return all_private_trainloaders
    elif args.dataset == "retinopathy":
        all_private_datasets, _ = get_retinopathy_train_data(args, mode="train")
        class_indices = get_class_indices(all_private_datasets, args)
        data_indices = [[] for _ in range(args.num_models)]
        for c in range(args.num_classes):
            if c in args.weak_classes:
                size_weak = int(
                    args.weak_class_ratio * (
                            len(class_indices[c]) / args.num_models)
                )
                size_strong = (
                                      len(class_indices[
                                              c]) - size_weak * args.num_querying_parties
                              ) // (args.num_models - args.num_querying_parties)
                for i in range(args.num_models):
                    if i < args.num_querying_parties:
                        begin = i * size_weak
                        end = (i + 1) * size_weak
                    else:
                        begin = (
                                size_weak * args.num_querying_parties
                                + (i - args.num_querying_parties) * size_strong
                        )
                        if i == args.num_models - 1:
                            end = len(class_indices[c])
                        else:
                            end = (
                                    size_weak * args.num_querying_parties
                                    + (
                                            i + 1 - args.num_querying_parties) * size_strong
                            )
                    data_indices[i].append(class_indices[c][begin:end])
            else:
                size = len(class_indices[c]) // args.num_models
                for i in range(args.num_models):
                    begin = i * size
                    if i == args.num_models - 1:
                        end = len(class_indices[c])
                    else:
                        end = (i + 1) * size
                    data_indices[i].append(class_indices[c][begin:end])
        data_indices = [
            np.concatenate(data_indices[i], axis=0) for i in
            range(args.num_models)
        ]
        assert sum(
            [len(data_indices[i]) for i in range(args.num_models)]) == len(
            all_private_datasets
        )
        assert len(set(np.concatenate(data_indices, axis=0))) == len(
            all_private_datasets
        )
        all_augmented_dataloaders = []
        for i in range(args.num_querying_parties):
            private_dataset = Subset(all_private_datasets, data_indices[i])
            query_dataset = QuerySet(
                args, transform=get_retinopathy_transform(args=args), id=i
            )
            augmented_dataset = ConcatDataset([private_dataset, query_dataset])
            augmented_dataloader = DataLoader(
                augmented_dataset, batch_size=args.batch_size, shuffle=True,
                **kwargs
            )
            all_augmented_dataloaders.append(augmented_dataloader)
        return all_augmented_dataloaders
    else:
        raise Exception(args.datasets_exception)


def load_private_data(args):
    """Load labeled private data for training private models."""
    kwargs = get_kwargs(args=args)
    args.kwargs = kwargs
    if args.dataset in ["mnist", "fashion-mnist"]:
        return get_mnist_private_data(args=args)
    elif args.dataset == "svhn":
        return get_svhn_private_data(args=args)
    elif args.dataset.startswith("cifar"):
        return get_cifar_private_data(args=args)
    elif args.dataset.startswith("chexpert"):
        return get_chexpert_private_data(args=args, kwargs=kwargs)
    elif args.dataset == "retinopathy":
        return get_retinopathy_private_data(args=args)
    elif args.dataset == "celeba":
        return get_celeba_private_data(args=args)
    elif args.dataset == "coco":
        return get_coco_private_data(args=args)
    elif args.dataset == "pascal":
        return get_pascal_private_data(args=args)
    elif args.dataset in args.xray_datasets:
        return get_xray_private_dataloaders(args=args)

    # return get_cxpert_debug_dataloaders(args=args)
    else:
        raise Exception(args.datasets_exception)


def load_ordered_unlabeled_data(args, indices, unlabeled_dataset):
    """Load unlabeled private data according to a specific order."""
    args.kwargs = get_kwargs(args=args)
    if args.dataset == args.xray_datasets:
        return load_ordered_unlabeled_xray(
            args=args, indices=indices, dataset=unlabeled_dataset
        )

    # A part of the original testset is loaded according to a specific order.
    unlabeled_dataset = Subset(unlabeled_dataset, indices)
    unlabeled_dataloader = DataLoader(
        unlabeled_dataset, batch_size=args.batch_size, shuffle=False,
        **args.kwargs
    )
    return unlabeled_dataloader


def get_train_set(args):
    """
    This is a previous approach where the unlabeled and test data together
    were kept together. However, it was too entangled.

    :param args:
    :return:
    """
    if "mnist" in args.dataset:
        dataset = get_mnist_dataset(args=args, train=True)
    elif args.dataset == "svhn":
        dataset = datasets.SVHN(
            root=args.dataset_path,
            split="train",
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.45242317, 0.45249586, 0.46897715),
                        (0.21943446, 0.22656967, 0.22850613),
                    ),
                ]
            ),
            download=True,
        )
    elif args.dataset.startswith("cifar"):
        if args.dataset == "cifar10":
            datasets_cifar = datasets.CIFAR10
        elif args.dataset == "cifar100":
            datasets_cifar = datasets.CIFAR100
        else:
            raise Exception(args.datasets_exception)
        dataset = datasets_cifar(
            root=args.dataset_path,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.49421429, 0.4851314, 0.45040911),
                        (0.24665252, 0.24289226, 0.26159238),
                    ),
                ]
            ),
            download=True,
        )
    elif args.dataset.startswith("chexpert"):
        dataset = get_chexpert_train_set(args=args)
    elif args.dataset == "retinopathy":
        dataset, _ = get_retinopathy_train_data(args, mode="train")
    elif args.dataset == "celeba":
        dataset = get_celeba_train_set(args=args)
    elif args.dataset == "coco":
        dataset = get_coco_train_set(args=args)
    elif args.dataset in args.xray_datasets:
        return get_xray_train_data(args=args)
    else:
        raise Exception(args.datasets_exception)
    return dataset


def get_non_train_set(args):
    """
    This is a previous approach where the unlabeled and test data together
    were kept together. However, it was too entangled.

    :param args:
    :return:
    """
    if "mnist" in args.dataset:
        dataset = get_mnist_dataset(args=args, train=False)
    elif args.dataset == "svhn":
        dataset = datasets.SVHN(
            root=args.dataset_path,
            split="test",
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.45242317, 0.45249586, 0.46897715),
                        (0.21943446, 0.22656967, 0.22850613),
                    ),
                ]
            ),
            download=True,
        )
    elif args.dataset.startswith("cifar"):
        if args.dataset == "cifar10":
            datasets_cifar = datasets.CIFAR10
        elif args.dataset == "cifar100":
            datasets_cifar = datasets.CIFAR100
        else:
            raise Exception(args.datasets_exception)
        dataset = datasets_cifar(
            root=args.dataset_path,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.49421429, 0.4851314, 0.45040911),
                        (0.24665252, 0.24289226, 0.26159238),
                    ),
                ]
            ),
            download=True,
        )
    elif args.dataset.startswith("chexpert"):
        dataset = get_chexpert_test_set(args=args, mode="train")
    elif args.dataset == "retinopathy":
        dataset, _ = get_retinopathy_train_data(args, mode="test")
    elif args.dataset == "celeba":
        dataset = get_celeba_test_set(args=args)
    elif args.dataset == "coco":
        dataset = get_coco_test_set(args=args)
    elif args.dataset in args.xray_datasets:
        # return get_xray_small_test_data(args=args)
        return get_xray_test_data(args=args)
    else:
        raise Exception(args.datasets_exception)
    return dataset


def get_test_set(args):
    """
    Get the REAL test set. This keeps the unlabeled and test data separately.

    :param args:
    :return: only the test data.
    """
    if args.dataset in args.xray_datasets:
        # return get_xray_small_test_data(args=args)
        return get_xray_test_data(args=args)


    elif args.dataset == "pascal":
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        val_transform = transforms.Compose(
            [
                # transforms.Scale(256),
                # transforms.CenterCrop(227),
                # transforms.CenterCrop(224),
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        # args.pascal_path = '/jmain01/home/JAD007/txk02/axs14-txk02/multi-label/VOC2012/'
        args.batch = 32
        args.pascal_path = os.path.join(args.data_dir, "VOC2012/")
        dataset = PascalDataLoader(args.pascal_path, "val",
                                   transform=val_transform)
        return dataset
    non_trained_set = get_non_train_set(args=args)
    if args.attacker_dataset == args.dataset:
        start = args.num_unlabeled_samples
    else:
        # TODO: this number is temporary (for mnist victim model only)
        start = 9000
    end = len(non_trained_set)
    assert end > start
    return Subset(dataset=non_trained_set, indices=list(range(start, end)))


def get_unlabeled_set(args):
    """
    Get the REAL unlabeled set.

    :param args:
    :return: only the unlabeled data.
    """
    if args.dataset in args.xray_datasets:
        return get_xray_unlabeled_data(args=args)

    elif args.dataset == "pascal":
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        val_transform = transforms.Compose(
            [
                # transforms.Scale(256),
                # transforms.CenterCrop(227),
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        # args.pascal_path = '/jmain01/home/JAD007/txk02/axs14-txk02/multi-label/VOC2012/'
        args.pascal_path = os.path.join(args.data_dir, "VOC2012/")
        args.batch = 32
        dataset = PascalDataLoader(args.pascal_path, "val",
                                   transform=val_transform)
        start = 0
        end = args.num_unlabeled_samples

        assert end > start
        subset = Subset(dataset=dataset, indices=list(range(start, end)))
        assert len(subset) == args.num_unlabeled_samples
        return subset

    non_trained_set = get_non_train_set(args=args)
    start = 0
    end = args.num_unlabeled_samples
    assert end > start
    subset = Subset(dataset=non_trained_set, indices=list(range(start, end)))
    assert len(subset) == args.num_unlabeled_samples
    return subset


def get_attacker_dataset(args, dataset_name):
    data_dir = args.data_dir
    if "mnist" in dataset_name:
        dataset = get_mnist_dataset_by_name(args, dataset_name, train=False)
    elif dataset_name == "svhn":
        svhn_transforms = []
        if "mnist" in args.dataset:
            # Transform SVHN images from the RGB to L - gray-scale 8 bit images.
            svhn_transforms.append(FromSVHNtoMNIST())
            svhn_transforms.append(transforms.ToTensor())
            # Normalize with the mean and std found for the new images.
            # This closely corresponds to the mean and std of the standard
            # values of mean and std for SVHN.
            svhn_transforms.append(
                transforms.Normalize((0.45771828,), (0.21816934,)))
            svhn_transforms.append(transforms.RandomCrop((28, 28)))
        else:
            svhn_transforms.append(transforms.ToTensor())
            svhn_transforms.append(
                transforms.Normalize(
                    (0.45242317, 0.45249586, 0.46897715),
                    (0.21943446, 0.22656967, 0.22850613),
                )
            )
        dataset_path = os.path.join(data_dir, "SVHN")
        dataset = datasets.SVHN(
            root=dataset_path,
            split="test",
            transform=transforms.Compose(svhn_transforms),
            download=True,
        )
    elif dataset_name.startswith("cifar"):
        if dataset_name == "cifar10":
            datasets_cifar = datasets.CIFAR10
            dataset_path = os.path.join(data_dir, "CIFAR10")
        elif dataset_name == "cifar100":
            datasets_cifar = datasets.CIFAR100
            dataset_path = os.path.join(data_dir, "CIFAR100")
        else:
            raise Exception(args.datasets_exception)
        dataset = datasets_cifar(
            root=dataset_path,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.49421429, 0.4851314, 0.45040911),
                        (0.24665252, 0.24289226, 0.26159238),
                    ),
                ]
            ),
            download=True,
        )
    elif dataset_name == "tinyimages":
        dataset = get_extra_cifar10_data_from_ti()
    elif dataset_name.startswith("chexpert"):
        dataset = get_chexpert_test_set(args=args, mode="train")
    elif dataset_name == "retinopathy":
        dataset, _ = get_retinopathy_train_data(args, mode="test")
    elif dataset_name == "celeba":
        dataset = get_celeba_test_set(args=args)
    elif dataset_name == "coco":
        dataset = get_coco_test_set(args=args)
    else:
        raise Exception(args.datasets_exception)
    return dataset


def get_unlabeled_perfectly_balanced_indices(dataset, args):
    """
    For each querying party get evenly distributed samples for each class.

    :param dataset: the test set
    :param args: arguments
    :return: indices for each querying party
    """
    data_indices = [[] for _ in range(args.num_querying_parties)]
    class_indices = get_class_indices(dataset, args)
    len_dataset = len(dataset)
    for c in range(args.num_classes):
        len_class = len(class_indices[c])
        last_index = get_unlabled_last_index(
            num_unlabeled_samples=args.num_unlabeled_samples,
            len_dataset=len_dataset,
            len_class=len_class,
        )
        c_class_indices = class_indices[c][:last_index]
        size = len(c_class_indices) // args.num_querying_parties
        for i in range(args.num_querying_parties):
            begin = i * size
            if i == args.num_querying_parties - 1:
                end = len(c_class_indices)
            else:
                end = (i + 1) * size
            data_indices[i].append(c_class_indices[begin:end])

    data_indices = [
        np.concatenate(data_indices[i], axis=0)
        for i in range(args.num_querying_parties)
    ]
    # Shuffle in place the content of indices per a querying party. This prevents us from seeing data with the same label in consecutive order and not seeing the some labels.
    """
    Without shuffle:
        Check query-answer pairs.
        Label counts: [101,   0,   0,   0,   0,   0,   0,   0,   0,   0]
        Class ratios: [100.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.]
        Number of samples: 101
        
    With shuffle:
        Check query-answer pairs.
        Label counts: [10, 15, 12, 18, 11, 11,  6, 14,  6, 17]
        Class ratios: [ 8.33, 12.5 , 10.  , 15.  ,  9.17,  9.17,  5.  , 11.67,  5.  , 14.17]
        Number of samples: 120
    """
    for i in range(args.num_querying_parties):
        np.random.shuffle(data_indices[i])
    return data_indices


def get_unlabeled_standard_indices(args):
    """
    Get the indices for each querying party from the test data.

    :param args: arguments
    :return: indices for each querying party
    """
    data_indices = [[] for _ in args.querying_parties]
    num_querying_parties = len(args.querying_parties)
    # Only a part of the original test set is used for the query selection.
    size = args.num_unlabeled_samples // num_querying_parties
    for i in range(num_querying_parties):
        begin = i * size
        # Is it the last querying party?
        if i == num_querying_parties - 1:
            end = args.num_unlabeled_samples
        else:
            end = (i + 1) * size
        indices = list(range(begin, end))
        data_indices[i] = indices
    return data_indices


def get_unlabeled_indices(args, dataset):
    if args.balance_type == "perfect":
        data_indices = get_unlabeled_perfectly_balanced_indices(
            args=args, dataset=dataset
        )
    elif args.balance_type == "standard":
        data_indices = get_unlabeled_standard_indices(args=args)
    else:
        raise Exception(f"Unknown balance type: {args.balance_type}.")

    num_querying_parties = len(args.querying_parties)
    # Test correctness of the computed indices by summations.
    assert (
            sum([len(data_indices[i]) for i in range(num_querying_parties)])
            == args.num_unlabeled_samples
    )
    assert len(
        set(np.concatenate(data_indices, axis=0))) == args.num_unlabeled_samples

    return data_indices


def load_dev_dataloader(args):
    kwargs = get_kwargs(args=args)
    if args.dataset == "celeba":
        dataset = get_celeba_dev_set(args=args)
    elif args.dataset in args.xray_datasets:
        dataset = get_chexpert_dev_set(args=args)

    elif args.dataset == "pascal":
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        val_transform = transforms.Compose(
            [
                # transforms.Scale(256),
                # transforms.CenterCrop(227),
                transforms.RandomSizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

        args.pascal_path = "/jmain01/home/JAD007/txk02/axs14-txk02/multi-label/VOC2012/"
        args.batch = 32
        dataset = PascalDataLoader(args.pascal_path, "test",
                                   transform=val_transform)

    else:
        raise Exception(f"Unsupported dataset: {args.dataset}.")
    dataloader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=False, **kwargs
    )
    return dataloader


def load_unlabeled_dataloaders(args, unlabeled_dataset=None):
    """
    Load unlabeled private data for query selection.
    :return: all_unlabeled_dataloaders data loaders for each querying party
    """
    kwargs = get_kwargs(args=args)
    if args.dataset in args.xray_datasets:
        return get_xray_unlabeled_dataloaders(
            args=args, unlabeled_dataset=unlabeled_dataset
        )

    all_unlabeled_dataloaders = []

    if unlabeled_dataset is None:
        unlabeled_dataset = get_unlabeled_set(args=args)

    unlabeled_indices = get_unlabeled_indices(args=args,
                                              dataset=unlabeled_dataset)
    # Create data loaders.
    for indices in unlabeled_indices:
        unlabeled_dataset = Subset(unlabeled_dataset, indices)
        unlabeled_dataloader = DataLoader(
            unlabeled_dataset, batch_size=args.batch_size, shuffle=False,
            **kwargs
        )
        all_unlabeled_dataloaders.append(unlabeled_dataloader)
    return all_unlabeled_dataloaders


def get_kwargs(args):
    kwargs = {"num_workers": args.num_workers,
              "pin_memory": True} if args.cuda else {}
    return kwargs


def load_training_data(args):
    """Load labeled data for training non-private baseline models."""
    kwargs = get_kwargs(args=args)
    if "mnist" in args.dataset:
        trainset = get_mnist_dataset(args=args, train=True)
        trainloader = DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, **kwargs
        )
    elif args.dataset == "svhn":
        trainset = datasets.SVHN(
            root=args.dataset_path,
            split="train",
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.43768212, 0.44376972, 0.47280444),
                        (0.19803013, 0.20101563, 0.19703615),
                    ),
                ]
            ),
            download=True,
        )
        extraset = datasets.SVHN(
            root=args.dataset_path,
            split="extra",
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.42997558, 0.4283771, 0.44269393),
                        (0.19630221, 0.1978732, 0.19947216),
                    ),
                ]
            ),
            download=True,
        )
        trainloader = DataLoader(
            ConcatDataset([trainset, extraset]),
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs,
        )
    elif args.dataset.startswith("cifar"):
        if args.dataset == "cifar10":
            datasets_cifar = datasets.CIFAR10
        elif args.dataset == "cifar100":
            datasets_cifar = datasets.CIFAR100
        else:
            raise Exception(args.datasets_exception)
        trainset = datasets_cifar(
            args.dataset_path,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.Pad(4),
                    transforms.RandomCrop(32),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.49139969, 0.48215842, 0.44653093),
                        (0.24703223, 0.24348513, 0.26158784),
                    ),
                ]
            ),
            download=True,
        )
        trainloader = DataLoader(
            trainset, batch_size=args.batch_size, shuffle=True, **kwargs
        )

    elif args.dataset.startswith("chexpert"):
        dataset_type = get_chexpert_dataset_type(args=args)
        dataset = dataset_type(
            in_csv_path=args.cfg.train_csv, cfg=args.cfg, mode="train"
        )
        # Only a part of the original train set is used for training.
        train_dataset = Subset(dataset, list(range(0, args.num_train_samples)))
        trainloader = DataLoader(
            train_dataset,
            batch_size=args.cfg.train_batch_size,
            drop_last=True,
            shuffle=True,
            **kwargs,
        )
    elif args.dataset == "retinopathy":
        train_dataset, _ = get_retinopathy_train_data(args, mode="train")
        trainloader = DataLoader(
            train_dataset,
            batch_size=args.cfg.train_batch_size,
            drop_last=True,
            shuffle=True,
            **kwargs,
        )
    else:
        raise Exception(args.datasets_exception)
    return trainloader


def load_evaluation_dataloader(args):
    """Load labeled data for evaluation."""
    kwargs = get_kwargs(args=args)
    dataset = get_test_set(args=args)
    evalloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, **kwargs
    )
    return evalloader


def load_unlabeled_dataloader(args):
    """Load all unlabeled data."""
    kwargs = get_kwargs(args=args)
    dataset = get_unlabeled_set(args=args)
    unlabeled_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, **kwargs
    )
    return unlabeled_loader


def get_unlabled_last_index(num_unlabeled_samples, len_dataset, len_class):
    """
    For example, for CIFAR100 we have len_dataset 10000 for test data.
    The number of samples per class is 1000.
    If we want 9000 unlabeled samples then the ratio_unlabeled is 9/10.
    The number of samples per class for the unlabeled dataset is 9/10*100=90.
    If the number of samples for the final test is 1000 samples and we have 100
    classes, then the number of samples per class will be 10 (only).

    :param num_unlabeled_samples: number of unlabeled samples from the test set
    :param len_dataset: the total number of samples in the intial test set
    :param len_class: the number of samples for a given class
    :return: for the array of sample indices for the class, the last index for
    the unlabeled part

    >>> num_unlabeled_samples = 9000
    >>> len_dataset = 10000
    >>> len_class = 100
    >>> result = get_unlabled_last_index(num_unlabeled_samples=num_unlabeled_samples, len_dataset=len_dataset, len_class=len_class)
    >>> assert result == 90
    >>> # print('result: ', result)
    """
    ratio_unlabeled = num_unlabeled_samples / len_dataset
    last_unlabeled_index = int(ratio_unlabeled * len_class)
    return last_unlabeled_index


def regularize_loss(model):
    loss = 0
    for param in list(model.children())[0].parameters():
        loss += 2e-5 * torch.sum(torch.abs(param))
    return loss


def get_loss_criterion(model, args):
    """
    Get the loss criterion.

    :param model: model
    :param args: arguments
    :return: the loss criterion (funciton like to be called)
    """
    if args.loss_type == "MSE":
        criterion = nn.MSELoss()
    elif args.loss_type == "BCE":
        criterion = nn.BCELoss()
    elif args.loss_type == "BCEWithLogits":
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss_type == "CE":
        criterion = nn.CrossEntropyLoss()
    elif args.loss_type == "AsymmetricLossOptimized":
        criterion = AsymmetricLossOptimized()
    elif args.loss_type == "MultiLabelSoftMarginLoss":
        weight = torch.tensor(
            [
                0.87713311,
                1.05761317,
                0.73638968,
                1.11496746,
                0.78593272,
                1.33506494,
                0.4732965,
                0.514,
                0.47548566,
                1.9469697,
                0.97348485,
                0.43670348,
                1.15765766,
                1.06639004,
                0.13186249,
                1.05544148,
                1.71906355,
                1.04684318,
                1.028,
                0.93624772,
            ]
        ).cuda()

        criterion = nn.MultiLabelSoftMarginLoss(weight=weight)
    else:
        raise Exception(f"Unknown loss type: {args.loss_type}.")

    if args.dataset == "retinopathy" and args.architecture == "RetinoNet":
        def criterion_custom(input, target):
            loss = criterion(input=input, target=target)
            loss += regularize_loss(model)

        criterion = criterion_custom

    return criterion


def pure_model(model):
    """
    Extract the proper model if enclosed in DataParallel (distributed model
    feature).

    :param model: a model
    :return: pure PyTorch model
    """
    if hasattr(model, "module"):
        return model.module
    else:
        return model


def add_regularization_loss(loss, args, model, data):
    # Regularize the weight matrix when label_concat is used.
    if args.label_concat_reg:
        if not args.label_concat:
            raise Exception("args.label_concat must be true")
        weight = pure_model(model).classifier.weight
        num_labels = len(default_pathologies)
        num_datasets = weight.shape[0] // num_labels
        weight_stacked = weight.reshape(num_datasets, num_labels, -1)
        label_concat_reg_lambda = torch.tensor(0.1).to(weight.device).float()
        for task in range(num_labels):
            dists = torch.pdist(weight_stacked[:, task], p=2).mean()
            loss += label_concat_reg_lambda * dists

    loss = loss.sum()

    if args.featurereg:
        feat = pure_model(model).features(data)
        loss += feat.abs().sum()

    if args.weightreg:
        loss += pure_model(model).classifier.weight.abs().sum()

    return loss


def task_loss(target, output, criterion, weights):
    """
    Compute the loss per task / label.

    :param target: target labels
    :param output: predicted labels
    :param criterion: loss criterion
    :param weights: the weight per task / label
    :return: the computed loss
    """
    loss = torch.zeros(1).to(output.device).to(torch.float32)
    for task in range(min(target.shape[1], output.shape[1])):
        task_output = output[:, task]
        task_target = target[:, task]
        mask = ~torch.isnan(task_target)
        task_output = task_output[mask]
        task_target = task_target[mask]
        if len(task_target) > 0:
            task_loss = criterion(task_output.float(), task_target.float())
            if weights is None:
                loss += task_loss
            else:
                loss += weights[task] * task_loss

    return loss


def get_task_weights(args, trainloader, device):
    """
    Compute the weights per task / label. This is done based on number of
    missing labels in the dataset.

    :param args: global args for the program
    :param trainloader: data loader for training
    :param device: where to put the weights
    :return: the computed weights
    """
    if args.taskweights is False:
        return None

    """
    trainloader.dataset.labels:
    
    Important detail: the labels have to be for this very dataset, subset,
    or merged set. If we go to the nested dataset in a subset, then we migth
    retrieve the labels for the whole dataset. However, we only should use
    labels from this very dataset (and not its parent datasets, or only
    a single child dataset, etc.)."
    """
    if not hasattr(trainloader.dataset, "labels"):
        print("dataset: ", trainloader.dataset)
        raise Exception("The dataset has to have labels for task weights.")

    labels = trainloader.dataset.labels

    if labels is None:
        raise Exception("The dataset has to have labels for task weights.")

    weights = np.nansum(labels, axis=0)

    if weights.max() == 0:
        # No missing labels at all. We do not need the weights.
        return None

    weights = weights.max() - weights + weights.mean()
    weights = weights / weights.max()
    weights = torch.from_numpy(weights).to(device).float()
    # print("task weights", weights)
    return weights


def compute_loss(target, output, criterion, weights, args, model, data):
    """
    Compute the loss.

    :param target: target labels
    :param output: predicted labels
    :param criterion: loss criterion
    :param weights: the weight per task / label
    :return: the computed loss
    """
    if args.dataset in args.xray_datasets:
        loss = task_loss(
            target=target, output=output, criterion=criterion, weights=weights
        )
        loss = add_regularization_loss(args=args, model=model, data=data,
                                       loss=loss)
        return loss
    else:
        loss = criterion(output, target)
        return loss


def train(model, trainloader, optimizer, criterion, args):
    """Train a given model on a given dataset using a given optimizer,
    loss criterion, and other arguments, for one epoch."""
    model.train()
    losses = []

    device, _ = get_device(args=args)

    weights = get_task_weights(args=args, trainloader=trainloader,
                               device=device)

    for batch_id, (data, target) in enumerate(trainloader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        if args.loss_type in {"MSE", "BCE", "BCEWithLogits"}:
            data = data.to(torch.float32)
            target = target.to(torch.float32)
        else:
            target = target.to(torch.long)

        optimizer.zero_grad()
        if args.dataset == "cxpert" and (
                args.retrain_fine_tune or args.use_pretrained_models):
            data = data.repeat(1, 3, 1, 1)  # Addition for retraining
        # print("SHAPE", data.size())
        output = model(data)

        loss = compute_loss(
            target=target,
            output=output,
            criterion=criterion,
            weights=weights,
            args=args,
            model=model,
            data=data,
        )

        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    train_loss = np.mean(losses)
    return train_loss


def evaluate_multiclass(model, dataloader, args):
    """
    Evaluation for standard multiclass classification.
    Evaluate metrics such as accuracy, detailed acc, balanced acc, auc of a given model on a given dataset.

    Accuracy detailed - evaluate the class-specific accuracy of a given model on a given dataset.

    :return:
    detailed_acc: A 1-D numpy array of length L = num-classes, containing the accuracy for each class.

    """
    model.eval()
    losses = []
    correct = 0
    total = len(dataloader.dataset)
    correct_detailed = np.zeros(args.num_classes, dtype=np.int64)
    wrong_detailed = np.zeros(args.num_classes, dtype=np.int64)
    raw_softmax = None
    raw_logits = None
    raw_preds = []
    raw_targets = []
    criterion = get_loss_criterion(model=model, args=args)
    with torch.no_grad():
        for data, target in dataloader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()

            output = model(data)

            loss = criterion(input=output, target=target)
            losses.append(loss.item())

            preds = output.data.argmax(axis=1)
            labels = target.data.view_as(preds)
            correct += preds.eq(labels).cpu().sum().item()
            softmax_outputs = F.softmax(output, dim=1)
            softmax_outputs = softmax_outputs.detach().cpu().numpy()
            outputd = output.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy().astype(int)
            preds_np = preds.detach().cpu().numpy().astype(int)

            if raw_softmax is None:
                raw_softmax = softmax_outputs
            else:
                raw_softmax = np.append(raw_softmax, softmax_outputs, axis=0)
            if raw_logits is None:
                raw_logits = outputd
            else:
                raw_logits = np.append(raw_logits, outputd, axis=0)
            raw_targets = np.append(raw_targets, labels_np)
            raw_preds = np.append(raw_preds, preds_np)

            for label, pred in zip(target, preds):
                if label == pred:
                    correct_detailed[label] += 1
                else:
                    wrong_detailed[label] += 1

    loss = np.mean(losses)

    acc = 100.0 * correct / total

    balanced_acc = metrics.balanced_accuracy_score(
        y_true=raw_targets,
        y_pred=raw_preds,
    )

    if (np.round(raw_softmax.sum(axis=1)) == 1).all() and raw_targets.size > 0:
        try:
            auc = get_auc(
                classification_type=args.class_type,
                y_true=raw_targets,
                y_pred=raw_softmax,
                num_classes=args.num_classes,
            )
        except ValueError as err:
            print("Error occurred: ", err)
            # Transform to list to print the full array.
            print("y_true: ", raw_targets.tolist())
            print("y_pred: ", raw_softmax.tolist())
            auc = 0
    else:
        auc = 0

    assert correct_detailed.sum() + wrong_detailed.sum() == total
    acc_detailed = 100.0 * correct_detailed / (
            correct_detailed + wrong_detailed)

    mAP_score = mAP(
        targs=one_hot_numpy(raw_targets.astype(np.int), args.num_classes),
        preds=one_hot_numpy(raw_preds.astype(np.int), args.num_classes),
    )

    result = {
        metric.loss: loss,
        metric.acc: acc,
        metric.balanced_acc: balanced_acc,
        metric.auc: auc,
        metric.acc_detailed: acc_detailed,
        metric.map: mAP_score,
    }

    return result


def evaluate_multilabel(model, dataloader, args):
    """
    Evaluate metrics such as accuracy, detailed acc, balanced acc, auc of a given model on a given dataset.

    Evaluation for args.class_type == 'multilabel' classification (assign 0 or 1 to each output value).
    E.g., for the CelebA dataset we have 40 output attributes (such as male, no beard, etc.) and for each input image we decide if a given attribute is
    absent (label 0) or absent (label 1).

    Accuracy detailed - evaluate the class-specific accuracy of a given model on a given dataset.

    :return:
    detailed_acc: A 1-D numpy array of length L = num-classes, containing the accuracy for each class.

    """
    model.eval()
    losses = []
    criterion = get_loss_criterion(model=model, args=args)
    if args.pick_labels is not None and args.pick_labels != [-1]:
        num_labels = len(args.pick_labels)
    else:
        num_labels = args.num_classes

    task_outputs = {}
    task_targets = {}
    for task in range(num_labels):
        task_outputs[task] = []
        task_targets[task] = []

    device, _ = get_device(args=args)

    weights = get_task_weights(args=args, trainloader=dataloader, device=device)

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(dataloader):
            if args.cuda:
                data = data.cuda().to(torch.float32)
                # if (
                #         args.commands == ["retrain_private_models"]
                #         or args.commands == ["train_private_models"]
                # ) and args.dataset == "cxpert":
                if args.dataset == "cxpert" and (args.commands != [
                    "test_models"] or args.use_pretrained_models) and (
                        args.retrain_fine_tune or args.use_pretrained_models):
                    data = data.repeat(1, 3, 1, 1)  # Addition for cxpert
                target = target.cuda().to(torch.float32)
                # print("target", target)

            output = model(data)
            # print("data_id", batch_idx)
            # print("data", data)
            # print("output",output)
            loss = compute_loss(
                target=target,
                output=output,
                criterion=criterion,
                weights=weights,
                args=args,
                model=model,
                data=data,
            )
            losses.append(loss.item())
            output = output.detach().cpu()

            if args.sigmoid_op == "apply":
                output = torch.sigmoid(output)
                # print("sigmoid applied")

            for task in range(num_labels):
                task_output = output[:, task]
                task_target = target[:, task]
                mask = ~torch.isnan(task_target)
                task_output = task_output[mask]
                task_target = task_target[mask]
                task_output = task_output.detach().cpu().numpy()
                task_target = task_target.detach().cpu().numpy()
                task_outputs[task].append(task_output)
                task_targets[task].append(task_target)
                # print("task output", task_output)
                # print("task target", task_target)

    loss = np.mean(losses)

    for task in range(num_labels):  # per task/label/class
        task_outputs[task] = np.concatenate(task_outputs[task])
        task_targets[task] = np.concatenate(task_targets[task])

    # if args.debug is True:
    #     # taus = np.linspace(0.0, 1.0, 100)
    #     taus = np.array(args.multilabel_prob_threshold)
    # else:
    #     taus = np.array(args.multilabel_prob_threshold)
    #
    # for tau in taus:
    #     args.multilabel_prob_threshold = tau
    #     metrics = compute_metrics_multilabel(
    #         args=args, task_outputs=task_outputs, task_targets=task_targets,
    #         loss=loss)
    #     print(tau, ',', metrics[metric.balanced_acc])
    metrics = compute_metrics_multilabel(
        args=args, task_outputs=task_outputs, task_targets=task_targets,
        loss=loss
    )

    return metrics


def compute_metrics_multilabel_from_preds_targets(args, preds, targets):
    if preds.shape != targets.shape:
        raise Exception(f"The shape of predictions preds is {preds.shape}"
                        f" while the shape of targets is {targets.shape}.")
    num_labels = preds.shape[1]

    task_outputs = {}
    task_targets = {}
    for task in range(num_labels):
        task_outputs[task] = []
        task_targets[task] = []

    for task in range(num_labels):
        task_output = preds[:, task]
        task_target = targets[:, task]
        mask = ~np.isnan(task_target)  # remove nan
        task_output = task_output[mask]

        task_target = task_target[mask]
        task_outputs[task] = task_output
        task_targets[task] = task_target
        # print("TARGETS", task_targets[task])
        # print("OUTPUTS", task_outputs[task])

    metrics = compute_metrics_multilabel(
        args=args, task_outputs=task_outputs, task_targets=task_targets,
        loss=None
    )
    return metrics


def compute_metrics_multilabel(
        args,
        task_outputs: Dict[int, np.array],
        task_targets: Dict[int, np.array],
        loss=None,
):
    if args.debug is True:
        name = f"_raw_{args.dataset}.npy"
        save_obj(file="outputs" + name, obj=task_outputs)
        save_obj(file="targets" + name, obj=task_targets)

    accs = []
    balanced_accs = []
    task_aucs = []
    task_maps = []

    # if args.pick_labels is not None and args.pick_labels != [-1]:
    #     task_outputs = pick_labels_rows(
    #         target_labels_index=args.pick_labels, labels=task_outputs
    #     )
    #     task_targets = pick_labels_rows(
    #         target_labels_index=args.pick_labels, labels=task_targets
    #     )

    num_labels = len(task_targets)
    threshold = np.array([args.multilabel_prob_threshold])

    if len(threshold) == 1:
        threshold = np.repeat(threshold, num_labels)

    tp = np.zeros(num_labels)
    fp = np.zeros(num_labels)
    fn = np.zeros(num_labels)
    tn = np.zeros(num_labels)

    for task in range(num_labels):
        targets = task_targets[task]
        outputs = task_outputs[
            task
        ]  # outputs are sigmoid/softmax i.e. alll are between 0 and 1.
        # print("TARGETS", targets)
        # print("OUTPUTS", outputs)
        assert not np.any(np.isnan(targets))
        if len(np.unique(targets)) > 1:
            # print("out1", outputs)
            task_auc = metrics.roc_auc_score(y_true=targets, y_score=outputs)
            task_map = average_precision(target=targets, output=outputs)
            if args.sigmoid_op != "apply":
                temp = torch.from_numpy(outputs)
                outputs = torch.sigmoid(temp)
                outputs = outputs.numpy()
                # print("out2", outputs)
            preds = outputs > threshold[task]
            balanced_acc = metrics.balanced_accuracy_score(y_true=targets,
                                                           y_pred=preds)
            acc = metrics.accuracy_score(y_true=targets, y_pred=preds)

            tp[task] += ((preds + targets) == 2).sum(axis=0)
            fp[task] += ((preds - targets) == 1).sum(axis=0)
            fn[task] += ((preds - targets) == -1).sum(axis=0)
            tn[task] += ((preds + targets) == 0).sum(axis=0)

            task_aucs.append(task_auc)
            task_maps.append(task_map)
            balanced_accs.append(balanced_acc)
            accs.append(acc)
        else:
            task_aucs.append(np.nan)
            task_maps.append(np.nan)
            balanced_accs.append(np.nan)
            accs.append(np.nan)

    acc, acc_detailed = get_metrics_processed(values=accs)
    balanced_acc, balanced_acc_detailed = get_metrics_processed(
        values=balanced_accs)
    auc, auc_detailed = get_metrics_processed(values=task_aucs)
    mAP_score, map_detailed = get_metrics_processed(values=task_maps)

    p_c, r_c, f_c = [], [], []
    for i in range(len(tp)):
        if tp[i] > 0:
            # precision
            pc = tp[i] / (tp[i] + fp[i]) * 100.0
            # recall
            rc = tp[i] / (tp[i] + fn[i]) * 100.0
            fc = 2 * pc * rc / (pc + rc)
        else:
            pc, rc, fc = 0.0, 0.0, 0.0
        p_c.append(pc)
        r_c.append(rc)
        f_c.append(fc)

    mean_p_c = sum(p_c) / len(p_c)
    mean_r_c = sum(r_c) / len(r_c)
    mean_f_c = sum(f_c) / len(f_c)

    p_o = tp.sum() / (tp + fp).sum() * 100.0
    r_o = tp.sum() / (tp + fn).sum() * 100.0
    f_o = 2 * p_o * r_o / (p_o + r_o)

    result = {
        metric.loss: loss,
        metric.acc: acc,
        metric.acc_detailed: acc_detailed,
        metric.balanced_acc: balanced_acc,
        metric.balanced_acc_detailed: balanced_acc_detailed,
        metric.auc: auc,
        metric.auc_detailed: auc_detailed,
        metric.map: mAP_score,
        metric.map_detailed: map_detailed,
        metric.pc: mean_p_c,
        metric.rc: mean_r_c,
        metric.fc: mean_f_c,
        metric.po: p_o,
        metric.ro: r_o,
        metric.fo: f_o,
    }

    return result


def get_metrics_processed(values):
    values = np.array(values)
    values = values[~np.isnan(values)]
    value = np.mean(values)
    return value, values


def one_hot(indices, num_classes: int) -> Tensor:
    """
    Convert labels into one-hot vectors.

    Args:
        indices: a 1-D vector containing labels.
        num_classes: number of classes.

    Returns:
        A 2-D matrix containing one-hot vectors, with one vector per row.
    """
    onehot = torch.zeros((len(indices), num_classes))
    for i in range(len(indices)):
        onehot[i][indices[i]] = 1
    return onehot


def one_hot_numpy(indices, num_classes: int) -> np.ndarray:
    """
    Convert labels into one-hot vectors.

    Args:
        indices: a 1-D vector containing labels.
        num_classes: number of classes.

    Returns:
        A 2-D matrix containing one-hot vectors, with one vector per row.
    """
    return one_hot(indices=indices, num_classes=num_classes).numpy()


def augmented_print(text, file, flush: bool = False) -> None:
    """Print to both the standard output and the given file."""
    assert isinstance(text, str)
    print(text)
    if isinstance(file, str):
        openfile = open(file, "a")
        openfile.write(text + "\n")
        if flush:
            sys.stdout.flush()
            openfile.flush()
        openfile.close()
    else:
        file.write(text + "\n")
        if flush:
            sys.stdout.flush()
            file.flush()


def extract_metrics(inputs):
    """
    Get only the keys and value for metrics.

    :param inputs: a dict
    :return: dict with metrics only
    """
    metric_keys = set(metric)
    outputs = {}
    for key, value in inputs.items():
        if key in metric_keys:
            outputs[key] = value
    return outputs


def from_result_to_str(
        result: Dict[metric, Union[int, float, str, np.ndarray, list]],
        sep: str = ";",
        inner_sep=";",
) -> str:
    """
    Transform the result in a form of a dict to a pretty string.

    :param result: result in a form of a dict
    :param sep: separator between key-value pairs
    :param inner_sep: separator between keys and values
    :return: pretty string
    """
    out = ""
    for key, value in result.items():
        if value is not None:
            out += str(key) + inner_sep
            out += get_value_str(value=value)
            out += sep
    return out


def get_value_str(value, separator=","):
    if isinstance(value, (int, float, str)):
        out = str(value)
    else:
        # print(__file__ + ' key: ', key)
        out = np.array2string(value, precision=4, separator=separator)
    return out


def update_summary(summary: dict, result: dict) -> dict:
    """
    Append values from result to summary.

    :param summary: summary dict (with aggregated results)
    :param result: result dict (with partial results)
    :return: the updated summary dict
    >>> a = {'a': 1, 'b': 2}
    >>> b = {'a': [3]}
    >>> c = update_summary(summary=b, result=a)
    >>> assert c['a'] == [3, 1]
    """
    for key, value in summary.items():
        if key in result.keys():
            new_value = result[key]
            summary[key].append(new_value)
    return summary


def class_ratio(dataset, args):
    """The ratio of each class in the given dataset."""
    counts = np.zeros(args.num_classes, dtype=np.int64)
    total_count = 0
    for data_index in range(len(dataset)):
        # Get dataset item.
        data_item = dataset[data_index]
        # Get labels.
        label = data_item[1]
        # This is for the multilabel classification.
        if args.class_type in [
            "multilabel",
            "multilabel_counting",
            "multilabel_tau",
            "multilabel_tau_dep",
            "multilabel_pate",
            "multilabel_tau_pate",
            "multilabel_powerset",
        ]:
            if torch.is_tensor(label):
                label = label.detach().cpu().numpy()
            label[np.isnan(label)] = 0
            label = label.astype(np.int64)
            counts += label
            total_count += sum(label)
            if args.debug:
                # assert {0, 1} == set(label)
                assert set(label).issubset({0, 1})
        # for class_index, value in enumerate(label):
        #     value = int(value)
        #     if value == 1:
        #         counts[class_index] += 1
        elif args.class_type in ["multiclass", "binary",
                                 "multiclass_confidence"]:
            index = int(label)
            counts[index] += 1
            total_count += 1
        else:
            raise Exception(f"Unknown class type: {args.class_type}.")

    assert counts.sum() == total_count
    return counts, 100.0 * counts / len(dataset)


def get_class_indices(dataset, args):
    """The indices of samples belonging to each class."""
    indices = [[] for _ in range(args.num_classes)]
    for i in range(len(dataset)):
        # dataset_i = dataset[i]
        # print('dataset[i]: ', dataset_i)
        # dataset_i_1 = dataset_i[1]
        # print('dataset[i][1]: ', dataset_i_1)
        # add index for a given class
        indices[dataset[i][1]].append(i)
    indices = [np.asarray(indices[i]) for i in range(args.num_classes)]

    # Double assert below to check if the number of collected indices for each class add up to total length of the dataset.
    assert sum([len(indices[i]) for i in range(args.num_classes)]) == len(
        dataset)
    assert len(set(np.concatenate(indices, axis=0))) == len(dataset)

    return indices


def get_scheduler(args, optimizer, trainloader=None):
    scheduler_type = args.scheduler_type
    # https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
    if scheduler_type == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            factor=args.schedule_factor,
            patience=args.schedule_patience,
        )
    # elif scheduler_type == 'OneCycleLR':
    #     if trainloader is None:
    #         raise Exception(
    #             'The trainloader has to be provided to scheduler for the OneCycleLR.')
    #     scheduler = OneCycleLR(
    #         optimizer=optimizer, max_lr=2e-4,
    #         epochs=args.num_epochs,
    #         steps_per_epoch=len(trainloader))
    elif scheduler_type == "MultiStepLR":
        milestones = args.scheduler_milestones
        if milestones is None:
            milestones = [
                int(args.num_epochs * 0.5),
                int(args.num_epochs * 0.75),
                int(args.num_epochs * 0.9),
            ]
        scheduler = MultiStepLR(
            optimizer=optimizer, milestones=milestones,
            gamma=args.schedule_factor
        )
    elif scheduler_type == "custom":
        scheduler = None
    else:
        raise Exception("Unknown scheduler type: {}".format(scheduler_type))
    return scheduler


def get_optimizer(params, args, lr=None):
    if lr is None:
        lr = args.lr
    if args.optimizer == "SGD":
        return SGD(
            params, lr=lr, momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == "Adadelta":
        return Adadelta(params, lr=lr, weight_decay=args.weight_decay)
    elif args.optimizer == "Adagrad":
        return Adagrad(params, lr=lr, weight_decay=args.weight_decay)
    elif args.optimizer == "Adam":
        return Adam(
            params, lr=lr, weight_decay=args.weight_decay,
            amsgrad=args.adam_amsgrad
        )
    elif args.optimizer == "RMSprop":
        return RMSprop(
            params, lr=lr, momentum=args.momentum,
            weight_decay=args.weight_decay
        )
    else:
        raise Exception("Unknown optimizer : {}".format(args.optimizer))


def distribute_model(args, model):
    device, device_ids = get_device(args=args)
    model = DataParallel(model, device_ids=device_ids).to(device)
    return model


def eval_distributed_model(args, model, dataloader):
    model = distribute_model(args=args, model=model)
    return eval_model(args=args, model=model, dataloader=dataloader)


def eval_model(args, model, dataloader):
    if args.class_type in ["multiclass", "binary"]:
        result = evaluate_multiclass(model=model, dataloader=dataloader,
                                     args=args)
    elif args.class_type in [
        "multilabel",
        "multilabel_counting",
        "multilabel_tau",
        "multilabel_tau_dep",
        "multilabel_tau_pate",
    ]:
        result = evaluate_multilabel(model=model, dataloader=dataloader,
                                     args=args)
    else:
        raise Exception(f"Unsupported args.class_type: {args.class_type}.")
    return result


def get_model_params(model, args):
    if args.architecture.startswith("tresnet"):
        return get_tresnet_params(
            model=model, lr=args.lr, weight_decay=args.weight_decay
        )
    else:
        return model.parameters()


def train_model(args, model, trainloader, evalloader, patience=None):
    device, device_ids = get_device(args=args)
    model = DataParallel(model, device_ids=device_ids).to(device).train()
    model_params = get_model_params(model=model, args=args)
    optimizer = get_optimizer(params=model_params, args=args)
    scheduler = get_scheduler(args=args, optimizer=optimizer,
                              trainloader=trainloader)
    criterion = get_loss_criterion(model=model, args=args)

    if patience is not None:
        # create variables for the patience mechanism
        best_loss = None
        patience_counter = 0

    start_epoch = 0
    save_model_path = getattr(args, "save_model_path", None)
    if save_model_path is not None:
        filename = "checkpoint-1.pth.tar"  # .format(model.module.name)
        filepath = os.path.join(save_model_path, filename)

        if os.path.exists(filepath):
            checkpoint = torch.load(filepath)
            if hasattr(model, "module"):
                model.module.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint["state_dict"])
            start_epoch = checkpoint["epoch"]
            print(
                "Restarted from checkpoint file {} at epoch {}".format(
                    filepath, start_epoch
                )
            )
    print("STARTED TRAINING")
    for epoch in range(start_epoch, args.num_epochs):
        start = time.time()
        train_loss = train(
            model=model,
            trainloader=trainloader,
            args=args,
            optimizer=optimizer,
            criterion=criterion,
        )
        # Scheduler step is based only on the train data, we do not use the
        # test data to schedule the decrease in the learning rate.
        if args.scheduler_type == "OneCycleLR":
            scheduler.step()
        else:
            scheduler.step(train_loss)
        stop = time.time()
        epoch_time = stop - start

        if patience is not None:
            result_test = train_model_log(
                args=args,
                epoch=epoch,
                model=model,
                epoch_time=epoch_time,
                trainloader=trainloader,
                evalloader=evalloader,
            )
            if result_test is None:
                raise Exception(
                    "Fatal Error, result should not be None after training model"
                )
            if best_loss is None or best_loss < result_test[metric.loss]:
                best_loss = result_test[metric.loss]
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        else:
            train_model_log(
                args=args,
                epoch=epoch,
                model=model,
                epoch_time=epoch_time,
                trainloader=trainloader,
                evalloader=evalloader,
            )


def bayesian_optimization_training_loop(
        args, model, adaptive_dataset, evalloader, patience=10,
        num_optimization_loop=20
):
    train_model_function = train_with_bayesian_optimization(
        args, adaptive_dataset, evalloader, patience=patience
    )

    best_parameters, values, _, _ = optimize(
        parameters=[
            {
                "name": "lr",
                "type": "range",
                "log_scale": True,
                "bounds": [args.lr / 1000, args.lr * 10],
            },
            {
                "name": "batch_size",
                "type": "range",
                "value_type": "int",
                "log_scale": True,
                "bounds": [
                    max(1, int(args.batch_size / 16)),
                    max(int(args.batch_size * 16), 128),
                ],
            },
        ],
        objective_name="val_loss",
        evaluation_function=train_model_function,
        minimize=True,
        total_trials=num_optimization_loop,
    )

    return best_parameters


def train_with_bayesian_optimization(args, adaptive_dataset, evalloader,
                                     patience=10):
    device, device_ids = get_device(args=args)

    def train_model_wrapper_for_bayesian_optimization(parameters):
        # create a new model
        model = get_private_model_by_id(args=args, id=0)
        model = DataParallel(model, device_ids=device_ids).to(device).train()
        model_params = get_model_params(model=model, args=args)
        lr = parameters.get("lr")
        batch_size = parameters.get("batch_size")

        trainloader = DataLoader(
            adaptive_dataset, batch_size=batch_size, shuffle=False,
            **args.kwargs
        )

        optimizer = get_optimizer(params=model_params, args=args, lr=lr)
        scheduler = get_scheduler(
            args=args, optimizer=optimizer, trainloader=trainloader
        )
        criterion = get_loss_criterion(model=model, args=args)

        # create variables for the patience mechanism
        best_loss = None
        patience_counter = 0
        result_test = {}

        start_epoch = 0
        save_model_path = getattr(args, "save_model_path", None)
        if save_model_path is not None:
            filename = "checkpoint-1.pth.tar"  # .format(model.module.name)
            filepath = os.path.join(save_model_path, filename)

            if os.path.exists(filepath) and args.retrain_extracted_model:
                try:
                    checkpoint = torch.load(filepath)
                    if hasattr(model, "module"):
                        model.module.load_state_dict(checkpoint["state_dict"])
                    else:
                        model.load_state_dict(checkpoint["state_dict"])
                    print(
                        "Restarted from checkpoint file {} at iteration {}".format(
                            filepath, checkpoint["epoch"]
                        )
                    )
                except Exception:
                    print(
                        "find trained model but cannot read, may be a model from previous generation of code"
                    )
                    print("train from scratch instead")
        print("STARTED TRAINING")
        for epoch in range(0, args.num_epochs):
            start = time.time()
            train_loss = train(
                model=model,
                trainloader=trainloader,
                args=args,
                optimizer=optimizer,
                criterion=criterion,
            )
            # Scheduler step is based only on the train data, we do not use the
            # test data to schedule the decrease in the learning rate.
            if args.scheduler_type == "OneCycleLR":
                scheduler.step()
            else:
                scheduler.step(train_loss)
            stop = time.time()
            epoch_time = stop - start

            result_test = train_model_log(
                args=args,
                epoch=epoch,
                model=model,
                epoch_time=epoch_time,
                trainloader=trainloader,
                evalloader=evalloader,
            )
            if not result_test:
                raise Exception(
                    "Fatal Error, result should not be None after training model"
                )

            if epoch == 0:
                # record best loss
                best_loss = result_test[metric.loss]
            if patience is not None:
                if best_loss > result_test[metric.loss]:
                    patience_counter = 0
                    best_loss = result_test[metric.loss]
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

        balanced_acc = result_test.get(metric.balanced_acc, 0)
        print(
            f"lr:{lr}, "
            f"bs:{batch_size}, "
            f"val_loss:{best_loss}, "
            f"balanced acc:{balanced_acc}"
        )

        return best_loss

    return train_model_wrapper_for_bayesian_optimization


def train_with_bayesian_optimization_with_best_hyperparameter(
        args, model, adaptive_dataset, evalloader, parameters, patience=10
):
    device, device_ids = get_device(args=args)

    model = DataParallel(model, device_ids=device_ids).to(device).train()
    model_params = get_model_params(model=model, args=args)
    lr = parameters.get("lr")
    batch_size = parameters.get("batch_size")

    trainloader = DataLoader(
        adaptive_dataset, batch_size=batch_size, shuffle=False, **args.kwargs
    )

    optimizer = get_optimizer(params=model_params, args=args, lr=lr)
    scheduler = get_scheduler(args=args, optimizer=optimizer,
                              trainloader=trainloader)
    criterion = get_loss_criterion(model=model, args=args)

    # create variables for the patience mechanism
    best_loss = None
    patience_counter = 0
    result_test = {}

    start_epoch = 0
    save_model_path = getattr(args, "save_model_path", None)
    if save_model_path is not None:
        filename = "checkpoint-1.pth.tar"  # .format(model.module.name)
        filepath = os.path.join(save_model_path, filename)

        if os.path.exists(filepath) and args.retrain_extracted_model:
            try:
                checkpoint = torch.load(filepath)
                if hasattr(model, "module"):
                    model.module.load_state_dict(checkpoint["state_dict"])
                else:
                    model.load_state_dict(checkpoint["state_dict"])
                print(
                    "Restarted from checkpoint file {} at iteration {}".format(
                        filepath, checkpoint["epoch"]
                    )
                )
            except Exception:
                print(
                    "find trained model but cannot read, may be a model from previous generation of code"
                )
                print("train from scratch instead")
    print("STARTED TRAINING")
    for epoch in range(0, args.num_epochs):
        start = time.time()
        train_loss = train(
            model=model,
            trainloader=trainloader,
            args=args,
            optimizer=optimizer,
            criterion=criterion,
        )
        # Scheduler step is based only on the train data, we do not use the
        # test data to schedule the decrease in the learning rate.
        if args.scheduler_type == "OneCycleLR":
            scheduler.step()
        else:
            scheduler.step(train_loss)
        stop = time.time()
        epoch_time = stop - start

        result_test = train_model_log(
            args=args,
            epoch=epoch,
            model=model,
            epoch_time=epoch_time,
            trainloader=trainloader,
            evalloader=evalloader,
        )
        if not result_test:
            raise Exception(
                "Fatal Error, result should not be None after training model"
            )

        if epoch == 0 or best_loss < result_test[metric.loss]:
            # record best loss
            best_loss = result_test[metric.loss]
        if patience is not None:
            if best_loss < result_test[metric.loss]:
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

    balanced_acc = result_test.get(metric.balanced_acc, 0)
    print(
        f"lr:{lr}, "
        f"bs:{batch_size}, "
        f"val_loss:{best_loss}, "
        f"balanced acc:{balanced_acc}"
    )
    return model


def train_model_log(args, epoch, epoch_time, model, trainloader, evalloader):
    log_every = args.log_every_epoch
    print("EPOCH: ", epoch)
    if log_every != 0 and epoch % log_every == 0:
        start_time = time.time()
        result_train = eval_model(model=model, dataloader=trainloader,
                                  args=args)
        result_test = eval_model(model=model, dataloader=evalloader, args=args)
        stop_time = time.time()
        eval_time = stop_time - start_time
        if epoch == 0:
            header = [
                "epoch",
                "train_" + str(metric.loss),
                "test_" + str(metric.loss),
                "train_" + str(metric.balanced_acc),
                "test_" + str(metric.balanced_acc),
                "train_" + str(metric.auc),
                "test_" + str(metric.auc),
                "train_" + str(metric.map),
                "test_" + str(metric.map),
                "eval_time",
                "epoch_time",
            ]
            header_str = args.sep.join(header)
            print(header_str)
            best_loss = result_test[metric.loss]
        data = [
            epoch,
            result_train[metric.loss],
            result_test[metric.loss],
            result_train[metric.balanced_acc],
            result_test[metric.balanced_acc],
            result_train[metric.auc],
            result_test[metric.auc],
            result_train[metric.map],
            result_test[metric.map],
            eval_time,
            epoch_time,
        ]
        data_str = args.sep.join([str(f"{x:.4f}") for x in data])
        print(data_str)

        # Checkpoint
        save_model(args, model, result_test=result_test)

    save_model(args, model, result_test=None)

    try:
        return result_test
    except NameError:
        return eval_model(model=model, dataloader=evalloader, args=args)


def save_model(args, model, epoch=-1, result_test=None):
    save_model_path = getattr(args, "save_model_path", None)
    if save_model_path is not None:
        if result_test is not None:
            state = result_test
        else:
            state = {}
        raw_model = getattr(model, "module", None)
        if raw_model is None:
            raw_model = model
        state["state_dict"] = raw_model.state_dict()
        if epoch == -1:
            epoch = args.num_epochs
        state["epoch"] = epoch
        filename = "checkpoint-{}.pth.tar".format("resnet50")  # raw_model.name)
        filepath = os.path.join(save_model_path, filename)
        torch.save(state, filepath)


def pick_labels_cols_get_axis(labels):
    """

    Args:
        labels: the total storage for the labels (votes)

    Returns: axis where the labels are

    """
    labels_num_dimensions = len(labels.shape)
    assert labels_num_dimensions > 0
    if labels_num_dimensions == 1:
        axis = 0
    else:
        axis = 1
    return axis


def pick_labels_cols(target_labels_index: List[int],
                     labels: np.ndarray, axis: int = None) -> np.array:
    if axis is None:
        axis = pick_labels_cols_get_axis(labels=labels)
    return np.take(labels, indices=target_labels_index, axis=axis)
    return target_labels


def pick_labels_cols_torch(target_labels_index: List[int],
                           labels: torch.tensor) -> torch.tensor:
    dim = pick_labels_cols_get_axis(labels=labels)
    return labels.index_select(dim=dim,
                               index=torch.LongTensor(target_labels_index))


def retain_labels_cols(target_labels_index: List[int],
                       labels: np.array) -> np.array:
    assert len(labels.shape) > 1
    num_cols = labels.shape[1]
    target_labels_index = set(target_labels_index)
    for idx in range(num_cols):
        if idx not in target_labels_index:
            labels[:, idx] = np.nan
    return labels


def pick_labels_rows(
        target_labels_index: List[int], labels: Dict[int, np.array]
) -> Dict[int, np.array]:
    target_labels = {}
    for target_idx, label_idx in enumerate(target_labels_index):
        target_labels[target_idx] = labels[label_idx]
    return target_labels


def print_metrics_detailed(results):
    """
    Print the 4 main metrics detailed in 4 columns.
    :param results: the results with
    """
    arrays = []
    for metric_key in [
        metric.acc_detailed,
        metric.balanced_acc_detailed,
        metric.auc_detailed,
        metric.map_detailed,
    ]:
        arrays.append(results.get(metric_key, None))
    arrays = [x for x in arrays if x is not None]
    expanded_arrays = []
    for array in arrays:
        expanded_arrays.append(np.expand_dims(array, 1))
    all = np.concatenate(expanded_arrays, axis=1)
    print("Print for each label separately: ")
    print("acc,bac,auc,map")
    all_str = "\n".join(",".join(str(y) for y in x) for x in all)
    print(all_str)


def from_confidence_scores_to_votes(confidence_scores, args):
    """

    Args:
        confidence_scores: confidence scores for each teacher and data point of
            size (num_teachers, num_data_points, num_classes)
        args: program arguments

    Returns:
        votes for each class from the teachers of size
            (num_data_points, num_classes)

    """
    # Index of the class that was voted for.
    votes_indices = confidence_scores.argmax(axis=-1)
    num_teachers, num_points = votes_indices.shape
    votes = np.zeros((num_points, args.num_classes))
    for teacher_id in range(num_teachers):
        votes_teacher = one_hot_numpy(
            indices=votes_indices[teacher_id], num_classes=args.num_classes
        )
        votes += votes_teacher
    return votes


def get_one_hot_confidence_bins(args, confidence_scores, votes):
    """

    Args:
        args: program arguments
        confidence_scores: softmax confidence scores for each data point
            shape: (num_teachers, num_data_points, num_classes).
        votes: number of votes for each data points and class from the teachers.
            shape: (num_data_points, num_classes)

    Returns:
        bin for the confident score that the wining class

    """
    num_teachers, num_data_points, num_classes = confidence_scores.shape
    assert num_classes == args.num_classes
    num_confidence_bins = args.bins_confidence
    bins = np.arange(num_confidence_bins + 1) / num_confidence_bins
    # winning_classes for each data point (most teachers voted for these
    # classes)
    winning_classes = votes.argmax(axis=-1)
    one_hot_confidence_bins = np.zeros((num_data_points, num_classes))
    for data_id, win_class_id in enumerate(winning_classes):
        # Extract the softmax value for the winning vote.
        confidence_score = confidence_scores[:, data_id, win_class_id]
        # np.histogram is computed only over a flattened array.
        one_hot_confidence_bin, _ = np.histogram(a=confidence_score, bins=bins)
        if not np.sum(one_hot_confidence_bin) == num_teachers:
            raise Exception(
                "There should be as many votes for confidence bins as there are"
                " teachers."
            )
        one_hot_confidence_bins[data_id] = one_hot_confidence_bin

    return one_hot_confidence_bins


def non_cumulative(array: np.ndarray) -> np.ndarray:
    """
    Transform from cumulative to selective values.

    Args:
        array: input cumulative values

    Returns:
        array with selective values
    >>> a = np.array([1,2,3])
    >>> b = non_cumulative(a)
    >>> assert np.array_equal(np.array([1,1,1]), b)
    """
    result = np.zeros_like(array)
    result[0] = array[0]
    for i in range(1, len(array)):
        result[i] = array[i] - array[i - 1]
    return result


@njit
def to_str(a: np.ndarray) -> str:
    return ",".join([str(x) for x in a])


def from_str(a: str) -> np.ndarray:
    return [int(x) for x in a.split(",")]


def get_bin(x: int, n: int) -> str:
    """

    Args:
        x: the input number
        n: total number of bits

    Returns: n-bit representation of number x.

    """
    return format(x, 'b').zfill(n)


def fill_v_key_count_powerset(num_labels: int) -> (
        np.ndarray, dict):
    """
    Map indexes (simple 0,1,2, etc.) to the strings representing the powerset
    labels. The votes are for indexes and then we can re-map from the winning
    index to the binary vector of {0,1} labels.

    Args:
        num_labels: the number of labels
        is_indexed: put the indexes of the labels as values of dictionary.
        Otherwise value 0 is assigned to each label and this can be used as a
        starting counter to count how many teachers vote for a given label.

    Returns:
        ordered all possible class labels and a dictionary with mapping between
        the class indexes and class labels.

    """
    # Map from binary label vectors to the initial zero count (to be filled in
    # with counts of how many times a given (class) label is voted for).
    class_labels = []
    # Map from the idx in the histogram to a binary label vector.
    map_idx_label = {}
    for idx, label in enumerate(range(2 ** num_labels)):
        # print(f'map idx to class label: {idx}')
        bit_label = get_bin(x=label, n=num_labels)
        # Transform bit label to string with every bit separated by a comma.
        str_label = ",".join([x for x in bit_label])
        # For the value in the dictionary put either index of the class or
        # starting counter 0.
        class_labels.append(str_label)
        map_idx_label[idx] = str_label
    return np.array(class_labels), map_idx_label


def transform_votes_powerset(input_votes: np.ndarray) -> (
        np.ndarray, dict):
    """
    Transform votes from (num_models, num_samples, num_labels) to
    (num_samples, dict[label: num votes]).

    Args:
        input_votes: initial votes.

    Returns: transformed votes.
    """

    # Votes in a standard form with dimension (num_sample, histogram).
    num_samples, num_models, num_labels = input_votes.shape
    print('num_samples: ', num_samples)
    print('num_models: ', num_models)
    print('num_labels: ', num_labels)

    start = time.time()
    class_labels, map_idx_label = fill_v_key_count_powerset(
        num_labels=num_labels)
    stop = time.time()
    print(f'Elapsed time to map index to str label: {stop - start} (sec).')
    start = time.time()
    histograms = generate_histograms_powerset(input_votes=input_votes,
                                              class_labels=class_labels)
    stop = time.time()
    print(f'Elapsed time to create histograms: {stop - start} (sec).')
    return histograms, map_idx_label


def generate_histograms_powerset(input_votes: np.ndarray,
                                 class_labels: np.ndarray) -> np.ndarray:
    """
    Transform votes from (num_samples, num_models, num_labels) to
    (num_samples, dict[label: num votes]).

    Args:
        input_votes: initial votes.
        class_labels: all possible classes from labels


    Returns: histograms: histogram for each sample.
    """
    # labels_counts: map labels to counts (the counts are initialized to zeros
    #       to be used to count how many times a given label was voted for).
    labels_counts = {}
    for class_label in class_labels:
        labels_counts[class_label] = 0
    histograms = []
    num_samples, num_models, _ = input_votes.shape
    # for sample in tqdm(range(num_samples)):
    for sample in range(num_samples):
        # print(f'sample number: {sample}')
        used_labels = set()
        for model in range(num_models):
            labels = to_str(input_votes[sample][model])
            used_labels.add(labels)
            labels_counts[labels] += 1
        histogram = np.array(list(labels_counts.values()))
        histograms.append(histogram)
        for labels in used_labels:
            # Reset the counters to reuse the labels_counts dictionary.
            labels_counts[labels] = 0
    histograms = np.array(histograms)
    return histograms


def generate_histogram_powerset(input_votes: np.ndarray,
                                class_labels: np.ndarray) -> np.ndarray:
    """
    Transform votes from (num_models, num_labels) to
    dict[label: num votes].

    Args:
        input_votes: initial votes.
        class_labels: all possible classes from labels


    Returns: histogram: histogram for the sample
    """
    # labels_counts: map labels to counts (the counts are initialized to zeros
    #       to be used to count how many times a given label was voted for).
    labels_counts = {}
    for class_label in class_labels:
        labels_counts[class_label] = 0
    num_models = input_votes.shape[0]
    for model in range(num_models):
        labels = to_str(input_votes[model])
        labels_counts[labels] += 1
    histogram = np.array(list(labels_counts.values()))
    return histogram


def get_multilabel_powerset_filenames(args):
    if args.pick_labels is None:
        pick_labels = ''
    else:
        pick_labels = len(args.pick_labels)

    prefix = f"{args.command}_multilabel_powerset_{args.dataset}_" \
             f"num_labels_{pick_labels}"
    vote_count_filename = f"{prefix}_vote_count.npy"
    class_labels_filename = f"{prefix}_class_labels.npy"

    return vote_count_filename, class_labels_filename


def get_class_labels_and_map_powerset(args, num_labels,
                                      class_labels_filename=None):
    if class_labels_filename is None:
        _, class_labels_filename = get_multilabel_powerset_filenames(args=args)
    if os.path.isfile(class_labels_filename):
        # print(
        #     f"load multilabel powerset iterative class labels: {class_labels_filename}")
        start = time.time()
        class_labels = np.load(class_labels_filename)
        map_idx_label = {}
        for idx, class_label in enumerate(class_labels):
            map_idx_label[idx] = class_label
        stop = time.time()
        # It takes about 150 sec to generate max_idx_label for Pascal VOC dataset.
        # print(f"Elapsed time: {stop - start} (sec).")
    else:
        start = time.time()
        class_labels, map_idx_label = fill_v_key_count_powerset(
            num_labels=num_labels)
        stop = time.time()
        print(f'Generation time for class labels: {stop - start} (sec).')

        class_labels = map_idx_label.values()
        class_labels = np.array(list(class_labels))
        np.save(class_labels_filename, class_labels)

    return class_labels, map_idx_label


def get_vote_count_and_map_powerset(args, votes_all):
    vote_count_filename, class_labels_filename = get_multilabel_powerset_filenames(
        args=args)

    num_labels = votes_all.shape[-1]
    class_labels, map_idx_label = get_class_labels_and_map_powerset(
        args=args, num_labels=num_labels,
        class_labels_filename=class_labels_filename)

    # Retrieve values from the cache if we need them for all the test samples.
    # Otherwise, the number of queries are limited for the evaluation of the
    # big ensemble so we do not need to retrieve it from cache. Additinoally,
    # for the query part, not sequential samples might be chosen so we cannot
    # rely on the cache but have to recompute the vote_counts for the
    # histograms.
    if args.command == 'evaluate_big_ensemble_model':
        if os.path.isfile(vote_count_filename):
            print(
                f'load multilabel powerset vote counts: {vote_count_filename}')
            start = time.time()
            vote_count = np.load(vote_count_filename)
            stop = time.time()
            # It takes about 420 sec (7 min) to generate vote_count.
            print(f"Elapsed time: {stop - start} (sec).")
        else:
            vote_count = generate_histograms_powerset(
                input_votes=votes_all, class_labels=class_labels)
            np.save(vote_count_filename, vote_count)
    else:
        vote_count = generate_histograms_powerset(
            input_votes=votes_all, class_labels=class_labels)

    return vote_count, map_idx_label


def test_bin():
    labels_counts, map_idx_label = fill_v_key_count_powerset(num_labels=2)
    print('labels_counts: ', labels_counts)
    print('map_idx_label: ', map_idx_label)


if __name__ == "__main__":
    test_bin()

# if __name__ == "__main__":
#     import doctest
#
#     doctest.testmod()
