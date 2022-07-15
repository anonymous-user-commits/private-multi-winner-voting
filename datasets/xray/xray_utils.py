from enum import Enum

import numpy as np
import torchvision
from torch.utils.data import DataLoader

from datasets.xray.dataset_pathologies import default_pathologies
from datasets.xray.xray_datasets import CheX_Dataset
from datasets.xray.xray_datasets import ConcatDataset
from datasets.xray.xray_datasets import MIMIC_Dataset
from datasets.xray.xray_datasets import PC_Dataset
from datasets.xray.xray_datasets import SubsetDataset
from datasets.xray.xray_datasets import ToPILImage
from datasets.xray.xray_datasets import VinBig_Dataset
from datasets.xray.xray_datasets import XRayCenterCrop
from datasets.xray.xray_datasets import XRayResizer
from datasets.xray.xray_datasets import relabel_dataset
from queryset import QuerySet


class DataTypes(Enum):
    """The type of data used."""
    train = 0
    dev = 1
    unlabeled = 2
    test = 3
    debug = 4
    small_test = 5


def get_data_augmentation(args, data_type):
    augs = []
    augs.append(ToPILImage())

    if data_type in [DataTypes.train, DataTypes.debug]:
        """Apply data augmentation only during training."""
        affine = torchvision.transforms.RandomAffine(
            args.data_aug_rot,
            translate=(args.data_aug_trans, args.data_aug_trans),
            scale=(1.0 - args.data_aug_scale, 1.0 + args.data_aug_scale))
        augs.append(affine)

    augs.append(torchvision.transforms.ToTensor())

    data_aug = torchvision.transforms.Compose(augs)
    return data_aug


def get_data(args, data_type, mix_transform=None):
    """
    Get data.
    :param args:
    :param data_type: type of data
    :return:
    """
    transforms = torchvision.transforms.Compose(
        [XRayCenterCrop(), XRayResizer(224)])

    data_aug = get_data_augmentation(args=args, data_type=data_type)

    if data_type == DataTypes.debug:
        csvpath = 'debug.csv'
        mode_type = 'valid'
    elif data_type == DataTypes.small_test:
        csvpath = 'valid.csv'
        mode_type = 'valid'
    else:
        csvpath = 'train.csv'
        mode_type = 'train'

    if args.dataset == 'cxpert':
        dataset = CheX_Dataset(
            imgpath=args.dataset_path,
            csvpath=args.dataset_path + csvpath,
            views=args.xray_views,
            mode_type=mode_type,
            transform=transforms,
            data_aug=data_aug,
            mix_transform=mix_transform,
            unique_patients=False)
        # ones_per_example(args=args, dataset=dataset)
    elif args.dataset == 'padchest':
        dataset = PC_Dataset(
            imgpath=args.dataset_path + "/PC/images-224", views=args.xray_views,
            transform=transforms, data_aug=data_aug, unique_patients=False)
    elif args.dataset == 'mimic':
        dataset = MIMIC_Dataset(
            imgpath=args.dataset_path,
            csvpath=args.dataset_path + "/train.csv",
            transform=transforms, data_aug=data_aug, views=args.xray_views,
            unique_patients=False)
    elif args.dataset == 'vin':
        dataset = VinBig_Dataset(
            imgpath=args.dataset_path + "/vin/train",
            csvpath=args.dataset_path + "/vin/train.csv",
            views=args.xray_views, transform=transforms, data_aug=data_aug)
    else:
        raise Exception(f"Unsupported dataset: {args.dataset}.")

    # Relabel labels to establish common set of 18 labels for x-rays.
    relabel_dataset(dataset=dataset, pathologies=default_pathologies)

    return dataset


def get_train_from_train_set(args) -> SubsetDataset:
    """

    Returns: return subset of the whole training set that is used for the
    training of private models.

    """
    dataset = get_data(args=args, data_type=DataTypes.train)
    start = 0
    end = args.num_train_samples
    return SubsetDataset(dataset, list(range(start, end)))


def get_dev_from_train_set(args):
    """

    Returns: return subset of the whole training set that is used for the
    tuning of private models. This is the validation set.

    """
    dataset = get_data(args=args, data_type=DataTypes.train)
    start = args.num_train_samples
    end = start + args.num_dev_samples
    return SubsetDataset(dataset, list(range(start, end)))


def get_unlabeled_from_train_set(args, mix_transform=None):
    """

    :param args: program parameters
    :mix_transform: additional data transformations for the MixMatch algorithm

    Returns: return subset of the whole training set that is used for the
    labeling.

    """
    dataset = get_data(args=args, data_type=DataTypes.train,
                       mix_transform=mix_transform)
    start = args.num_train_samples + args.num_dev_samples
    end = start + args.num_unlabeled_samples
    return SubsetDataset(dataset, list(range(start, end)))


def get_test_from_train_set(args):
    """

    Returns: return subset of the whole training set that is used for the
    testing.

    """
    dataset = get_data(args=args, data_type=DataTypes.train)
    start = args.num_train_samples + args.num_dev_samples + args.num_unlabeled_samples
    end = start + args.num_test_samples
    assert end == args.num_all_samples
    return SubsetDataset(dataset, list(range(start, end)))


def get_data_set(args, data_type: DataTypes, mix_transform=None):
    """

    Args:
        args: program arguments
        data_type: type of the data: train, test, valid
        mix_transform: transformation for the semi-supervised training with
            the MixMatch algorithm.

    Returns:
        a dataset

    """
    if data_type == DataTypes.train:
        return get_train_from_train_set(args=args)
    elif data_type == DataTypes.debug:
        return get_dev_from_train_set(args=args)
    elif data_type == DataTypes.unlabeled:
        return get_unlabeled_from_train_set(args=args,
                                            mix_transform=mix_transform)
    elif data_type == DataTypes.test:
        # return get_data(args=args, data_type=DataTypes.test,
        #                 mix_transform=mix_transform)
        return get_test_from_train_set(args=args)
    elif data_type == DataTypes.small_test:
        return get_data(args=args, data_type=DataTypes.small_test,
                        mix_transform=mix_transform)
    else:
        raise Exception(f"Unknown data type: {data_type}.")


def get_xray_train_data(args):
    return get_data_set(args=args, data_type=DataTypes.train)


def get_xray_dev_data(args):
    return get_data_set(args=args, data_type=DataTypes.dev)


def get_xray_unlabeled_data(args, mix_transform=None):
    return get_data_set(args=args, data_type=DataTypes.unlabeled,
                        mix_transform=mix_transform)


def get_xray_test_data(args, mixtransform=None):
    return get_data_set(args=args, data_type=DataTypes.test,
                        mix_transform=mixtransform)


def get_xray_small_test_data(args, mixtransform=None):
    return get_data_set(args=args, data_type=DataTypes.small_test,
                        mix_transform=mixtransform)


def get_xray_debug_data(args):
    dataset = get_data(args=args, data_type=DataTypes.debug)
    assert len(dataset) == 33
    return dataset


def get_dataloaders(args, dataset):
    """
    Medical data require special handling

    :param args: Program parameters.
    :param dataset: Dataset to be divided evenly among args.num_models.
    :return: data loaders for the non-overlapping datasets.
    """
    dataset_size = len(dataset) // args.num_models
    data_loaders = []
    for i in range(args.num_models):
        begin = i * dataset_size
        if i == args.num_models - 1:
            end = len(dataset)
        else:
            end = (i + 1) * dataset_size
        indices = list(range(begin, end))
        dataset_i = SubsetDataset(dataset, indices)
        kwargs = args.kwargs
        data_loader = DataLoader(
            dataset=dataset_i,
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs)
        data_loaders.append(data_loader)
    return data_loaders


def check_dataset(args):
    if not args.dataset in args.xray_datasets:
        raise Exception(f"Unsupported dataset: {args.dataset}")


def get_xray_private_dataloaders(args):
    check_dataset(args=args)
    private_dataset = get_xray_train_data(args=args)
    return get_dataloaders(args=args, dataset=private_dataset)


def get_xray_unlabeled_dataloaders(args, unlabeled_dataset=None):
    check_dataset(args=args)
    if unlabeled_dataset is None:
        unlabeled_dataset = get_xray_unlabeled_data(args=args)
    return get_dataloaders(args=args, dataset=unlabeled_dataset)


def get_xray_debug_dataloaders(args):
    check_dataset(args=args)
    debug_dataset = get_xray_debug_data(args=args)
    data_loader = DataLoader(
        dataset=debug_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **args.kwargs)
    # get_dataloaders(args=args, dataset=debug_dataset)
    return [data_loader]


def load_ordered_unlabeled_data(args, indices):
    """
    :param args: Program params.
    :param indices: which indices to extract
    :return: the data loader for the subset of the unlabeled data
    """
    dataset = get_xray_unlabeled_data(args=args)
    return load_ordered_unlabeled_xray(args=args, indices=indices,
                                       dataset=dataset)


def load_ordered_unlabeled_xray(args, indices, dataset):
    """
    :param args: Program params.
    :param indices: which indices to extract.
    :param dataset: the unlabeled dataset.
    :return: the data loader for the subset of the unlabeled data.
    """
    dataset_i = SubsetDataset(dataset, indices)
    kwargs = args.kwargs
    data_loader = DataLoader(
        dataset=dataset_i,
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs)
    return data_loader


# def xray_load_private_data_and_qap(args):
#     all_private_datasets = get_xray_train_data(args=args)
#     private_dataset_size = len(all_private_datasets) // args.num_models
#     all_augmented_dataloaders = []
#     # Get the data for re-training, thus with the training augmentation.
#     transform = get_data_augmentation(args, data_type=DataTypes.train)
#     target_transform = None
#     for i in args.querying_parties:
#         begin = i * private_dataset_size
#         if i == args.num_models - 1:
#             end = len(all_private_datasets)
#         else:
#             end = (i + 1) * private_dataset_size
#         indices = list(range(begin, end))
#         private_dataset = SubsetDataset(all_private_datasets, indices)
#
#         query_dataset = QuerySet(args=args, transform=transform, id=i,
#                                  target_transform=target_transform)
#         augmented_dataset = ConcatDataset([private_dataset])
#         augmented_dataset.append(query_dataset=query_dataset)
#         augmented_dataloader = DataLoader(augmented_dataset,
#                                           batch_size=args.batch_size,
#                                           shuffle=True, **args.kwargs)
#         all_augmented_dataloaders.append(augmented_dataloader)
#     return all_augmented_dataloaders

def xray_load_private_data_and_qap(args):
    # New function for current retraining (with only one model) and no use of the training data for the private model
    # all_private_datasets = get_xray_train_data(args=args)
    # private_dataset_size = len(all_private_datasets) // args.num_models
    all_augmented_dataloaders = []
    # Get the data for re-training, thus with the training augmentation.
    transform = get_data_augmentation(args, data_type=DataTypes.train)
    target_transform = None
    for i in args.querying_parties:
        query_dataset = QuerySet(args=args, transform=transform, id=i,
                                 target_transform=target_transform)
        augmented_dataloader = DataLoader(query_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=True, **args.kwargs)
        all_augmented_dataloaders.append(augmented_dataloader)
        print("num of retrain queries", len(query_dataset))
    return all_augmented_dataloaders


def ones_per_example(args, dataset=None):
    if dataset is None:
        all_set = get_data(args=args, data_type=DataTypes.train)
    ones = []
    for i, (_, target) in enumerate(dataset):
        # print('target: ', target)
        ones.append((target == 1).sum().item())
        if i % 5000 == 0:
            print(i, ',mean,', np.mean(ones), ',median,', np.median(ones),
                  ',std,', np.std(ones))
    print(',mean,', np.mean(ones), ',median,', np.median(ones), ',std,',
          np.std(ones))
