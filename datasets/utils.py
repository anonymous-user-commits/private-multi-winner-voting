import getpass
import json
import numpy as np
import warnings

import os

from utils import get_cfg, class_ratio, augmented_print


def set_dataset(args):
    # Dataset
    args.weak_classes = set_weak_classes(weak_classes=args.weak_classes)
    args.dataset = args.dataset.lower()
    args.datasets = ['mnist', 'fashion-mnist', 'svhn', 'cifar10', 'cifar100',
                     'chexpert', 'retinopathy', 'celeba',
                     'coco'] + args.xray_datasets
    args.datasets_string = ",".join(args.datasets)
    args.datasets_exception = \
        f'Dataset name must be in: {args.datasets_string}. Check if the ' \
        f'option is supported for your dataset.'
    user = getpass.getuser()
    if args.dataset == 'mnist':
        args.dataset_path = os.path.join(args.data_dir, 'MNIST')
        args.num_unlabeled_samples = 9000
        args.num_dev_samples = 0
        args.num_classes = 10
        # Hyper-parameter delta in (eps, delta)-DP.
        args.delta = 1e-5
        args.num_teachers_private_knn = 300
        # args.sigma_gnmax_private_knn = 28
    elif args.dataset == 'fashion-mnist':
        args.dataset_path = os.path.join(args.data_dir, 'Fashion-MNIST')
        args.num_unlabeled_samples = 9000
        args.num_classes = 10
        args.delta = 1e-5
        args.num_teachers_private_knn = 300
        # args.sigma_gnmax_private_knn = 28
    elif args.dataset == 'svhn':
        args.dataset_path = os.path.join(args.data_dir, 'SVHN')
        args.num_unlabeled_samples = 10000
        args.num_classes = 10
        args.delta = 1e-6
        args.num_teachers_private_knn = 800
        # args.sigma_gnmax_private_knn = 100
    elif args.dataset == 'pascal':
        args.dataset_path = os.path.join(args.data_dir, 'VOC2012')
        # args.num_unlabeled_samples = 4000
        # args.num_unlabeled_samples = 5000
        args.num_unlabeled_samples = 5823
        args.num_classes = 20
        args.delta = 1e-5
    elif args.dataset == 'cifar10':
        args.dataset_path = os.path.join(args.data_dir, 'CIFAR10')
        args.num_unlabeled_samples = 9000
        args.num_classes = 20
        args.delta = 1e-5
    elif args.dataset == 'cifar10':
        args.dataset_path = os.path.join(args.data_dir, 'CIFAR10')
        args.num_unlabeled_samples = 9000
        args.num_classes = 10
        args.delta = 1e-5
        args.num_teachers_private_knn = 300
        # args.sigma_gnmax_private_knn = 28
    elif args.dataset == 'cifar100':
        args.dataset_path = os.path.join(args.data_dir, 'CIFAR100')
        args.num_unlabeled_samples = 9000
        args.num_classes = 100
        args.delta = 1e-5
    elif args.dataset.startswith('chexpert'):
        args.dataset_path = os.path.join(
            args.data_dir, 'CheXpert-v1.0-small/')
        args.num_train_samples = 207000
        args.num_dev_samples = 3000
        args.num_unlabeled_samples = 9000
        args.num_classes = 2
        args.delta = 1e-6
        cfg_path = f'datasets/chexpert/config/capc_{args.class_type}_disease_small.json'
        args.cfg = get_cfg(cfg_path=cfg_path)
        if args.verbose is True:
            print(json.dumps(args.cfg, indent=4))
    elif args.dataset == 'retinopathy':
        args.dataset_path = os.path.join(f'/home/{user}/data/', 'diabetic/')
        args.num_train_samples = 31126  # 35126 - 3000 - 1000
        args.num_unlabeled_samples = 3000  # 1000 for each of querying parties
        args.num_classes = 5
        args.delta = 1e-5
    elif args.dataset == 'celeba':
        args.taskweights = False
        args.adam_amsgrad = False
        args.dataset_path = os.path.join(args.data_dir, 'celeba')
        args.num_all_samples = 202599
        args.num_dev_samples = 3000
        args.num_unlabeled_samples = 9000
        args.num_test_samples = 3000
        args.num_train_samples = args.num_all_samples - args.num_dev_samples - args.num_unlabeled_samples - args.num_test_samples
        num_samples_per_model = args.num_train_samples / args.num_models
        print('num_train_samples: ', args.num_train_samples)
        print('num_models: ', args.num_models)
        print('num_samples_per_model: ', num_samples_per_model)
        args.num_classes = 40
        args.delta = 1e-6
    elif args.dataset == 'coco':
        args.dataset_path = os.path.join(f'/home/{user}/data/',
                                         'deprecated/coco')

        # Number of samples in the coco 2017 original train set.
        args.coco_num_train_samples = 117266
        # Number of samples in the coco validation set.
        args.coco_num_test_samples = 4952
        args.num_all_samples = args.coco_num_train_samples + args.coco_num_test_samples
        args.num_dev_samples = 3000
        args.num_unlabeled_samples = 9000
        args.num_test_samples = args.coco_num_test_samples
        args.num_train_samples = args.num_all_samples - args.num_dev_samples - args.num_unlabeled_samples - args.num_test_samples
        args.num_classes = 80  # 80 object categories
        args.delta = 1e-6
    elif args.dataset in args.xray_datasets:
        args.optimizer = 'Adam'
        args.taskweights = True
        args.adam_amsgrad = True
        # args.delta = 1e-6 #1e-4
        args.delta = 1e-4
        args.num_classes = 18
        args.num_dev_samples = 0
        # unlabeled + test samples
        args.num_unlabeled_samples = 3000
        args.num_test_samples = 1000
        if args.dataset == 'padchest':
            args.num_unlabeled_samples = 3000
            args.num_test_samples = 1000

        if args.dataset == 'cxpert':
            args.dataset_path = os.path.join(args.data_dir,
                                             'CheXpert-v1.0-small/')
            if args.xray_views == ['PA']:
                args.num_all_samples = 29420
            elif args.xray_views == ['AP']:
                args.num_all_samples = 161590
            elif args.xray_views == ['AP', 'PA']:
                args.num_all_samples = 191010
            elif args.xray_views == ['lateral']:
                args.num_all_samples = 32404
            else:
                # total number of samples without any filtering
                args.num_all_samples = 223414
        elif args.dataset == 'padchest':
            args.dataset_path = args.data_dir
            if args.xray_views == ['PA']:
                args.num_all_samples = 91658
            elif args.xray_views == ['AP']:
                args.num_all_samples = 4554
            elif args.xray_views == ['AP', 'PA']:
                args.num_all_samples = 96212
            elif args.xray_views == ['lateral']:
                args.num_all_samples = 0
            else:
                # total number of samples without any filtering
                args.num_all_samples = 96212
        elif args.dataset == 'vin':
            args.dataset_path = args.data_dir
            # total number of samples without any filtering
            args.num_all_samples = 15000
        elif args.dataset == 'mimic':
            args.dataset_path = args.data_dir
            if args.xray_views == ['PA']:
                warnings.warn(
                    'Only PA views not supported. Using all frontal images.')
                args.xray_views = ['frontal']
                args.num_all_samples = 248236
            elif args.xray_views == ['AP']:
                warnings.warn(
                    'Only AP views not supported. Using all frontal images.')
                args.xray_views = ['frontal']
                args.num_all_samples = 248236
            elif args.xray_views == ['AP', 'PA'] or args.xray_views == [
                'frontal']:
                args.xray_views = ['frontal']
                args.num_all_samples = 248236
            elif args.xray_views == ['lateral']:
                args.num_all_samples = 120743
            else:
                # total number of samples without any filtering
                args.num_all_samples = 369126
        else:
            raise Exception(f"Unsupported dataset: {args.dataset}.")

        args.num_train_samples = args.num_all_samples - args.num_dev_samples - args.num_unlabeled_samples - args.num_test_samples


    else:
        raise Exception(
            f"For dataset: {args.dataset}. " + args.datasets_exception)


def set_weak_classes(weak_classes):
    """
    Set weak classes.

    :param weak_classes: string with weak classes
    :return: array with int weak classes
    >>> weak_classes = '1,2'
    >>> weak_classes_normal = set_weak_classes(weak_classes=weak_classes)
    >>> assert len(weak_classes_normal) == 2
    >>> assert weak_classes_normal[0] == 1
    >>> assert weak_classes_normal[1] == 2
    >>> weak_classes_empty = ''
    >>> weak_classes_empty_array = set_weak_classes(weak_classes_empty)
    >>> assert len(weak_classes_empty_array) == 0
    >>> assert weak_classes_empty_array == []
    """
    if weak_classes is None or weak_classes == '':
        weak_classes = []
    else:
        assert type(weak_classes) == str
        weak_classes = [int(c) for c in str(weak_classes).split(',')]
    return weak_classes


def get_dataset_full_name(args):
    dataset = args.dataset

    if args.dataset.startswith('chexpert'):
        dataset += '-' + args.class_type

    if args.dataset_type == 'imbalanced':
        dataset += '-' + args.dataset_type
    elif args.dataset_type == 'balanced':
        if args.balance_type == 'perfect':
            dataset += '-' + args.dataset_type + '-' + args.balance_type
        elif args.balance_type == 'standard':
            pass
        else:
            raise Exception(
                f'Unknown balance type: {args.balance_type}.')
    else:
        raise Exception(f'Unknown dataset type: {args.dataset_type}.')
    return dataset


def show_dataset_stats(dataset, file, args, dataset_name=''):
    """
    Show statistics about this dataset.

    :param dataset: the loader for the dataset
    :param file: where to write the log
    :param args: arguments
    :param dataset_name: is it test or train
    :return: nothing
    """
    counts, ratios = class_ratio(dataset, args)
    label_counts = np.array2string(counts, separator=', ')
    augmented_print(
        f"Label counts for {dataset_name} set: {label_counts}.",
        file)
    ratios = np.array2string(ratios, precision=2, separator=', ')
    augmented_print(f"Class ratios for {dataset_name} set: {ratios}.", file)
    augmented_print(
        f"Number of {dataset_name} samples: {len(dataset)}.", file)


if __name__ == "__main__":
    class Args:
        dataset = 'celeba'
        weak_classes = None


    args = Args()
    set_dataset(args=args)
    num_train = args.num_train_samples
    num_models = 50
    num_samples_per_model = num_train / num_models
    print('num samples per model: ', num_samples_per_model)
