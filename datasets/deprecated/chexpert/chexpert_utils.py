from torch.utils.data import Subset, DataLoader, ConcatDataset

from datasets.deprecated.chexpert.data.dataset import ImageDataset
from datasets.deprecated.chexpert.data.pos_disease import PosDiseaseDataset
from datasets.deprecated.chexpert.data.single_disease import \
    SingleDiseaseDataset


def get_chexpert_all_private_datasets(args):
    dataset_type = get_chexpert_dataset_type(args=args)
    dataset = dataset_type(
        in_csv_path=args.cfg.train_csv, cfg=args.cfg,
        mode='train')
    # Only a part of the original train set is used for training.
    all_private_datasets = Subset(
        dataset, list(range(0, args.num_train_samples)))
    private_dataset_size = len(all_private_datasets) // args.num_models
    return all_private_datasets, private_dataset_size


def get_chexpert_private_data(args, kwargs):
    if not args.dataset.startswith('chexpert'):
        return None
    all_private_datasets, private_dataset_size = get_chexpert_all_private_datasets(
        args=args)
    all_private_trainloaders = []
    for i in range(args.num_models):
        begin = i * private_dataset_size
        if i == args.num_models - 1:
            end = len(all_private_datasets)
        else:
            end = (i + 1) * private_dataset_size
        indices = list(range(begin, end))
        private_dataset = Subset(all_private_datasets, indices)
        private_trainloader = DataLoader(
            private_dataset,
            batch_size=args.batch_size,
            drop_last=True,
            shuffle=True,
            **kwargs)
        all_private_trainloaders.append(private_trainloader)
    return all_private_trainloaders


def get_chexpert_dataset_type(args):
    if args.chexpert_dataset_type == 'single':
        assert len(args.cfg.num_classes) == 1
        dataset_type = SingleDiseaseDataset
    elif args.chexpert_dataset_type == 'multi':
        assert len(args.cfg.num_classes) > 1
        dataset_type = ImageDataset
    elif args.chexpert_dataset_type == 'pos':
        dataset_type = PosDiseaseDataset
    else:
        raise Exception(
            f"Unknown args.chexpert_dataset_type: {args.chexpert_dataset_type}.")
    return dataset_type


def get_chexpert_train_set(args):
    train_set, _ = get_chexpert_all_private_datasets(args=args)
    return train_set


def get_chexpert_dev_set(args, mode='train'):
    dataset_type = get_chexpert_dataset_type(args=args)
    dataset = dataset_type(
        in_csv_path=args.cfg.train_csv,
        cfg=args.cfg,
        mode=mode)
    # Only a part of the original train set is used for training.
    indices = list(
        range(args.num_train_samples,
              args.num_train_samples + args.num_dev_samples))
    dev_set = Subset(dataset, indices)
    return dev_set


def get_chexpert_dev_loader(args, mode='train'):
    kwargs = args.kwargs
    dev_set = get_chexpert_dev_set(args=args, mode=mode)
    dev_loader = DataLoader(
        dev_set,
        batch_size=args.batch_size,
        shuffle=False,
        **kwargs)
    return dev_loader


def get_chexpert_test_set(args, mode='train'):
    dataset_type = get_chexpert_dataset_type(args=args)
    dataset = dataset_type(
        in_csv_path=args.cfg.train_csv,
        cfg=args.cfg,
        mode=mode)
    # Only a part of the original train set is used for training.
    indices = list(
        range(args.num_train_samples + args.num_dev_samples, len(dataset)))
    trainset_part = Subset(dataset, indices)
    validset = dataset_type(
        in_csv_path=args.cfg.dev_csv,
        cfg=args.cfg,
        mode=mode)
    test_set = ConcatDataset([trainset_part, validset])
    return test_set
