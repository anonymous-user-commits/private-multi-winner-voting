from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from .get_coco_data import get_all_coco_data
from .get_coco_data import get_coco_dataset


def get_coco_set(args, type):
    all_set = get_all_coco_data(args=args)

    if type == 'train':
        # Only a part of the dataset set is used for training.
        start_index = 0
        last_index = args.num_train_samples
    elif type == 'dev':
        start_index = args.num_train_samples
        last_index = start_index + args.num_dev_samples
    elif type == 'test':
        # The test data is used for retraining (the 1st part), and the
        # remaining part is used for the "proper" testing.
        start_index = args.num_train_samples + args.num_dev_samples
        last_index = start_index + args.num_unlabeled_samples + args.num_test_samples
        if last_index != args.num_all_samples:
            raise Exception(
                f"The indexes do not sum up to the total data size.")
    else:
        raise Exception(f"Unknown data set type for coco: {type}.")
    return Subset(all_set, list(range(start_index, last_index)))


def get_coco_train_set(args):
    return get_coco_set(args=args, type='train')


def get_coco_dev_set(args):
    return get_coco_set(args=args, type='dev')


def get_coco_test_set(args):
    return get_coco_set(args=args, type='test')


def get_coco_private_data(args):
    if args.dataset != 'coco':
        return None

    train_set = get_coco_train_set(args=args)
    train_set_size = len(train_set) // args.num_models

    is_additional = len(args.coco_additional_datasets) > 0
    if is_additional is True:
        additional_sets = []
        coco_data_types = args.coco_additional_datasets
        for data_type in coco_data_types:
            coco_set = get_coco_dataset(args=args, data_type=data_type)
            additional_sets.append(coco_set)
        additional_set = ConcatDataset(additional_sets)
        additional_set_size = len(additional_set) // args.num_models
    else:
        additional_set = None
        additional_set_size = 0

    all_private_trainloaders = []
    for i in range(args.num_models):

        # train set
        begin = i * train_set_size
        if i == args.num_models - 1:
            end = len(train_set)
        else:
            end = (i + 1) * train_set_size
        indices = list(range(begin, end))
        dataset = Subset(train_set, indices)

        if is_additional is True:
            begin = i * additional_set_size
            if i == args.num_models - 1:
                end = len(additional_set)
            else:
                end = (i + 1) * additional_set_size
            indices = list(range(begin, end))
            additional_dataset = Subset(additional_set, indices)
            dataset = ConcatDataset([dataset, additional_dataset])

        kwargs = args.kwargs
        private_trainloader = DataLoader(
            dataset=dataset,
            batch_size=args.batch_size,
            shuffle=True,
            **kwargs)
        all_private_trainloaders.append(private_trainloader)

    return all_private_trainloaders
