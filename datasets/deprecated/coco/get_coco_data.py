from torchvision import transforms
from torch.utils.data import ConcatDataset
import os
from datasets.deprecated.coco.helper_functions.helper_functions import \
    CocoDetection
from datasets.deprecated.coco.custom_labels import CocoDatasetCustom


def get_all_coco_data(args):
    # Gather all data sets into one set.
    coco_data_types = ['train', 'val']
    coco_sets = []
    for data_type in coco_data_types:
        coco_set = get_coco_dataset(args=args, data_type=data_type)
        coco_sets.append(coco_set)
    all_set = ConcatDataset(coco_sets)
    return all_set


def get_coco_dataset(args, data_type):
    normalize = transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])

    data_name = f"{data_type}{args.coco_version}"
    data_path = os.path.join(args.dataset_path, data_name)

    if args.coco_data_loader == 'standard':
        instances_path = os.path.join(
            args.dataset_path, f'annotations/instances_{data_name}.json')
        dataset = CocoDetection(
            root=data_path,
            annFile=instances_path,
            transform=transforms.Compose([
                transforms.Resize(
                    (args.coco_image_size, args.coco_image_size)),
                transforms.ToTensor(),
                normalize,
            ]))
    elif args.coco_data_loader == 'custom':
        file_dir = os.path.dirname(os.path.abspath(__file__))
        label_file = os.path.join(
            file_dir, f'./labels/{data_name}.pkl')
        dataset = CocoDatasetCustom(
            root=data_path,
            label_file=label_file,
            transform=transforms.Compose([
                transforms.Resize(
                    (args.coco_image_size, args.coco_image_size)),
                transforms.ToTensor(),
                normalize,
            ]))
    else:
        raise Exception(
            f"Unknown args.coco_data_loader: {args.coco_data_loader}")

    print("coco len(dataset)): ", len(dataset))
    return dataset
