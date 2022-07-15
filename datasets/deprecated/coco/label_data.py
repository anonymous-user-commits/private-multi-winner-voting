# Adopted from: https://github.com/allenai/elastic/blob/master/multilabel_classify.py
# special thanks to @hellbell
from getpass import getuser

import argparse
import os
import time
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.datasets as datasets
from PIL import Image
from torchvision import transforms

from datasets.deprecated.coco.models import create_model
from general_utils.save_load import save_obj

user = getuser()

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset_path', metavar='DIR',
                    default=f'/home/{user}/data/coco/',
                    help='Path to the dataset.')
parser.add_argument(
    '--model-name',
    default='tresnet_l',
    # default='tresnet_xl',
)
parser.add_argument(
    '--model-path',
    default=f'/home/{user}/models/multi_label/coco/MS_COCO_TRresNet_L_448_86.6.pth',
    # default=f'/home/{user}/models/multi_label/coco/MS_COCO_TResNet_xl_640_88.4.pth',
    # default=f'/home/{user}/models/multi_label/coco/Open_ImagesV6_TRresNet_L_448_86_3.pth',
    # default=f'/home/{user}/models/multi_label/coco/',
    type=str)
parser.add_argument('--num_classes', default=80)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('--coco_image_size', default=448, type=int,
                    metavar='N', help='input image size (default: 448)')
parser.add_argument('--coco_threshold', default=0.8, type=float,
                    metavar='N', help='threshold value')
parser.add_argument('-b', '--batch_size', default=32, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--print-freq', '-p', default=64, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('--coco_version', default='2017', type=str,
                    help='the year of the dataset')
parser.add_argument('--debug', type=bool, help="Debug mode of execution",
                    # default=True,
                    default=False,
                    )
parser.add_argument('--data_type', type=str, help="The type of the data.",
                    # default='smallval',
                    # default='test',
                    default='unlabeled',
                    )
parser.add_argument('--coco_data_loader', type=str,
                    help='standard or custom data loader, where custom uses'
                         'the pre-generated labels',
                    default='custom',
                    )


# from src.helper_functions.helper_functions import coco_classes_list

class CocoDatasetUnlabeled(datasets.coco.CocoDetection):
    def __init__(self, root, transform=None):
        self.root = root
        self.file_names = []
        for file in os.listdir(root):
            if file.endswith('.jpg'):
                self.file_names.append(file)
        print('len of file_names: ', len(self.file_names))
        self.ids = list(range(len(self.file_names)))
        self.transform = transform

    def __getitem__(self, index):
        img_path = self.file_names[index]
        img = Image.open(os.path.join(self.root, img_path)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img_path, img


def get_dataset(args):
    data_path = os.path.join(args.dataset_path, args.data_name)
    normalize = transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
    dataset = CocoDatasetUnlabeled(
        root=data_path,
        transform=transforms.Compose([
            transforms.Resize(
                (args.coco_image_size, args.coco_image_size)),
            transforms.ToTensor(),
            normalize,
        ]))
    return dataset


def label_multi(data_loader, model, args):
    print("Start labeling.")
    start_time = time.time()
    Sig = torch.nn.Sigmoid()

    labels = []
    for img_paths, images in data_loader:
        # target = target.max(dim=1)[0] # this is on the level of whole batch
        # compute output
        with torch.no_grad():
            output = Sig(model(images.cuda())).cpu()

        # measure accuracy and record loss
        preds = output.data.gt(args.coco_threshold).long()
        preds = list(preds.cpu().detach().numpy())
        for img_path, target in zip(img_paths,preds):
            labels.append((img_path, target))

    save_obj(file=f'./labels/{args.data_name}.pkl', obj=labels)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('elapsed time (sec): ', elapsed_time)


def main():
    args = parser.parse_args()
    args.cwd = os.getcwd()
    # setup model
    print('creating and loading the model...')
    state = torch.load(args.model_path, map_location='cpu')
    args.num_classes = state['num_classes']
    args.do_bottleneck_head = False
    model = create_model(args).cuda()
    model.load_state_dict(state['model'], strict=True)
    model.eval()

    args.data_name = args.data_type + args.coco_version

    unlabeled_dataset = get_dataset(args=args)

    data_loader = torch.utils.data.DataLoader(
        unlabeled_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    label_multi(data_loader=data_loader, model=model, args=args)


if __name__ == '__main__':
    main()
