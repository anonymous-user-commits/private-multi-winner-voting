import os

import torch
from torch.utils.data import Subset, DataLoader
from torchvision import transforms, datasets


def get_retinopathy_private_data(args):
    all_private_datasets, private_dataset_size = get_retinopathy_train_data(
        args, mode='train')
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
            shuffle=True,
            **args.kwargs)
        all_private_trainloaders.append(private_trainloader)
    return all_private_trainloaders


class KrizhevskyColorAugmentation(object):
    def __init__(self, sigma=0.5):
        self.sigma = sigma
        self.mean = torch.tensor([0.0])
        self.deviation = torch.tensor([sigma])

    def __call__(self, img, color_vec=None):
        sigma = self.sigma
        if color_vec is None:
            if not sigma > 0.0:
                color_vec = torch.zeros(3, dtype=torch.float32)
            else:
                color_vec = torch.distributions.Normal(self.mean,
                                                       self.deviation).sample(
                    (3,))
            color_vec = color_vec.squeeze()

        U = torch.tensor([[-0.56543481, 0.71983482, 0.40240142],
                          [-0.5989477, -0.02304967, -0.80036049],
                          [-0.56694071, -0.6935729, 0.44423429]],
                         dtype=torch.float32)
        EV = torch.tensor([1.65513492, 0.48450358, 0.1565086],
                          dtype=torch.float32)
        alpha = color_vec * EV
        noise = torch.matmul(U, alpha.t())
        noise = noise.view((3, 1, 1))
        return img + noise

    def __repr__(self):
        return self.__class__.__name__ + '(sigma={})'.format(self.sigma)


def get_retinopathy_transform(args):
    if args.architecture == 'RetinoNet':
        MEAN = [108.64628601 / 255, 75.86886597 / 255, 54.34005737 / 255]
        STD = [70.53946096 / 255, 51.71475228 / 255, 43.03428563 / 255]
        input_size = 112
        data_aug = {'scale': (1 / 1.15, 1.15),
                    'stretch_ratio': (0.7561, 1.3225),
                    # (1/(1.15*1.15) and 1.15*1.15)
                    'ratation': (-180, 180),
                    'translation_ratio': (40 / 112, 40 / 112),
                    # 40 pixel in the report
                    'sigma': 0.5}
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                size=input_size,
                scale=data_aug['scale'],
                ratio=data_aug['stretch_ratio']
            ),
            transforms.RandomAffine(
                degrees=data_aug['ratation'],
                translate=data_aug['translation_ratio'],
                scale=None,
                shear=None
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(tuple(MEAN), tuple(STD)),
            KrizhevskyColorAugmentation(sigma=data_aug['sigma'])
        ])
    elif args.architecture == 'SimpleRetinoNet':
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()])
    else:
        raise Exception(
            f'Unknown architecture for Retinopathy: {args.architecture}.')
    return transform


def get_retinopathy_train_data(args, mode):
    TRAIN_PATH = os.path.join(args.dataset_path, 'train/')
    transform = get_retinopathy_transform(args=args)
    dataset = datasets.ImageFolder(TRAIN_PATH, transform=transform)

    if mode == 'train':
        indices = list(range(args.num_train_samples))
        private_dataset_size = len(indices) // args.num_models
    elif mode == 'test':
        indices = list(range(args.num_train_samples, len(dataset)))
        private_dataset_size = None
    else:
        raise ValueError
    subset = Subset(dataset, indices)
    return subset, private_dataset_size