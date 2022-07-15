import os
import pickle

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms


class PseudoDataset(torch.utils.data.Dataset):
    # this class is taken from the dataset inference repo

    def __init__(self, x, y, transform=None):
        self.data = x
        self.targets = y
        self.transform = transform
        self.len = self.data.shape[0]

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return self.len


def get_transform():
    transform = transforms.Compose([
        transforms.Pad(4),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (
                0.49139969,
                0.48215842,
                0.44653093),
            (
                0.24703223,
                0.24348513,
                0.26158784)
        )])
    return transform


def get_extra_cifar10_data_from_ti():
    ti_filename = "ti_500K_pseudo_labeled.pickle"
    if os.path.exists(
            os.path.join("/home/nicolas/data/tinyimages/", ti_filename)):
        filepath = os.path.join("/home/nicolas/data/tinyimages/", ti_filename)
    else:
        filepath = os.path.join("D:/year4/capc/", ti_filename)
    with open(filepath, 'rb') as f:
        tinyimage = pickle.load(f)
    tinyimage_data = tinyimage['data']
    tinyimage_targets = tinyimage['extrapolated_targets']
    transform = get_transform()
    private_dataset = PseudoDataset(tinyimage_data, tinyimage_targets,
                                    transform=transform)
    # private_trainloader = DataLoader(private_dataset,
    #                                          batch_size=args.batch_size,
    #                                          shuffle=True, **args.kwargs)
    return private_dataset


if __name__ == "__main__":
    a = get_extra_cifar10_data_from_ti()
