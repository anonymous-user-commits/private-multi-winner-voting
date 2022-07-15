import os
from torchvision import datasets
from torchvision import transforms

from datasets.imagenet.imagenet_folder import ImageNet


def get_imagenet_dataset(args, train=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    preprocessing = [
        transforms.ToTensor(),
        normalize,
    ]

    # transfrom taken from https://github.com/pytorch/vision/issues/39
    # preprocessing used for all pretained models for torchvision
    if 'mnist' in args.dataset:
        # this means we are attacking a mnist model with imagenet, need to resize, crop and grayscale
        preprocessing.append(transforms.Resize(28))
        preprocessing.append(transforms.CenterCrop(28))
        preprocessing.append(transforms.Grayscale())

    elif 'imagenet' not in args.dataset:
        # this means we are stealing other model with imagenet data
        # need preprocessing steps
        raise Exception(
            "unimplemented model extraction attack with imagenet data, you might need to add customized preprocessing steps here")
    else:
        # just using imagenet data
        pass
    if os.path.exists("/home/nicolas/data/imagenet/"):
        imagenet_dataset = datasets.ImageNet(
            root="/home/nicolas/data/imagenet/",
            train=train,
            transform=transforms.Compose(preprocessing),
        )
    elif os.path.exists("/data/imagenet/"):
        imagenet_dataset = ImageNet(
            root="/data/imagenet/",
            train=train,
            transform=transforms.Compose(preprocessing),
        )
    else:
        print("Cannot find imagenet data")
    return imagenet_dataset
