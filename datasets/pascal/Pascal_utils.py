from torch.utils.data import DataLoader
import torch
import torchvision.transforms as transforms
from datasets.pascal.PascalLoader import DataLoader as PascalDataLoader


def get_dataloaders(args):
    data_loaders = []
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    for i in range(args.num_models):
        train_data = PascalDataLoader(args.data_dir, 'small_train_{}'.format(i),
                                      transform=train_transform)
        train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                                   batch_size=args.batch_size,
                                                   drop_last=True,
                                                   shuffle=True, )
        data_loaders.append(train_loader)
    return data_loaders
