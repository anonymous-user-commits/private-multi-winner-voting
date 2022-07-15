import torch
from torchvision import transforms
from datasets.deprecated.nih.pytorch_multi_class.nih_dataset import NihDataset


def get_data_loaders(train_df, valid_df, test_df,
                     train_batch_size=12,
                     valid_batch_size=12,
                     test_batch_size=100,
                     num_workers=8):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    crop = 224
    train_transform = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(crop, scale=(0.63, 1)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean,
            std=std)])

    valid_transform = transforms.Compose([
        transforms.Resize(230),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean,
            std=std)])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(crop),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean,
            std=std)]
    )
    dsetTrain = NihDataset(train_df, train_transform)
    dsetVal = NihDataset(valid_df, valid_transform)
    dsetTest = NihDataset(test_df, test_transform)

    trainloader = torch.utils.data.DataLoader(
        dataset=dsetTrain,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers)
    valloader = torch.utils.data.DataLoader(
        dataset=dsetVal,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(
        dataset=dsetTest,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers)

    return trainloader, valloader, testloader
