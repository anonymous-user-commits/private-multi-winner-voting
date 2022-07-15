import torch
from pprint import pprint
import numpy as np


def pred_acc(original, predicted):
    return torch.round(predicted).eq(original).sum().numpy() / len(original)


def fit_model(epochs, model, dataloader, phase='training',
              criterion=torch.nn.functional.binary_cross_entropy,
              optimizer=torch.optim.SGD):
    pprint("Epoch: {}".format(epochs))

    if phase == 'training':
        model.train_features_epoch()

    if phase == 'validataion':
        model.eval()

    running_loss = []
    running_acc = []
    b = 0
    for i, data in enumerate(dataloader):

        inputs, target = data['image'].cuda(), data['label'].float().cuda()

        if phase == 'training':
            optimizer.zero_grad()

        ops = model(inputs)

        acc_ = []
        for i, d in enumerate(ops, 0):
            acc = pred_acc(torch.Tensor.cpu(target[i]), torch.Tensor.cpu(d))
            acc_.append(acc)

        loss = criterion(ops, target)

        running_loss.append(loss.item())
        running_acc.append(np.asarray(acc_).mean())
        b += 1

        if phase == 'training':
            loss.backward()

            optimizer.step()

    total_batch_loss = np.asarray(running_loss).mean()
    total_batch_acc = np.asarray(running_acc).mean()

    pprint("{} loss is {} ".format(phase, total_batch_loss))
    pprint("{} accuracy is {} ".format(phase, total_batch_acc))

    return total_batch_loss, total_batch_acc


def main():
    print('train celeba')


if __name__ == '__main__':
    main()
