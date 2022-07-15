import torch

torch.manual_seed(0)
from torchvision import transforms
import numpy as np

np.random.seed(0)
from opacus import PrivacyEngine
from pascal_loader import DataLoader
from pascal_network import *
from torch.autograd import Variable
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score
import warnings
from architectures.densenet_pre import densenetpre
from parameters import get_parameters
from utils import get_non_train_set
from utils import get_train_set
from architectures.celeba_net import CelebaNet

warnings.filterwarnings("ignore")


def compute_ACC(labels, outputs):
    y_prob = torch.sigmoid(outputs)
    y_pred = (y_prob > 0.5)
    y_true = labels.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    acc = []
    for i in range(y_true.shape[0]):
        acc.append(accuracy_score(
            y_true=y_true[i],
            y_pred=y_pred[i],
        ))
    return np.mean(acc)


def compute_mAP(labels, outputs):
    y_true = labels.cpu().numpy()
    y_pred = outputs.cpu().numpy()
    AP = []
    for i in range(y_true.shape[0]):
        AP.append(average_precision_score(y_true[i], y_pred[i]))
    return np.mean(AP)


def compute_AUC(labels, outputs):
    y_true = labels.cpu().numpy()
    y_pred = outputs.cpu().numpy()
    AUC = []
    for i in range(y_true.shape[0]):
        AUC.append(roc_auc_score(
            y_true=y_true[i],
            y_score=y_pred[i],
            average='weighted'
        ))
    return np.mean(AUC)


def compute_BA(labels, outputs):
    y_prob = torch.sigmoid(outputs)
    y_pred = (y_prob > 0.5)
    y_true = labels.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    BA = []

    for i in range(y_true.shape[0]):
        BA.append(balanced_accuracy_score(
            y_true=y_true[i],
            y_pred=y_pred[i],
        ))

    return np.mean(BA)


def train(model, train_loader, optimizer, epoch, DPSGD, device, delta):
    model.train()
    criterion = nn.MultiLabelSoftMarginLoss(weight=weight)
    losses = []
    mAP = []

    for _batch_idx, (images, labels) in enumerate(train_loader):
        images = Variable(images)
        labels = Variable(labels)
        images = images.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        mAP.append(compute_mAP(labels.data, outputs.data))

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss = loss.cpu().data.numpy()
        losses.append(loss)

    if DPSGD:
        epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(delta)
    else:
        epsilon, best_alpha = 0, 0

    print(
        f"\tTrain Epoch: {epoch} \t"
        f"Loss: {np.mean(losses):.6f} "
        f"(ε = {epsilon:.2f}, δ = {delta}) for α = {best_alpha} "
    )


def test(net, epoch, test_loader):
    mAP = []
    AUC = []
    BA = []
    ACC = []
    net.eval()
    for i, (images, labels) in enumerate(test_loader):
        images = images.view((-1, 3, 224, 224))
        images = Variable(images, volatile=True)

        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)
        outputs = outputs.cpu().data

        mAP.append(compute_mAP(labels, outputs))
        AUC.append(compute_AUC(labels, outputs))
        BA.append(compute_BA(labels, outputs))
        ACC.append(compute_ACC(labels, outputs))
    print(
        "Testing: Epoch[{:0>3}] mAP:{:.2%} AUC:{:.2%} BA:{:.2%} ACC:{:.2%}".format(
            epoch, np.mean(mAP), np.mean(AUC), np.mean(BA), np.mean(ACC)))
    net.train()
    return np.mean(mAP), np.mean(AUC), np.mean(BA), np.mean(ACC)


if __name__ == '__main__':
    # Get arguments
    args = get_parameters()

    # setting the parameters
    DPSGD = args.DPSGD
    EPOCHS = args.DPSGD_EPOCHS
    BATCH_SIZE = args.DPSGD_BATCH_SIZE
    NOISE_MULTIPLIER = args.DPSGD_NOISE_MULTIPLIER
    device = torch.device("cuda")  # if torch.cuda.is_available() else "cpu")
    C_clip = args.DPSGD_CCLIP
    lr = args.DPSGD_LR
    pascal_path = args.DPSGD_PASCAL_PATH
    dataset = args.dataset

    # loading data
    if dataset == 'pascal':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_transform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        # DataLoader initialize
        train_data = DataLoader(args.pascal_path, 'train',
                                transform=train_transform)
        test_data = DataLoader(args.pascal_path, 'val', transform=val_transform)
    else:
        train_data = get_train_set(args=args)
        test_data = get_non_train_set(args=args)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_data, batch_size=1, shuffle=False)

    # Creating a model
    if dataset == 'pascal':
        net = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=20)
        net_state_dict = net.state_dict()
        pretrained_dict34 = torch.load(
            "resnet50-19c8e357.pth")
        pretrained_dict_1 = {k: v for k, v in pretrained_dict34.items() if
                             k in net_state_dict}
        net_state_dict.update(pretrained_dict_1)
        net.load_state_dict(net_state_dict)
    elif dataset in args.xray_datasets:
        net = densenetpre()
    elif dataset == 'celeba':
        net = CelebaNet()
    else:
        print('The model for this dataset is not available.')

    net.cuda()

    from opacus.dp_model_inspector import DPModelInspector
    # inspector = DPModelInspector()
    # inspector.validate(net)

    from opacus.utils import module_modification

    net = module_modification.convert_batchnorm_modules(net)
    net.cuda()
    inspector = DPModelInspector()
    print(f"Is the model valid? {inspector.validate(net)}")
    weight = torch.tensor(
        [0.87713311, 1.05761317, 0.73638968, 1.11496746, 0.78593272, 1.33506494,
         0.4732965, 0.514, 0.47548566, 1.9469697, 0.97348485, 0.43670348,
         1.15765766, 1.06639004, 0.13186249, 1.05544148, 1.71906355, 1.04684318,
         1.028, 0.93624772])
    weight = weight.cuda()

    optimizer = torch.optim.SGD(net.parameters(),
                                lr=lr, momentum=0.9, weight_decay=0.0001)

    if DPSGD:
        # Attaching a Differential Privacy Engine to the Optimizer
        privacy_engine = PrivacyEngine(net, batch_size=BATCH_SIZE,
                                       sample_size=5717, alphas=range(2, 32),
                                       noise_multiplier=NOISE_MULTIPLIER,
                                       max_grad_norm=C_clip, )
        privacy_engine.attach(optimizer)

    for epoch in range(EPOCHS):
        train(net, train_loader, optimizer, epoch, DPSGD, device=device,
              delta=1e-5)
        test(net, epoch, test_loader)
