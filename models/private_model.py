from architectures.celeba_net import CelebaNet
from architectures.fashion_mnist import FashionMnistNet
from architectures.mnist_net import MnistNet
# from architectures.mnist_net_pate import MnistNetPate
from architectures.mnist_net_pate_timing import MnistNetPate
from architectures.resnet import ResNet10, ResNet12, ResNet14, ResNet16, \
    ResNet18
from architectures.retino_net import RetinoNet, SimpleRetinoNet
from architectures.small_resnet import ResNet8
from architectures.tiny_resnet import ResNet6
from architectures.vggs import VGG
from datasets.deprecated.chexpert.model.classifier import SingleClassClassifier
from datasets.deprecated.coco.models.tresnet import tresnet_m
from datasets.deprecated.coco.models.tresnet import tresnet_l
from datasets.deprecated.coco.models.tresnet import tresnet_xl
from architectures.densenet import densenet121_capc
from architectures.densenet_xray import get_densenet121_xray
from architectures.densenet_pre import densenetpre
from architectures.resnet_pre import resnetpre
from models.utils_models import get_model_type_by_id, get_model_name_by_id
from architectures.PascalNetwork import *


def get_private_model(name, model_type, args):
    """Private model held by each party."""
    if args.use_pretrained_models == True:
        if model_type == 'densenet121_cxpert':
            model = densenetpre()
            model.dataset = 'cxpert'
        elif model_type == 'resnet50':
            model = resnetpre()
            model.dataset == 'pascal'
    elif model_type.startswith('VGG'):
        model = VGG(name=name, args=args, model_type=model_type)
    elif model_type == 'ResNet6':
        model = ResNet6(name=name, args=args)
    elif model_type == 'ResNet8':
        model = ResNet8(name=name, args=args)
    elif model_type == 'ResNet10':
        model = ResNet10(name=name, args=args)
    elif model_type == 'ResNet12':
        model = ResNet12(name=name, args=args)
    elif model_type == 'ResNet14':
        model = ResNet14(name=name, args=args)
    elif model_type == 'ResNet16':
        model = ResNet16(name=name, args=args)
    elif model_type == 'ResNet18':
        model = ResNet18(name=name, args=args)
    elif model_type == 'resnet50':
        return ResNet(Bottleneck, num_classes=args.num_classes)
    elif model_type.startswith('chexpert'):
        model = SingleClassClassifier(name=name, args=args)
    elif model_type == 'MnistNet':
        model = MnistNet(name=name, args=args)
    elif model_type == 'MnistNetPate':
        model = MnistNetPate(name=name, args=args)
    elif model_type == 'FashionMnistNet':
        model = FashionMnistNet(name=name, args=args)
    elif model_type == 'RetinoNet':
        model = RetinoNet(name=name, args=args)
    elif model_type == 'SimpleRetinoNet':
        model = SimpleRetinoNet(name=name, args=args)
    elif model_type == 'CelebaNet':
        model = CelebaNet(name=name, args=args)
    elif model_type == 'tresnet_m':
        model = tresnet_m(name=name, args=args)
    elif model_type == 'tresnet_l':
        model = tresnet_l(name=name, args=args)
    elif model_type == 'tresnet_xl':
        model = tresnet_xl(name=name, args=args)
    elif model_type == 'densenet121_capc':
        model = densenet121_capc(name=name, args=args)
    elif model_type == 'densenet121_mimic':
        model = get_densenet121_xray(name=name, args=args)
        model.dataset = 'mimic'
    elif model_type == 'densenet121_cxpert':
        model = get_densenet121_xray(name=name, args=args)
        model.dataset = 'cxpert'
    elif model_type == 'densenet121_padchest':
        model = get_densenet121_xray(name=name, args=args)
        model.dataset = 'padchest'
    elif model_type == 'densenet121_vin':
        model = get_densenet121_xray(name=name, args=args)
        model.dataset = 'vin'
    elif model_type in ['densenet121_' + str(x) for x in args.xray_datasets]:
        assert args.dataset in args.xray_datasets
        model = get_densenet121_xray(name=name, args=args)
    else:
        raise Exception(f'Unknown architecture: {model_type}')

    # Set the attributes if not already set.
    if getattr(model, 'dataset', None) == None:
        model.dataset = args.dataset
    if getattr(model, 'model_type', None) == None:
        model.model_type = model_type

    return model


def get_private_model_by_id(args, id=0):
    model_type = get_model_type_by_id(args=args, id=id)
    name = get_model_name_by_id(id=id)
    model = get_private_model(name=name, args=args, model_type=model_type)
    return model
