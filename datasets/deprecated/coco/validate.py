# Adopted from: https://github.com/allenai/elastic/blob/master/multilabel_classify.py
# special thanks to @hellbell
from getpass import getuser

import argparse
import numpy as np
import os
import time
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed

from datasets.deprecated.coco.get_coco_data import get_coco_dataset
from datasets.deprecated.coco.helper_functions.helper_functions import AverageMeter
from datasets.deprecated.coco.helper_functions.helper_functions import mAP
from datasets.deprecated.coco.models import create_model

user = getuser()

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset_path', metavar='DIR',
                    default=f'/home/{user}/data/coco/',
                    help='Path to the dataset.')
parser.add_argument(
    '--model-name',
    default='tresnet_m',
    # default='tresnet_l',
    # default='tresnet_xl',
)
parser.add_argument(
    '--model-path',
    default=f'/home/{user}/code/capc-learning/private-models/coco/tresnet_m/50-models/checkpoint-model(12).pth.tar',
    # default=f'/home/{user}/models/multi_label/coco/MS_COCO_TRresNet_L_448_86.6.pth',
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
parser.add_argument('-b', '--batch_size', default=16, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--print-freq', '-p', default=64, type=int,
                    metavar='N', help='print frequency')
parser.add_argument('--coco_version', default='2017', type=str,
                    help='the year of the dataset')
parser.add_argument('--debug', type=bool, help="Debug mode of execution",
                    # default=True,
                    default=False,
                    )
parser.add_argument('--coco_data_loader', type=str,
                    help='standard or custom data loader, where custom uses'
                         'the pre-generated labels',
                    default='custom',
                    )

from datasets.deprecated.coco.helper_functions.helper_functions import coco_classes_list


def main():
    args = parser.parse_args()
    args.cwd = os.getcwd()
    # setup model
    print('creating and loading the model...')
    state = torch.load(args.model_path, map_location='cpu')
    # args.num_classes = state['num_classes']
    args.num_classes = getattr(state, 'num_classes', args.num_classes)
    args.do_bottleneck_head = False
    model = create_model(args).cuda()
    if 'model' in state:
        state_dict = state['model']
    elif 'state_dict' in state:
        state_dict = state['state_dict']
    else:
        raise Exception('Cannot extract state_dict from the state.')
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    if 'idx_to_class' in state:
        classes_list = np.array(list(state['idx_to_class'].values()))
    else:
        classes_list = coco_classes_list
    print('done\n')

    # data_type = 'smallval'
    data_type = 'val'
    # data_type = 'train'

    val_dataset = get_coco_dataset(args=args, data_type=data_type)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    validate_multi(val_loader, model, args, classes_list)


def validate_multi(val_loader, model, args, classes_list):
    print("starting actual validation")
    batch_time = AverageMeter()
    prec = AverageMeter()
    rec = AverageMeter()

    Sig = torch.nn.Sigmoid()

    end = time.time()
    tp = torch.zeros(args.num_classes)
    fp = torch.zeros(args.num_classes)
    fn = torch.zeros(args.num_classes)
    tn = torch.zeros(args.num_classes)
    count = 0
    preds = []
    targets = []
    for batch_index, (input, target) in enumerate(val_loader):
        # target = target.max(dim=1)[0] # this is on the level of whole batch
        # compute output
        with torch.no_grad():
            output = Sig(model(input.cuda())).cpu()

        # for mAP calculation
        preds.append(output.cpu())
        targets.append(target.cpu())

        # measure accuracy and record loss
        pred = output.data.gt(args.coco_threshold).long()
        if args.debug:
            # print('target: ', target)
            # print('pred: ', pred)
            print('target names: ',
                  classes_list[target[0].cpu().detach().numpy() == 1])
            print('pred names: ',
                  classes_list[pred[0].cpu().detach().numpy() == 1])

        tp += (pred + target).eq(2).sum(dim=0)
        fp += (pred - target).eq(1).sum(dim=0)
        fn += (pred - target).eq(-1).sum(dim=0)
        tn += (pred + target).eq(0).sum(dim=0)
        count += input.size(0)

        this_tp = (pred + target).eq(2).sum()
        this_fp = (pred - target).eq(1).sum()
        this_fn = (pred - target).eq(-1).sum()
        this_tn = (pred + target).eq(0).sum()

        this_prec = this_tp.float() / (
                this_tp + this_fp).float() * 100.0 if this_tp + this_fp != 0 else 0.0
        this_rec = this_tp.float() / (
                this_tp + this_fn).float() * 100.0 if this_tp + this_fn != 0 else 0.0

        prec.update(float(this_prec), input.size(0))
        rec.update(float(this_rec), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        p_c, r_c, f_c = [], [], []
        for i in range(len(tp)):
            if tp[i] > 0:
                pc = float(tp[i].float() / (tp[i] + fp[i]).float()) * 100.0
                rc = float(tp[i].float() / (tp[i] + fn[i]).float()) * 100.0
                fc = 2 * pc * rc / (pc + rc)
            else:
                pc, rc, fc = 0.0, 0.0, 0.0
            p_c.append(pc)
            r_c.append(rc)
            f_c.append(fc)

        mean_p_c = sum(p_c) / len(p_c)
        mean_r_c = sum(r_c) / len(r_c)
        mean_f_c = sum(f_c) / len(f_c)

        p_o = tp.sum().float() / (tp + fp).sum().float() * 100.0
        r_o = tp.sum().float() / (tp + fn).sum().float() * 100.0
        f_o = 2 * p_o * r_o / (p_o + r_o)

        if batch_index % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Precision {prec.val:.2f} ({prec.avg:.2f})\t'
                  'Recall {rec.val:.2f} ({rec.avg:.2f})'.format(
                batch_index, len(val_loader), batch_time=batch_time,
                prec=prec, rec=rec))
            print(
                'P_C {:.2f} R_C {:.2f} F_C {:.2f} P_O {:.2f} R_O {:.2f} F_O {:.2f}'
                    .format(mean_p_c, mean_r_c, mean_f_c, p_o, r_o, f_o))

    mAP_score = mAP(torch.cat(targets).numpy(), torch.cat(preds).numpy())
    print("mAP score:", mAP_score)

    return


if __name__ == '__main__':
    start_time = time.time()

    main()

    stop_time = time.time()
    elapsed_time = stop_time - start_time
    print('elapsed_time: ', elapsed_time)
