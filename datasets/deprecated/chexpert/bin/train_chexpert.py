import copy

import argparse
import json
import logging
import numpy as np
import os
import subprocess
import sys
import time
import torch
import torch.nn.functional as F
from shutil import copyfile
from sklearn import metrics
from tensorboardX import SummaryWriter
from torch.nn import DataParallel
from torch.utils.data import DataLoader

from parameters import get_chexpert_paremeters
from utils import count_samples_per_class, get_optimizer
from utils import get_scheduler
from utils import get_device

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

from datasets.deprecated.chexpert.data.single_disease import SingleDiseaseDataset  # noqa
from model.classifier import Classifier  # noqa
from utils import get_cfg, class_wise_loss_reweighting  # noqa
from utils import get_timestamp
from utils import get_auc
from utils import lr_schedule  # noqa


def get_data_t(args, data, t):
    """
    Extract index t from the data based on the data type.
    :param args: the arguments for the program
    :param data: the data tensor
    :param t: the index
    :return: the extracted part of data at index t
    """
    if args.chexpert_dataset_type in ['multi', 'single']:
        data = data[:, t]
    elif args.chexpert_dataset_type in ['pos']:
        assert t == 0
    else:
        raise Exception(
            f'Unknown args.chexpert_dataset_type: {args.chexpert_dataset_type}.')
    return data


def get_target_output_t(args, target, output, t):
    """
    Get target and output from index t.

    :param args: the arguments of the program
    :param target: the expected target
    :param output: output from a model
    :param t: the index
    :return: the target and output at position t
    """
    if args.chexpert_dataset_type == 'pos':
        # We have one dimensional target with expected labels and output with
        # predicted labels.
        assert t == 0
    else:
        target = target[:, t]
        output = output[t]
    return target, output


def get_loss(output, target, t, device, cfg, args):
    target, output = get_target_output_t(args=args, target=target,
                                         output=output, t=t)

    if cfg.criterion == 'BCE':
        for num_class in cfg.num_classes:
            assert num_class == 1
        target = target.view(-1)
        output = output.view(-1)
        pos_weight = torch.from_numpy(
            np.array(cfg.pos_weight,
                     dtype=np.float32)).to(device).type_as(target)
        if cfg.batch_weight:
            if target.sum() == 0:
                loss = torch.tensor(0., requires_grad=True).to(device)
            else:
                weight = (target.size()[0] - target.sum()) / (target.sum())
                loss = F.binary_cross_entropy_with_logits(
                    output, target, pos_weight=weight)
        else:
            loss = F.binary_cross_entropy_with_logits(
                output, target, pos_weight=pos_weight[t])

        pred = torch.sigmoid(output).ge(0.5).float()
    elif cfg.criterion == 'CE':
        assert len(cfg.num_classes) == 1
        assert cfg.num_classes[0] == 2
        if cfg.weight_beta == -1.0:
            loss = F.cross_entropy(input=output, target=target)
        else:
            loss = F.cross_entropy(
                input=output, target=target,
                weight=torch.tensor(
                    cfg.class_weights,
                    device=output.device,
                    dtype=output.dtype,
                )
            )
        pred = output.argmax(dim=1)
    else:
        raise Exception(f'Unknown criterion: {cfg.criterion}')

    acc = pred.eq(target).float().sum() / len(target)
    balanced_acc = metrics.balanced_accuracy_score(
        y_true=target.detach().cpu().numpy(),
        y_pred=pred.detach().cpu().numpy(),
    )
    return (loss, acc, balanced_acc)


def set_class_weights(cfg, dataloader):
    # train_class_count = dataset.sample_counts_per_class()
    train_class_count = count_samples_per_class(dataloader=dataloader)
    samples_per_cls = list(train_class_count.values())
    if cfg.weight_beta == -1:
        class_weights = np.ones_like(samples_per_cls)
    else:
        class_weights = class_wise_loss_reweighting(
            beta=cfg.weight_beta,
            samples_per_cls=samples_per_cls,
        )
    cfg.class_weights = class_weights
    return class_weights


def train_epoch(summary, summary_dev, cfg, args, model, dataloader,
                dataloader_dev, optimizer, summary_writer, best_dict,
                dev_header):
    torch.set_grad_enabled(True)
    model.train_features_epoch()
    device = torch.device('cuda:{}'.format(args.device_ids[0]))
    if cfg.train_steps == 0:
        steps = len(dataloader)
    else:
        steps = cfg.train_steps
    # print('steps: ', steps)
    dataiter = iter(dataloader)
    label_header = cfg._label_header
    num_tasks = len(cfg.num_classes)

    time_now = time.time()
    loss_sum = np.zeros(num_tasks)
    summary_loss = np.zeros(num_tasks)
    acc_sum = np.zeros(num_tasks)
    balanced_acc_sum = np.zeros(num_tasks)
    best_model = copy.deepcopy(model)

    for step in range(steps):
        image, target = next(dataiter)
        image = image.to(device)
        target = target.to(device)
        output = model(image)

        # different number of tasks
        loss = 0
        for t in range(num_tasks):
            loss_t, acc_t, balanced_acc_t = get_loss(
                output=output, target=target, t=t,
                device=device, cfg=cfg, args=args)
            loss += loss_t
            loss_sum[t] += loss_t.item()
            summary_loss[t] += loss_t.item()
            acc_sum[t] += acc_t.item()
            balanced_acc_sum[t] += balanced_acc_t

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        summary['step'] += 1

        if cfg.log_every != 0 and summary['step'] % cfg.log_every == 0:
            time_spent = time.time() - time_now
            time_now = time.time()

            loss_sum /= cfg.log_every
            acc_sum /= cfg.log_every
            balanced_acc_sum /= cfg.log_every
            loss_str = ' '.join(map(lambda x: '{:.5f}'.format(x), loss_sum))
            acc_str = ' '.join(map(lambda x: '{:.3f}'.format(x), acc_sum))
            balanced_acc_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
                                            balanced_acc_sum))

            logging.info(
                '{}, Train, Epoch : {}, Step : {}, Loss : {}, '
                'Acc : {}, Balanced Acc : {}, Run Time : {:.2f} sec'
                    .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                            summary['epoch'], summary['step'], loss_str,
                            acc_str, balanced_acc_str, time_spent))

            for t in range(num_tasks):
                summary_writer.add_scalar(
                    'train/loss_{}'.format(label_header[t]), loss_sum[t],
                    summary['step'])
                summary_writer.add_scalar(
                    'train/acc_{}'.format(label_header[t]), acc_sum[t],
                    summary['step'])
                summary_writer.add_scalar(
                    'train/balanced_acc_{}'.format(label_header[t]),
                    balanced_acc_sum[t],
                    summary['step'])

            loss_sum = np.zeros(num_tasks)
            acc_sum = np.zeros(num_tasks)
            balanced_acc_sum = np.zeros(num_tasks)

        if cfg.dev_every != 0 and summary['step'] % cfg.dev_every == 0:
            time_now = time.time()
            summary_dev, predlist, true_list = test_epoch(
                summary_dev, cfg, args, model, dataloader_dev)
            time_spent = time.time() - time_now

            auclist = []
            for i in range(len(cfg.num_classes)):
                y_pred = predlist[i]
                y_true = true_list[i]

                auc = get_auc(
                    classification_type=cfg.classification_type,
                    y_true=y_true,
                    y_pred=y_pred,
                    num_classes=cfg.num_classes[i],
                )

                auclist.append(auc)
            summary_dev['auc'] = np.array(auclist)

            loss_dev_str = ' '.join(map(lambda x: '{:.5f}'.format(x),
                                        summary_dev['loss']))
            acc_dev_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
                                       summary_dev['acc']))
            balanced_acc_dev_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
                                                summary_dev['balanced_acc']))
            auc_dev_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
                                       summary_dev['auc']))

            logging.info(
                '{}, Dev, '
                'Step : {}, '
                'Loss : {}, '
                'Acc : {}, '
                'Balanced Acc : {}, '
                'Auc : {},'
                'Mean auc: {:.3f}, '
                'Run Time : {:.2f} sec'.format(
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    summary['step'],
                    loss_dev_str,
                    acc_dev_str,
                    balanced_acc_dev_str,
                    auc_dev_str,
                    summary_dev['auc'].mean(),
                    time_spent))

            for t in range(len(cfg.num_classes)):
                summary_writer.add_scalar(
                    'dev/loss_{}'.format(dev_header[t]),
                    summary_dev['loss'][t], summary['step'])
                summary_writer.add_scalar(
                    'dev/acc_{}'.format(dev_header[t]), summary_dev['acc'][t],
                    summary['step'])
                summary_writer.add_scalar(
                    'dev/balanced_acc_{}'.format(dev_header[t]),
                    summary_dev['balanced_acc'][t],
                    summary['step'])
                summary_writer.add_scalar(
                    'dev/auc_{}'.format(dev_header[t]), summary_dev['auc'][t],
                    summary['step'])

            save_best = False

            mean_acc = summary_dev['acc'][cfg.save_index].mean()
            if mean_acc >= best_dict['acc_dev_best']:
                best_dict['acc_dev_best'] = mean_acc
                if cfg.best_target == 'acc':
                    save_best = True

            mean_balanced_acc = summary_dev['balanced_acc'][
                cfg.save_index].mean()
            if mean_balanced_acc >= best_dict['balanced_acc_dev_best']:
                best_dict['balanced_acc_dev_best'] = mean_balanced_acc
                if cfg.best_target == 'balanced_acc':
                    save_best = True

            mean_auc = summary_dev['auc'][cfg.save_index].mean()
            if mean_auc >= best_dict['auc_dev_best']:
                best_dict['auc_dev_best'] = mean_auc
                if cfg.best_target == 'auc':
                    save_best = True

            mean_loss = summary_dev['loss'][cfg.save_index].mean()
            if mean_loss <= best_dict['loss_dev_best']:
                best_dict['loss_dev_best'] = mean_loss
                if cfg.best_target == 'loss':
                    save_best = True

            if save_best:
                best_model = copy.deepcopy(model)
                torch.save(
                    {'epoch': summary['epoch'],
                     'step': summary['step'],
                     'acc_dev_best': best_dict['acc_dev_best'],
                     'balanced_acc_dev_best': best_dict[
                         'balanced_acc_dev_best'],
                     'auc_dev_best': best_dict['auc_dev_best'],
                     'loss_dev_best': best_dict['loss_dev_best'],
                     'state_dict': model.module.state_dict()},
                    os.path.join(args.save_path, 'best{}.ckpt'.format(
                        best_dict['best_idx']))
                )
                best_dict['best_idx'] += 1
                if best_dict['best_idx'] > cfg.save_top_k:
                    best_dict['best_idx'] = 1
                logging.info(
                    '{}, Best, '
                    'Step : {}, '
                    'Loss : {}, '
                    'Acc : {}, '
                    'Balanced Acc : {}, '
                    'Auc :{}, '
                    'Best Auc : {:.3f}'.format(
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        summary['step'],
                        loss_dev_str,
                        acc_dev_str,
                        balanced_acc_dev_str,
                        auc_dev_str,
                        best_dict['auc_dev_best']))
        model.train_features_epoch()
        torch.set_grad_enabled(True)

    summary['epoch'] += 1

    summary_loss /= steps
    summary['loss'] = summary_loss

    return summary, best_dict, best_model


def test_epoch(summary, cfg, args, model, dataloader):
    torch.set_grad_enabled(False)
    model.eval()
    device = torch.device('cuda:{}'.format(args.device_ids[0]))
    steps = len(dataloader)
    dataiter = iter(dataloader)
    num_tasks = len(cfg.num_classes)

    loss_sum = np.zeros(num_tasks)
    acc_sum = np.zeros(num_tasks)
    balanced_acc_sum = np.zeros(num_tasks)

    pred_list = list(x for x in range(len(cfg.num_classes)))
    true_list = list(x for x in range(len(cfg.num_classes)))
    for step in range(steps):
        image, target = next(dataiter)
        image = image.to(device)
        target = target.to(device)
        output = model(image)
        # different number of tasks
        for t in range(len(cfg.num_classes)):
            loss_t, acc_t, balanced_acc_t = get_loss(
                output=output, target=target, t=t,
                device=device, cfg=cfg, args=args)
            target, output = get_target_output_t(args=args, target=target,
                                                 output=output, t=t)
            # AUC
            if cfg.criterion == 'BCE':
                output_tensor = torch.sigmoid(
                    output.view(-1)).cpu().detach().numpy()
            elif cfg.criterion == 'CE':
                output_tensor = torch.softmax(output, dim=1)
                output_tensor = output_tensor.cpu().detach().numpy()
            else:
                raise Exception(f"Unknown criterion: {cfg.criterion}")

            target_tensor = target.view(-1).cpu().detach().numpy()
            if step == 0:
                pred_list[t] = output_tensor
                true_list[t] = target_tensor
            else:
                pred_list[t] = np.concatenate(
                    (pred_list[t], output_tensor),
                    axis=0)
                true_list[t] = np.append(true_list[t], target_tensor)

            loss_sum[t] += loss_t.item()
            acc_sum[t] += acc_t.item()
            balanced_acc_sum[t] += balanced_acc_t
    summary['loss'] = loss_sum / steps
    summary['acc'] = acc_sum / steps
    summary['balanced_acc'] = balanced_acc_sum / steps

    return summary, pred_list, true_list


def log_result(cfg, predlist, true_list, summary_dev, summary_train,
               summary_writer, epoch, time_spent, best_dict, model, args):
    auclist = []
    for i in range(len(cfg.num_classes)):
        y_pred = predlist[i]
        y_true = true_list[i]

        auc = get_auc(
            classification_type=cfg.classification_type,
            y_true=y_true,
            y_pred=y_pred,
            num_classes=cfg.num_classes[i],
        )

        auclist.append(auc)
    summary_dev['auc'] = np.array(auclist)

    loss_dev_str = ' '.join(map(lambda x: '{:.5f}'.format(x),
                                summary_dev['loss']))
    acc_dev_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
                               summary_dev['acc']))
    balanced_acc_dev_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
                                        summary_dev['balanced_acc']))
    auc_dev_str = ' '.join(map(lambda x: '{:.3f}'.format(x),
                               summary_dev['auc']))

    logging.info(
        '{}, Dev, '
        'Epoch : {}, '
        'Step : {}, '
        'Loss : {}, '
        'Acc : {}, '
        'Balanced Acc : {}, '
        'Auc : {}, '
        'Mean auc: {:.3f}, '
        'Run Time : {:.2f} sec'.format(
            time.strftime("%Y-%m-%d %H:%M:%S"),
            epoch,
            summary_train['step'],
            loss_dev_str,
            acc_dev_str,
            balanced_acc_dev_str,
            auc_dev_str,
            summary_dev['auc'].mean(),
            time_spent))

    dev_header = cfg._label_header
    for t in range(len(cfg.num_classes)):
        summary_writer.add_scalar(
            'dev/loss_{}'.format(dev_header[t]), summary_dev['loss'][t],
            summary_train['step'])
        summary_writer.add_scalar(
            'dev/acc_{}'.format(dev_header[t]), summary_dev['acc'][t],
            summary_train['step'])
        summary_writer.add_scalar(
            'dev/balanced_acc_{}'.format(dev_header[t]),
            summary_dev['balanced_acc'][t],
            summary_train['step'])
        summary_writer.add_scalar(
            'dev/auc_{}'.format(dev_header[t]), summary_dev['auc'][t],
            summary_train['step'])

    save_best = False
    mean_acc = summary_dev['acc'][cfg.save_index].mean()
    if mean_acc >= best_dict['acc_dev_best']:
        best_dict['acc_dev_best'] = mean_acc
        if cfg.best_target == 'acc':
            save_best = True

    mean_balanced_acc = summary_dev['balanced_acc'][cfg.save_index].mean()
    if mean_balanced_acc >= best_dict['balanced_acc_dev_best']:
        best_dict['balanced_acc_dev_best'] = mean_balanced_acc
        if cfg.best_target == 'balanced_acc':
            save_best = True

    mean_auc = summary_dev['auc'][cfg.save_index].mean()
    if mean_auc >= best_dict['auc_dev_best']:
        best_dict['auc_dev_best'] = mean_auc
        if cfg.best_target == 'auc':
            save_best = True

    mean_loss = summary_dev['loss'][cfg.save_index].mean()
    if mean_loss <= best_dict['loss_dev_best']:
        best_dict['loss_dev_best'] = mean_loss
        if cfg.best_target == 'loss':
            save_best = True

    if save_best:
        torch.save(
            {'epoch': summary_train['epoch'],
             'step': summary_train['step'],
             'acc_dev_best': best_dict['acc_dev_best'],
             'balanced_acc_dev_best': best_dict['balanced_acc_dev_best'],
             'auc_dev_best': best_dict['auc_dev_best'],
             'loss_dev_best': best_dict['loss_dev_best'],
             'state_dict': model.module.state_dict()},
            os.path.join(args.save_path,
                         'best{}.ckpt'.format(best_dict['best_idx']))
        )
        best_dict['best_idx'] += 1
        if best_dict['best_idx'] > cfg.save_top_k:
            best_dict['best_idx'] = 1
        logging.info(
            '{}, Best, '
            'Step : {}, '
            'Loss : {}, '
            'Acc : {}, '
            'Balanced Acc : {}, '
            'Auc : {}, '
            'Best Auc : {:.3f}'.format(
                time.strftime("%Y-%m-%d %H:%M:%S"),
                summary_train['step'],
                loss_dev_str,
                acc_dev_str,
                balanced_acc_dev_str,
                auc_dev_str,
                best_dict['auc_dev_best']))
    torch.save({'epoch': summary_train['epoch'],
                'step': summary_train['step'],
                'acc_dev_best': best_dict['acc_dev_best'],
                'balanced_acc_dev_best': best_dict['balanced_acc_dev_best'],
                'auc_dev_best': best_dict['auc_dev_best'],
                'loss_dev_best': best_dict['loss_dev_best'],
                'state_dict': model.module.state_dict()},
               os.path.join(args.save_path, 'train.ckpt'))


def run(args, model, dataloader_train, dataloader_dev, dataloader_eval):
    cfg = args.cfg
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if args.logtofile is True:
        logging.basicConfig(filename=args.save_path + '/log.txt',
                            filemode="w", level=logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO)

    if not args.resume:
        with open(os.path.join(args.save_path, 'cfg.json'), 'w') as f:
            for key, value in cfg.items():
                if isinstance(value, np.ndarray):
                    cfg[key] = value.tolist()
            json.dump(cfg, f, indent=1)

    device, device_ids = get_device(args=args)

    if args.verbose is True:
        from torchsummary import summary
        if cfg.fix_ratio:
            h, w = cfg.long_side, cfg.long_side
        else:
            h, w = cfg.height, cfg.width
        summary(model.to(device), (3, h, w))
    model = DataParallel(model, device_ids=device_ids).to(device).train()
    if args.pre_train is not None:
        if os.path.exists(args.pre_train):
            ckpt = torch.load(args.pre_train, map_location=device)
            model.module.load_state_dict(ckpt)

    optimizer = get_optimizer(model.parameters(), args=args)

    scheduler = get_scheduler(args=args, optimizer=optimizer)

    copy_src_folder = False
    if copy_src_folder:
        src_folder = os.path.dirname(os.path.abspath(__file__)) + '/../'
        dst_folder = os.path.join(args.save_path, 'classification')
        rc, size = subprocess.getstatusoutput(
            'du --max-depth=0 %s | cut -f1' % src_folder)
        if rc != 0:
            raise Exception('Copy folder error : {}'.format(rc))
        print('size: ', size)
        rc, err_msg = subprocess.getstatusoutput(
            'cp -R %s %s' % (src_folder, dst_folder))
        if rc != 0:
            raise Exception('copy folder error : {}'.format(err_msg))

    copyfile(cfg.data_path + cfg.train_csv,
             os.path.join(args.save_path, 'train.csv'))
    copyfile(cfg.data_path + cfg.dev_csv,
             os.path.join(args.save_path, 'dev.csv'))
    if cfg.weight_beta != -1:
        cfg.class_weights = set_class_weights(
            cfg=cfg, dataloader=dataloader_train.dataset)

    summary_train = {'epoch': 0, 'step': 0}
    summary_dev = {'loss': float('inf'), 'acc': 0.0}
    summary_writer = SummaryWriter(args.save_path)
    epoch_start = 0
    best_model = model
    best_dict = {
        "acc_dev_best": 0.0,
        "balanced_acc_dev_best": 0.0,
        "auc_dev_best": 0.0,
        "loss_dev_best": float('inf'),
        "fused_dev_best": 0.0,
        "best_idx": 1
    }

    if args.resume:
        ckpt_path = os.path.join(args.save_path, 'train.ckpt')
        ckpt = torch.load(ckpt_path, map_location=device)
        model.module.load_state_dict(ckpt['state_dict'])
        best_model = model
        summary_train = {'epoch': ckpt['epoch'], 'step': ckpt['step']}
        best_dict['acc_dev_best'] = ckpt['acc_dev_best']
        best_dict['balanced_acc_dev_best'] = ckpt['balanced_acc_dev_best']
        best_dict['loss_dev_best'] = ckpt['loss_dev_best']
        best_dict['auc_dev_best'] = ckpt['auc_dev_best']
        epoch_start = ckpt['epoch']

    epoch = epoch_start
    train_loss = 0
    for epoch in range(epoch_start, args.num_epochs):
        start_epoch_time = time.time()
        summary_train, best_dict, best_model = train_epoch(
            summary_train, summary_dev, cfg, args, model,
            dataloader_train, dataloader_dev, optimizer,
            summary_writer, best_dict, dev_header=cfg._label_header)
        end_epoch_time = time.time()
        epoch_time = end_epoch_time - start_epoch_time

        losses = summary_train['loss']
        train_loss = losses.mean()

        if epoch == 0:
            header_log = ['epoch', 'train_loss', 'epoch_time']
            print(";".join([str(item) for item in header_log]))
        data_log = [epoch, train_loss, epoch_time]
        print(";".join([str(item) for item in data_log]))

        if args.scheduler_type == 'custom':
            lr = lr_schedule(args.lr, args.lr_factor, summary_train['epoch'],
                             args.lr_epochs)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        else:
            scheduler.step(train_loss)

        if cfg.test_every != 0 and epoch % cfg.test_every == 0:
            time_now = time.time()
            summary_dev, predlist, true_list = test_epoch(
                summary_dev, cfg, args, best_model, dataloader_eval)
            time_spent = time.time() - time_now

            log_result(cfg=cfg, predlist=predlist, true_list=true_list,
                       summary_dev=summary_dev, summary_train=summary_train,
                       summary_writer=summary_writer, epoch=epoch,
                       time_spent=time_spent, best_dict=best_dict,
                       model=best_model,
                       args=args)

    time_now = time.time()
    summary_dev, predlist, true_list = test_epoch(
        summary_dev, cfg, args, best_model, dataloader_eval)
    time_spent = time.time() - time_now

    log_result(cfg=cfg, predlist=predlist, true_list=true_list,
               summary_dev=summary_dev, summary_train=summary_train,
               summary_writer=summary_writer, epoch=epoch,
               time_spent=time_spent, best_dict=best_dict, model=best_model,
               args=args)

    summary_writer.close()
    result = {}
    result['train_loss'] = train_loss

    for key in ['acc', 'balanced_acc', 'auc', 'loss']:
        # print(key, ": ", summary_dev[key])
        value = summary_dev[key]
        if isinstance(value, np.ndarray):
            value = value[cfg.save_index].mean()
            result[key] = value

    for key in ['acc', 'balanced_acc', 'auc']:
        value = summary_dev[key][cfg.save_index]
        result[key + '_detailed'] = value

    # Extract the best model from DataParallel object.
    best_model = best_model.module

    return result, best_model


def get_data(args):
    cfg = args.cfg
    dataset_train = SingleDiseaseDataset(
        in_csv_path=cfg.train_csv, cfg=cfg,
        mode='train')
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True, shuffle=True)
    dataset_dev = SingleDiseaseDataset(
        in_csv_path=cfg.dev_csv, cfg=cfg, mode='dev')
    dataloader_dev = DataLoader(
        dataset_dev,
        batch_size=args.eval_batch_size, num_workers=args.num_workers,
        drop_last=False, shuffle=False)
    return dataloader_train, dataloader_dev


def main():
    timestamp = get_timestamp()

    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of workers to fetch data.')
    parser.add_argument('--cfg_path',
                        # default='../config/ady_small.json',
                        default='../config/single_disease_small.json',
                        metavar='CFG_PATH', type=str,
                        help="Path to the config file in json format")
    parser.add_argument('--verbose',
                        default=True,
                        type=bool,
                        help="Detail info")
    parser = get_chexpert_paremeters(parser=parser, timestamp=timestamp)
    args = parser.parse_args()

    if args.verbose is True:
        print('Using the specified args:')
        print(args)

    args.cfg = get_cfg(args.cfg_path)
    if args.verbose is True:
        print(json.dumps(args.cfg, indent=4))

    dataloader_train, dataloader_dev = get_data(args=args)

    model = Classifier(cfg=args.cfg)
    run(args=args, model=model, dataloader_train=dataloader_train,
        dataloader_dev=dataloader_dev)


if __name__ == '__main__':
    start_time = time.time()
    main()
    stop_time = time.time()
    print('elapsed time: ', stop_time - start_time)
