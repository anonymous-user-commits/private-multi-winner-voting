import os

import torch

import utils
from datasets.deprecated.chexpert.bin import train_chexpert
from datasets.deprecated.chexpert.chexpert_utils import get_chexpert_dev_loader
from datasets.utils import show_dataset_stats
from errors import check_perfect_balance_type
from models.private_model import get_private_model_by_id
from models.utils_models import get_model_name_by_id
from utils import eval_distributed_model
from utils import from_result_to_str
from utils import metric
from utils import train_model
from utils import update_summary


def query_ensemble_model_with_virtual_parties(args, file):
    if args.num_models % args.num_querying_parties != 0:
        raise ValueError(
            'Make sure that the number of virtual parties created is divisible by the total models.')
    args.num_models = args.num_querying_parties
    # Data loaders
    if args.dataset_type == 'imbalanced':
        all_private_trainloaders = utils.load_private_data_imbalanced(args)
    elif args.dataset_type == 'balanced':
        if args.balance_type == 'standard':
            all_private_trainloaders = utils.load_private_data(args)
        elif args.balance_type == 'perfect':
            check_perfect_balance_type(args=args)
            all_private_trainloaders = utils.load_private_data_imbalanced(
                args)
        else:
            raise Exception(
                f'Unknown args.balance_type: {args.balance_type}.')
    else:
        raise Exception(f'Unknown dataset type: {args.dataset_type}.')

    evalloader = utils.load_evaluation_dataloader(args)

    # Logs about the eval set
    show_dataset_stats(dataset=evalloader.dataset,
                       args=args, file=file, dataset_name='eval')

    # Training
    summary = {
        metric.loss: [],
        metric.acc: [],
        metric.balanced_acc: [],
        'auc': [],
    }
    for id in range(args.begin_id, args.end_id):
        utils.augmented_print("##########################################",
                              file)

        # Private model for initial training.
        model = get_private_model_by_id(args=args, id=id)

        trainloader = all_private_trainloaders[id]

        # Logs about the train set
        show_dataset_stats(dataset=trainloader.dataset,
                           args=args,
                           file=file,
                           dataset_name='private train')
        utils.augmented_print(
            "Steps per epoch: {:d}".format(len(trainloader)), file)

        if args.dataset.startswith('chexpert'):
            devloader = get_chexpert_dev_loader(args=args)
            result, best_model = train_chexpert.run(
                args=args,
                model=model,
                dataloader_train=trainloader,
                dataloader_dev=devloader,
                dataloader_eval=evalloader,
            )
        else:
            train_model(
                args=args,
                model=model,
                trainloader=trainloader,
                evalloader=evalloader)
            result = eval_distributed_model(
                model=model, dataloader=evalloader, args=args)

        model_name = get_model_name_by_id(id=id)
        result['model_name'] = model_name
        result_str = from_result_to_str(result=result, sep=' | ',
                                        inner_sep=': ')
        utils.augmented_print(text=result_str, file=file, flush=True)
        summary = update_summary(summary=summary, result=result)

        # Checkpoint
        state = result
        state['state_dict'] = model.state_dict()
        filename = "checkpoint-{}.pth.tar".format(model_name)
        filepath = os.path.join(args.private_model_path, filename)
        torch.save(state, filepath)
