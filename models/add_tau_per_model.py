import os
from parameters import get_parameters
import utils
from models.load_models import load_private_model_by_id
from general_utils.compute_taus import compute_taus_per_label
import torch


def get_model_probs_targets(model, args, dataloader):
    dataset = dataloader.dataset
    dataset_len = len(dataset)
    probs = None
    targets = None
    model = utils.distribute_model(args=args, model=model)
    end = 0
    for data, target in dataloader:
        batch_size = data.shape[0]
        begin = end
        end = begin + batch_size
        if args.cuda:
            data = data.cuda()
        # Generate raw ensemble votes
        output = model(data)
        output = output.detach().cpu()

        prob_outputs = torch.sigmoid(output)

        if probs is None:
            num_labels = prob_outputs.shape[1]
            probs = torch.zeros((dataset_len, num_labels))
            targets = torch.zeros((dataset_len, num_labels))

        probs[begin:end] = prob_outputs
        targets[begin:end] = target

    return probs, targets


def set_taus(args):
    all_private_trainloaders = utils.load_private_data(args=args)
    evalloader = utils.load_evaluation_dataloader(args)

    for id in range(args.begin_id, args.end_id):
        trainloader = all_private_trainloaders[id]
        model = load_private_model_by_id(args=args, id=id,
                                         model_path=args.save_model_path)
        model_name = model.name
        args.sigmoid_op = 'apply'
        result = utils.eval_distributed_model(
            model=model, dataloader=evalloader, args=args)

        result_str = utils.from_result_to_str(result=result, sep='\n',
                                              inner_sep=args.sep)
        print('before set taus: ', result_str)

        # all_targets = utils.get_all_targets_numpy(dataloader=trainloader,
        #                                           args=args)
        probs, targets = get_model_probs_targets(model=model, args=args,
                                                 dataloader=trainloader)

        probs = probs.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        taus, label_weights = compute_taus_per_label(votes=probs,
                                                     targets=targets)
        model.set_op_threshs(op_threshs=taus)

        args.sigmoid_op = 'disable'
        result = utils.eval_distributed_model(
            model=model, dataloader=evalloader, args=args)

        result_str = utils.from_result_to_str(result=result, sep='\n',
                                              inner_sep=args.sep)
        print('after set taus: ', result_str)

        # Checkpoint
        state = result
        del model.op_threshs
        state['state_dict'] = model.state_dict()
        state['taus'] = taus
        state['label_weights'] = label_weights
        filename = "checkpoint-{}.pth.tar".format(model_name)
        filepath = os.path.join(args.private_model_path, filename)
        torch.save(state, filepath)


if __name__ == "__main__":
    args = get_parameters()
    set_taus(args=args)
