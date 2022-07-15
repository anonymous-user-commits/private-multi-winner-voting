import math
import numpy as np
import os
import time
import torch
from sklearn import metrics
from torch.nn.functional import softmax
from tqdm import tqdm

from analysis.multiple_counting import sample_bounded_noise
from analysis.multiple_counting import sample_gaussian_noise
from autodp.utils import clip_votes_tensor
from models.ensemble_model import EnsembleModel
from models.load_models import load_private_model_by_id
from utils import augmented_print
from utils import compute_metrics_multilabel
from utils import compute_metrics_multilabel_from_preds_targets
from utils import distribute_model
from utils import from_confidence_scores_to_votes
from utils import from_str
from utils import get_all_targets_numpy
from utils import get_indexes
from utils import get_one_hot_confidence_bins
from utils import get_value_str
from utils import metric
from utils import one_hot
from utils import pick_labels_cols
from utils import pick_labels_general
from utils import result
from utils import generate_histogram_powerset
from utils import get_class_labels_and_map_powerset
from utils import get_vote_count_and_map_powerset


class BigEnsembleModel(EnsembleModel):
    """Noisy ensemble of private models.
    We evaluate the ensemble by querying the constituent models one by one.
    We load a model, query it, and then discard the model.
    """

    def __init__(self, model_id: int, args):
        super(BigEnsembleModel, self).__init__(
            model_id=model_id, private_models=None, args=args)
        print("Building big ensemble model '{}'!".format(self.name))

        self.num_classes = args.num_classes
        self.num_labels = args.num_classes

        # Skip the private model for the answering party id that
        # built this ensemble.
        self.model_ids = [i for i in
                          range(args.num_models)]  # if i != model_id]

    def __len__(self):
        return len(self.model_ids)

    def evaluate(self, evalloader, args):
        """Evaluate the accuracy of noisy ensemble model."""
        gap_list = np.zeros(args.num_classes, dtype=np.float64)
        correct = np.zeros(args.num_classes, dtype=np.int64)
        wrong = np.zeros(args.num_classes, dtype=np.int64)
        with torch.no_grad():
            for data, target in evalloader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                # Generate raw ensemble votes
                votes = torch.zeros((data.shape[0], self.num_classes))
                for model in self.models:
                    output = model(data)
                    onehot = one_hot(output.data.max(dim=1)[1].cpu(),
                                     self.num_classes)
                    votes += onehot
                # Add Gaussian noise
                assert args.sigma_gnmax >= 0
                if args.sigma_gnmax > 0:
                    noise = torch.from_numpy(
                        np.random.normal(0., args.sigma_gnmax, (
                            data.shape[0], self.num_classes))).float()
                    votes += noise
                sorted_votes = votes.sort(dim=-1, descending=True)[0]
                gaps = (sorted_votes[:, 0] - sorted_votes[:, 1]).numpy()
                preds = votes.max(dim=1)[1].numpy().astype(np.int64)
                target = target.data.cpu().numpy().astype(np.int64)
                for label, pred, gap in zip(target, preds, gaps):
                    gap_list[label] += gap
                    if label == pred:
                        correct[label] += 1
                    else:
                        wrong[label] += 1
        total = correct.sum() + wrong.sum()
        assert total == len(evalloader.dataset)
        return 100. * correct.sum() / total, 100. * correct / (
                correct + wrong), gap_list.sum() / total, gap_list / (
                       correct + wrong)

    def get_votes_confidence_scores(self, dataloader, args) -> (
            np.ndarray, np.ndarray):
        """

        Args:
            dataloader: torch data loader
            args: program arguments

        Returns:
            votes and softmax confidence scores for each data point

        """
        dataset = dataloader.dataset
        dataset_len = len(dataset)
        votes = torch.zeros((dataset_len, self.num_classes))
        num_models = len(self.model_ids)
        confidence_scores = torch.zeros(
            (num_models, dataset_len, self.num_classes))
        with torch.no_grad():
            for model_nr, id in enumerate(self.model_ids):
                model = load_private_model_by_id(
                    args=args, id=id, model_path=args.private_model_path)
                model = distribute_model(args=args, model=model)
                correct = 0
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
                    preds = output.argmax(dim=1)
                    labels = target.view_as(preds)
                    correct += preds.eq(labels).sum().item()
                    batch_votes = one_hot(preds, self.num_classes)
                    votes[begin:end] += batch_votes
                    softmax_scores = softmax(output, dim=1)
                    confidence_scores[model_nr, begin:end, :] = softmax_scores
                acc = correct / dataset_len
                print(f'model id {id} with acc: {acc}')
            votes = votes.numpy()
            assert np.all(votes.sum(axis=-1) == len(self.model_ids))
        return votes, confidence_scores

    def get_votes_multiclass_confidence(self, dataloader, args) -> np.ndarray:
        """

        Args:
            dataloader: torch data loader
            args: program arguments

        Returns:
            confidence scores for each data point

        """
        _, confidence_scores = self.get_votes_confidence_scores(
            dataloader=dataloader, args=args)
        return confidence_scores

    def get_votes_multiclass(self, dataloader, args) -> np.ndarray:
        """

        Args:
            dataloader: torch data loader
            args: program arguments

        Returns:
            votes for each data point

        """
        votes, _ = self.get_votes_confidence_scores(
            dataloader=dataloader, args=args)
        return votes

    def get_acc_votes_targets(self,
                              votes: torch.tensor,
                              targets: torch.tensor) -> (np.float,
                                                         np.ndarray):
        preds = votes.argmax(dim=-1)
        results = preds.eq(targets).to(torch.int)
        acc_detailed = results.to(float).mean(dim=-1).cpu().detach().numpy()
        acc = acc_detailed.mean()
        return acc, acc_detailed

    def get_preds(self, votes: np.ndarray, class_type: str, threshold: float):
        """
        Transform votes into predictions.

        :param votes: the votes - either counts of positive and negative votes
        for each label or the probability of a label being positive.
        :param class_type: the type of the classification task
        :param threshold: the probability threshold for predictions from the
        probabilities
        :return: the predictions
        """
        if class_type in ['multilabel', 'multiclass']:
            preds = votes.argmax(axis=-1)
        elif class_type == ['multilabel_counting',
                            'multilabel_counting_gaussian']:
            preds = np.array(votes > threshold)
        else:
            raise Exception(f"Unknown class_type: {class_type}.")
        return preds

    def get_votes_probs(self, dataloader, args, save_all_votes=False) -> (
            np.ndarray, np.ndarray, np.ndarray):
        """
        Get both: the votes for the multilabel and probs (probabilities) for the
        multilabel_counting.

        :param dataloader: data loader
        :param args: program params
        :param save_all_votes: save all votes separately
        :return: votes, probs
        """
        dataset = dataloader.dataset
        dataset_len = len(dataset)
        num_labels = self.num_labels

        # Save all votes independent of the args.pick_labels.
        # if args.pick_labels is not None and args.pick_labels != [-1]:
        #     num_labels = len(args.pick_labels)

        sum_acc = None  # sum accuracy from all models
        sum_balanced_acc = None
        sum_auc = None
        sum_map = None
        sum_acc_detailed = None  # sum detailed accuracy from all models
        sum_balanced_acc_detailed = None
        nr_models = len(self.model_ids)

        if save_all_votes or ((args.debug is True) and (
                args.commands[0] == 'evaluate_big_ensemble_model')):
            save_all_votes = True
        else:
            save_all_votes = False
        # two (2) positions are for the positive and negative counts.
        # the votes are for each label separately.
        votes = torch.zeros((dataset_len, num_labels, 2), dtype=torch.float32)
        probs = torch.zeros((dataset_len, num_labels), dtype=torch.float32)

        if save_all_votes is True:
            model_votes = torch.zeros((nr_models, dataset_len, num_labels),
                                      dtype=torch.int64)
            target_labels = torch.zeros((dataset_len, num_labels),
                                        dtype=torch.float)
        else:
            model_votes = None
            target_labels = None

        with torch.no_grad():
            # Votes for each model seperately (Non noisy)
            for model_id in self.model_ids:
                start_time = time.time()
                model = load_private_model_by_id(args=args, id=model_id)
                model = distribute_model(args=args, model=model)
                task_outputs = {}
                task_targets = {}
                for task in range(num_labels):
                    task_outputs[task] = []
                    task_targets[task] = []
                end = 0
                for data_id, (data, target) in enumerate(dataloader):
                    batch_size = data.shape[0]
                    begin = end
                    end = begin + batch_size
                    if args.cuda:
                        data = data.cuda().to(torch.float32)

                    # Generate raw ensemble votes

                    # When we use the densenet121 models pretrained on ImageNet
                    # as the private models, their input has 3 channels whereas
                    # cxpert is originally 1 dimension (grayscale) so we repeat
                    # the single channel 3 times.
                    if (args.dataset == "cxpert") and (
                            args.use_pretrained_models == True):
                        data = data.repeat(1, 3, 1, 1)  # Addition for cxpert
                    output = model(data)
                    # print("data_id", data_id)
                    # print("data", data)
                    # print("output", output)
                    output = output.detach().cpu()

                    if args.sigmoid_op == 'apply':
                        output = torch.sigmoid(output)
                        # output = softmax(output)
                    # else:   # Changes here
                    #     assert output.min() >= 0.0 and output.max() <= 1.0

                    threshold = torch.tensor(args.multilabel_prob_threshold)
                    # if args.pick_labels is not None and (
                    #         args.pick_labels != [-1]):
                    #     threshold = pick_labels_torch(labels=threshold,
                    #                                   args=args)
                    #     output = pick_labels_torch(labels=output, args=args)
                    #     target = pick_labels_torch(labels=target, args=args)
                    preds_pos = (output > threshold)
                    preds_pos = preds_pos.to(torch.int64)
                    # print("PREDS_POS", preds_pos)

                    if save_all_votes is True:
                        # preds_pos = get_votes_only_for_dataset(
                        #     votes=preds_pos, dataset_name=args.dataset)
                        model_votes[model_id][begin:end] = preds_pos
                        target_labels[begin:end] = target

                    # tau_dep is for data dependent analysis_test with clipping.
                    # We checked that the tau clipping does not improve the privacy
                    # budget on its own. It should be incorporated into
                    # the analysis_test of PATE for the multilabel classification
                    # via clipping.
                    if (args.private_tau is not None) and (
                            args.private_tau > 0):
                        preds_pos = clip_votes_tensor(
                            votes=preds_pos, tau=args.private_tau,
                            norm=args.private_tau_norm
                        )
                    preds_neg = 1 - preds_pos
                    preds_sum = preds_pos.sum() + preds_neg.sum()
                    if not math.isclose(preds_sum, target.numel(),
                                        abs_tol=0.001):
                        raise Exception(
                            f'preds_sum: {preds_sum} is different '
                            f'than the target.numel: {target.numel()}.')

                    # TODO: supress votes and probabilities for un-trained on labels.

                    if hasattr(model, 'module'):
                        raw_model = model.module
                    else:
                        raw_model = model

                    cond_reweight = hasattr(raw_model, 'label_weights') and (
                            raw_model.label_weights is not None) and (
                                            args.label_reweight == 'apply')
                    if cond_reweight:
                        label_weights = torch.tensor(
                            raw_model.label_weights, device=preds_pos.device,
                            dtype=torch.float32)
                        # Re-weight the votes an
                        preds_pos = preds_pos * label_weights
                        preds_neg = preds_neg * label_weights

                    # Combine the positive and negative votes.
                    preds_pos_votes = preds_pos.unsqueeze(dim=-1)
                    preds_neg_votes = preds_neg.unsqueeze(dim=-1)

                    # Order matters: negative votes on the 0th position,
                    # positive votes on the 1st position (in the last dimension).
                    batch_votes = torch.cat(
                        (preds_neg_votes, preds_pos_votes), dim=-1)
                    if args.private_tau == 0:
                        # With weights or tau the preds_pos are no longer
                        # integer values.
                        assert torch.all(
                            batch_votes.argmax(dim=-1).eq(preds_pos))
                    votes[begin:end] += batch_votes
                    probs[begin:end] += output

                    for task in range(num_labels):
                        task_output = output[:, task]
                        task_target = target[:, task]
                        mask = ~torch.isnan(task_target)
                        task_output = task_output[mask]
                        task_target = task_target[mask]
                        task_output = task_output.detach().cpu().numpy()
                        task_target = task_target.detach().cpu().numpy()
                        task_outputs[task].append(task_output)
                        task_targets[task].append(task_target)
                        # print("task output", task_output)
                        # print("task target", task_target)

                for task in range(num_labels):  # per task/label/class
                    task_outputs[task] = np.concatenate(task_outputs[task])
                    task_targets[task] = np.concatenate(task_targets[task])
                # print("tasko", task_outputs)
                metrics = compute_metrics_multilabel(
                    # The metrics computed here seem to be correct
                    args=args, task_outputs=task_outputs,
                    task_targets=task_targets)

                acc = metrics[metric.acc]
                acc_detailed = metrics[metric.acc_detailed]

                # print(f'computed directly: '
                #       f'model id {id} with acc: {acc}, '
                #       f'detailed acc: {acc_detailed}')

                balanced_acc = metrics[metric.balanced_acc]
                balanced_acc_detailed = metrics[metric.balanced_acc_detailed]

                auc = metrics[metric.auc]
                auc_detailed = metrics[metric.auc_detailed]
                map = metrics[metric.map]

                elapsed_time = time.time() - start_time
                print(f'computed directly: '
                      f'model-id;{model_id};acc;{acc};balanced acc;{balanced_acc};'
                      f'auc;{auc};map;{map};time(sec);{elapsed_time}')
                # f'detailed balanced acc: {balanced_acc_detailed}')

                if sum_acc is None:
                    sum_acc = acc
                    sum_balanced_acc = balanced_acc
                    sum_auc = auc
                    sum_map = map
                    sum_acc_detailed = acc_detailed
                    sum_auc_detailed = auc_detailed
                    sum_balanced_acc_detailed = balanced_acc_detailed
                else:
                    sum_acc += acc
                    sum_balanced_acc += balanced_acc
                    sum_auc += auc
                    sum_map += map
                    sum_acc_detailed += acc_detailed
                    sum_auc_detailed += auc_detailed
                    sum_balanced_acc_detailed += balanced_acc_detailed

            avg_acc = sum_acc / nr_models
            avg_detailed_acc = sum_acc_detailed / nr_models

            avg_balanced_acc = sum_balanced_acc / nr_models
            avg_balanced_detailed_acc = sum_balanced_acc_detailed / nr_models

            avg_auc = sum_auc / nr_models
            avg_detailed_auc = sum_auc_detailed / nr_models
            avg_map = sum_map / nr_models

            print(f'ensemeble avg acc: {avg_acc},\n'
                  f'ensemeble avg balanced acc: {avg_balanced_acc},\n'
                  f'ensemeble avg auc: {avg_auc},\n'
                  f'ensemeble avg map: {avg_map},\n'
                  f'ensemble avg acc detailed: {avg_detailed_acc},\n'
                  f'ensemble avg auc detailed: {avg_detailed_auc},\n'
                  f'ensemble avg balanced acc detailed: {avg_balanced_detailed_acc}')

            # # collect all labels.
            # all_targets = get_all_targets(dataloader=dataloader, args=args)
            # acc, acc_detailed = self.get_acc_votes_targets(
            #     votes=votes, targets=all_targets)
            # print(f'from votes:\n'
            #       f'total acc: {acc},\n'
            #       f'detailed acc: {acc_detailed}')

            if save_all_votes is True:
                print("VOTES ARE SAVED")
                model_votes = model_votes.to(torch.int64).numpy()
                np.save(file=f'model_votes_{args.dataset}', arr=model_votes)

                target_labels = target_labels.numpy()
                np.save(file=f'target_labels_{args.dataset}', arr=target_labels)

            votes = votes.numpy()
            if args.private_tau == 0:
                assert np.all(votes.sum(axis=-1) == len(self.model_ids))
            probs = probs.numpy() / float(len(self.model_ids))

        return votes, probs, model_votes

    def get_votes_multilabel(self, dataloader, args) -> np.ndarray:
        votes, _, _ = self.get_votes_probs(dataloader=dataloader, args=args)
        return votes

    def get_votes_multilabel_counting(self, dataloader, args) -> np.ndarray:
        votes, probs, _ = self.get_votes_probs(dataloader=dataloader, args=args)

        if args.vote_type == 'discrete':
            probs = votes.max(axis=-1)

        return probs

    def get_votes_multilabel_tau(self, dataloader, args) -> np.ndarray:
        _, _, votes = self.get_votes_probs(dataloader=dataloader, args=args)
        return votes

    def get_votes(self, dataloader, args) -> np.ndarray:
        if args.class_type == 'multiclass':
            get_votes_method = self.get_votes_multiclass
        elif args.class_type == 'multiclass_confidence':
            get_votes_method = self.get_votes_multiclass_confidence
        elif args.class_type in ['multilabel',
                                 'multilabel_pate',
                                 'multilabel_tau_dep',
                                 'multilabel_tau_pate',
                                 ]:
            get_votes_method = self.get_votes_multilabel
        elif args.class_type in ['multilabel_counting',
                                 'multilabel_counting_gaussian',
                                 'multilabel_counting_laplace']:
            get_votes_method = self.get_votes_multilabel_counting
        elif args.class_type in ['multilabel_tau',
                                 'multilabel_tau_data_independent'
                                 # 'multilabel_tau_pate',
                                 ]:
            get_votes_method = self.get_votes_multilabel_tau
        elif args.class_type in ['multilabel_powerset']:
            get_votes_method = self.get_votes_multilabel_powerset
        else:
            raise Exception(f'Unknown args.class_type: {args.class_type}.')
        votes = get_votes_method(dataloader=dataloader, args=args)
        return votes

    def get_votes_cached(self, dataloader, args, dataset_type='',
                         party_id=None) -> np.ndarray:
        """
        The votes for the multilabel contain the positive and negative counts
        whereas the votes for the multilabel_counting contain the probability of
        a given label being present.

        The votes are also different for ensemble models that extract the votes
        from different teacher models, thus we add the self.name to the filename
        for the votes.

        :param dataset_type: is it test, train, validation, or unlabeled.
        """
        if party_id is None:
            party_id = 'no-id'

        if args.class_type == 'multilabel_counting':
            class_type = f"{args.class_type}{args.vote_type}"
        else:
            class_type = f"{args.class_type}"

        # filename = f'votes_{args.dataset}_{args.architecture}_' \
        #            f'num-models_{args.num_models}_{class_type}_{party_id}_' \
        #            f'name_{self.name}_data-type_{dataset_type}_' \
        #            f'sigma_gnmax_{args.sigma_gnmax}_' \
        #            f'sigma_threshold_{args.sigma_threshold}_' \
        #            f'threshold_{args.threshold}_'

        filename = f'votes_{args.dataset}_{args.architecture}_' \
                   f'num-models_{args.num_models}_{class_type}_{party_id}_' \
                   f'data-type_{dataset_type}_'

        if args.private_tau is not None and args.private_tau > 0:
            filename += f'-tau-{args.private_tau}'

        filename += '.npy'
        filename = filename.replace('(', '_').replace(')', '_')

        print('cached votes filename: ', filename, flush=True)
        filepath = os.path.join(args.ensemble_model_path, filename)
        if args.load_votes is True:
            augmented_print(f'filepath: {filepath}', file=args.log_file)
            if os.path.isfile(filepath):
                augmented_print(
                    f"Loading ensemble (query {args.class_type}) votes "
                    f"for {self.name} in {args.mode} mode!", args.log_file)
                votes = np.load(filepath, allow_pickle=True)
                print("LOADED VOTES")
            else:
                augmented_print(
                    f"Generating ensemble (query {args.class_type}) votes "
                    f"for {self.name} in {args.mode} mode!", args.log_file)
                votes = self.get_votes(args=args, dataloader=dataloader)
                np.save(file=filepath, arr=votes)
        else:
            votes = self.get_votes(args=args, dataloader=dataloader)
            np.save(file=filepath, arr=votes)

        print('votes shape: ', votes.shape, flush=True)

        if args.debug is True and args.class_type == 'multilabel_counting':
            self.compute_mus(votes=votes, args=args)
        return votes

    def compute_mus(self, votes, args):
        votes = pick_labels_general(labels=votes, args=args)
        tau = np.array(args.multilabel_prob_threshold)
        tau = np.expand_dims(tau, axis=0)
        tau = pick_labels_general(labels=tau, args=args).squeeze()

        mu = np.mean(votes)
        mus = np.mean(votes, axis=0)
        mus = get_value_str(mus)

        votes_low = np.copy(votes)
        votes_low[votes >= tau] = 0
        mu_low_sum = votes_low.sum()
        mu_low_counts = (votes_low != 0).sum()
        mu_low = mu_low_sum / mu_low_counts

        votes_high = np.copy(votes)
        votes_high[votes < tau] = 0
        mu_high_sum = votes_high.sum()
        mu_high_counts = (votes_high != 0).sum()
        mu_high = mu_high_sum / mu_high_counts

        assert mu_high_counts + mu_low_counts == votes.size

        print(
            f"num_models,{args.num_models},mu_low,{mu_low},mu_high,{mu_high},"
            f"mu,{mu},mus,{mus}")

    def inference(self, unlabeled_dataloader, args):
        """Generate raw ensemble votes for RDP analysis_test."""
        votes = self.get_votes(dataloader=unlabeled_dataloader, args=args)
        return votes

    def query(self, queryloader, args, indices_queried, votes_queried,
              targets=None):
        if args.class_type in ['multiclass']:
            return self.query_multiclass(
                queryloader=queryloader, args=args,
                indices_queried=indices_queried, votes=votes_queried)
        elif args.class_type in ['multiclass_confidence']:
            return self.query_multiclass_confidence(
                queryloader=queryloader, args=args,
                indices_queried=indices_queried, votes=votes_queried)
        elif args.class_type in ['multilabel',
                                 'multilabel_tau_dep',
                                 'multilabel_pate',
                                 'multilabel_tau_pate']:
            return self.query_multilabel(
                queryloader=queryloader, args=args,
                indices_queried=indices_queried, votes=votes_queried,
                targets=targets)
        elif args.class_type == 'multilabel_counting':
            return self.query_multilabel_counting(
                queryloader=queryloader, args=args,
                indices_queried=indices_queried, all_votes=votes_queried)
        elif args.class_type == 'multilabel_counting_gaussian':
            return self.query_multilabel_counting_gaussian(
                queryloader=queryloader, args=args,
                indices_queried=indices_queried, all_votes=votes_queried)
        elif args.class_type == 'multilabel_counting_laplace':
            return self.query_multilabel_counting_laplace(
                queryloader=queryloader, args=args,
                indices_queried=indices_queried, all_votes=votes_queried)
        elif args.class_type in ['multilabel_tau',
                                 'multilabel_tau_data_independent',
                                 # 'multilabel_tau_pate',
                                 ]:
            # return self.query_multilabel_tau(
            #     queryloader=queryloader, args=args, targets=targets,
            #     indices_queried=indices_queried, all_votes=votes_queried)
            return self.query_multilabel_tau_single_noise(
                queryloader=queryloader, args=args, targets=targets,
                indices_queried=indices_queried, all_votes=votes_queried)
        elif args.class_type in ['multilabel_powerset']:
            return self.query_multilabel_powerset(
                queryloader=queryloader, args=args, targets=targets,
                indices_queried=indices_queried, votes_all=votes_queried)
            # return self.query_multilabel_powerset_iterative(
            #     queryloader=queryloader, args=args, targets=targets,
            #     indices_queried=indices_queried, votes_all=votes_queried)
        else:
            raise Exception(f'Unknown args.class_type: {args.class_type}.')

    def query_multiclass(self, queryloader, args, indices_queried, votes):
        """Query a noisy ensemble model."""
        indices_queried = np.array(indices_queried)
        data_size = len(indices_queried)
        gaps_detailed = np.zeros(args.num_classes, dtype=np.float64)
        correct = np.zeros(args.num_classes, dtype=np.int64)
        wrong = np.zeros(args.num_classes, dtype=np.int64)
        # Thresholding mechanism (GNMax)
        if args.sigma_threshold > 0:
            noise_threshold = np.random.normal(
                loc=0.0, scale=args.sigma_threshold, size=data_size)
            vote_counts = votes.max(axis=-1)
            answered = (vote_counts + noise_threshold) > args.threshold
        else:
            answered = [True for _ in indices_queried]
        indices_answered = indices_queried[answered]
        # Gaussian mechanism
        assert args.sigma_gnmax > 0
        noise_gnmax = np.random.normal(0., args.sigma_gnmax, (
            data_size, self.num_classes))
        noisy_votes = votes + noise_gnmax
        preds = noisy_votes.argmax(axis=1).astype(np.int64)
        preds = preds[answered]
        # Gap between the ensemble votes of the two most probable classes.
        # Sort the votes in descending order.
        sorted_votes = np.flip(np.sort(votes, axis=1), axis=1)
        # Compute the gap between 2 votes with the largest counts.
        gaps = (sorted_votes[:, 0] - sorted_votes[:, 1])[answered]

        # Target labels
        targets = get_all_targets_numpy(dataloader=queryloader, args=args)
        targets = targets.astype(np.int64)
        targets = targets[answered]
        assert len(targets) == len(preds) == len(gaps) == len(indices_answered)
        for label, pred, gap in zip(targets, preds, gaps):
            gaps_detailed[label] += gap
            if label == pred:
                correct[label] += 1
            else:
                wrong[label] += 1
        total = correct.sum() + wrong.sum()
        assert len(indices_answered) == total
        acc = 100. * correct.sum() / total
        acc_detailed = 100. * correct / (correct + wrong)
        gaps_mean = gaps_detailed.sum() / total
        gaps_detailed = gaps_detailed / (correct + wrong)

        results = {
            result.predictions: preds,
            result.indices_answered: indices_answered,
            metric.gaps_mean: gaps_mean,
            metric.gaps_detailed: gaps_detailed,
            metric.acc: acc,
            metric.acc_detailed: acc_detailed,
            metric.balanced_acc: 'N/A',
            metric.auc: 'N/A',
            metric.map: 'N/A',
        }

        return results

    def query_multiclass_confidence(self, queryloader, args, indices_queried,
                                    votes):
        confidence_scores = votes
        votes = from_confidence_scores_to_votes(
            confidence_scores=confidence_scores, args=args)
        results = self.query_multiclass(queryloader=queryloader, args=args,
                                        indices_queried=indices_queried,
                                        votes=votes)

        one_hot_confidence_bins = get_one_hot_confidence_bins(
            args=args, confidence_scores=confidence_scores, votes=votes)

        data_size, num_bins = one_hot_confidence_bins.shape

        indices_queried = np.array(indices_queried)
        assert data_size == len(indices_queried)

        # Gaussian mechanism for the confidence scores
        assert args.sigma_gnmax_confidence > 0
        noise_gnmax = np.random.normal(0., args.sigma_gnmax_confidence, (
            data_size, num_bins))
        noisy_confidence_bins = one_hot_confidence_bins + noise_gnmax
        confidences = noisy_confidence_bins.argmax(axis=1).astype(np.int64)
        confidences = confidences[results[result.indices_answered]]
        results[result.confidence_scores] = confidences

        return results

    def query_multilabel(self, queryloader, args, indices_queried, votes,
                         targets=None):
        """Query a noisy ensemble model."""
        indices_queried = np.array(indices_queried)

        # Select only votes for the considered labels (either selected from a
        # dataset or specifically by this user.
        # votes = pick_labels_general(labels=votes, args=args)
        # print("votes", votes)
        # print("votes shape", votes.shape)
        data_size = len(indices_queried)
        num_labels = votes.shape[1]

        # Threshold mechanism.
        if args.sigma_threshold > 0:
            max_counts = votes.max(axis=-1)
            # print("max counts", max_counts)
            noise_threshold = np.random.normal(
                loc=0.0, scale=args.sigma_threshold,
                size=max_counts.shape)
            answered = (max_counts + noise_threshold) > args.threshold
            # print("answered", answered.shape)
            labels_answered = answered.astype(np.int64)
            # print("labels_answered", labels_answered)
            not_answered = np.invert(answered)
            count_answered = answered.sum()
            # print("count answered", count_answered)

            # print('number of not answered: ', not_answered.sum())
            # print('number of answered: ', answered.sum())
            # assert not_answered.sum() == 0
        else:
            # Do not use the threshold mechanism.
            labels_answered = np.ones((data_size, num_labels), dtype=bool)
            not_answered = np.zeros((data_size, num_labels), dtype=bool)
            count_answered = data_size * num_labels

        # GNMax mechanism - Gaussian based Noisy (arg)max mechanism for DP.
        if args.sigma_gnmax > 0:
            size = votes.shape
            noise_gnmax = np.random.normal(
                loc=0.0, scale=args.sigma_gnmax, size=size)
            noisy_votes = votes + noise_gnmax
        else:
            noisy_votes = votes

        preds = noisy_votes.argmax(axis=-1).astype(
            np.float)  # Gets argmax of noisy votes

        if targets is None:
            targets = get_all_targets_numpy(dataloader=queryloader, args=args)
            targets = pick_labels_general(labels=targets, args=args)

        if not_answered.sum() > 1:
            targets[not_answered] = np.nan
            preds[not_answered] = np.nan

        # balanced_acc, balanced_acc_detailed = self.get_multilabel_balanced_acc(
        #     all_targets=targets, all_preds=preds)
        metrics = compute_metrics_multilabel_from_preds_targets(
            targets=targets, preds=preds, args=args)

        # Gap between the ensemble votes of the two most probable classes.
        # Sort the votes in ascending order.
        sorted_votes = np.sort(votes, axis=-1)
        # Compute the gap between 2 votes with the largest counts.
        gaps = sorted_votes[:, :, -1] - sorted_votes[:, :, -2]
        assert (gaps >= 0).all()  # some might be np.nan

        gaps_mean = gaps.mean()
        # print("gaps mean", gaps_mean)
        gaps_detailed = gaps.mean(axis=-1)
        # print("gaps detailed", gaps_detailed)
        # print("gaps detailed: ",
        #       ",".join([str(x) for x in gaps.flatten()]))

        # If any label for a given query is answered then the whole query
        # is answered.
        is_query_answered = np.any(labels_answered, axis=1)
        indices_answered = indices_queried[is_query_answered]

        results = {
            result.predictions: preds,
            result.indices_answered: indices_answered,
            result.labels_answered: labels_answered,
            result.count_answered: count_answered,
            metric.gaps_mean: gaps_mean,
            metric.gaps_detailed: gaps_detailed,
            metric.acc: metrics[metric.acc],
            metric.acc_detailed: metrics[metric.acc_detailed],
            metric.balanced_acc: metrics[metric.balanced_acc],
            metric.balanced_acc_detailed: metrics[metric.balanced_acc_detailed],
            metric.auc: metrics[metric.auc],
            metric.auc_detailed: metrics[metric.auc_detailed],
            metric.map: metrics[metric.map],
            metric.map_detailed: metrics[metric.map_detailed],
        }

        return results

    def compute_tau_counting(self, votes, targets, args):
        votes = pick_labels_general(labels=votes, args=args)
        targets = pick_labels_general(labels=targets, args=args)

        print('tau,balanced accuracy')
        ba_max = 0.0
        tau_best = 0.0
        for tau in np.linspace(0, 1, 100):
            preds = np.copy(votes)
            preds[preds > tau] = 1
            preds[preds <= tau] = 0
            balanced_acc, _ = self.get_multilabel_balanced_acc(
                all_preds=preds, all_targets=targets)
            # print(tau, ',', balanced_acc)
            if balanced_acc > ba_max:
                tau_best = tau
                ba_max = balanced_acc

        print('globally best tau,', tau_best,
              ',highest balanced accuracy global,', ba_max)
        num_labels = votes.shape[1]
        print('label index,best tau,highest balanced accuracy per label')
        taus = []
        for label in range(num_labels):
            votes_label = votes[:, label]
            targets_label = targets[:, label]
            no_nans = ~np.isnan(targets_label)
            votes_label = votes_label[no_nans]
            targets_label = targets_label[no_nans]
            ba_max = 0.0
            tau_best = 0.0
            for tau in np.linspace(0, 1, 100):
                preds = np.copy(votes_label)
                preds[preds > tau] = 1
                preds[preds <= tau] = 0
                balanced_acc = metrics.balanced_accuracy_score(
                    y_true=targets_label, y_pred=preds)
                # print(label, ',', tau, ',', balanced_acc)
                if balanced_acc > ba_max:
                    tau_best = tau
                    ba_max = balanced_acc
            print(label, ',', tau_best, ',', ba_max)
            taus.append(tau_best)
        print('per label best taus: ', taus)
        return taus

    def query_multilabel_counting(self, queryloader, args, indices_queried,
                                  all_votes):
        """Query a noisy ensemble model."""
        indices_queried = np.array(indices_queried)
        data_size_all = len(indices_queried)
        all_targets = get_all_targets_numpy(dataloader=queryloader, args=args)
        if args.debug == True:
            self.compute_tau_counting(votes=all_votes, targets=all_targets,
                                      args=args)
        np.save('query_multilabel_counting_targets.npy', all_targets)
        # print('number of queries,accuracy')
        # print('universal constant,balanced accuracy')
        print('data size, universal constant, balanced accuracy')
        # data_size = data_size_all
        # data_size = 3000
        # data_size = 50
        universal_constants = [0.0]
        # for data_size in range(data_size_all, data_size_all + 1):
        # for data_size in [1000]:
        for data_size in [data_size_all]:
            # universal_constants = np.linspace(start=1.0, stop=0.0, num=10)
            # universal_constants = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
            #                        0.01]
            # universal_constants = [0.2, 0.1, 0.01]
            for universal_constant in universal_constants:
                # Bounded Noise Mechanism
                current_targets = all_targets[:data_size]
                current_votes = all_votes[:data_size]
                epsilon = args.budget / data_size
                num_labels = current_votes.shape[1]
                # delta = np.exp(-num_labels / (np.log(num_labels) ** 8))
                delta = args.delta

                if universal_constant > 0:
                    noise_matrix, noise_bound = sample_bounded_noise(
                        epsilon=epsilon, delta=delta, num_labels=num_labels,
                        universal_constant=universal_constant,
                        noise_type=args.count_noise, shape=current_votes.shape,
                        num_users=args.num_models)

                    noisy_votes = current_votes + noise_matrix
                else:
                    noisy_votes = current_votes

                threshold = np.array(args.multilabel_prob_threshold)

                compute_R = False
                if compute_R and args.dataset in args.xray_datasets:
                    indexes = get_indexes(dataset=args.dataset)
                    valid_threshold = threshold[indexes]
                    valid_votes = pick_labels_cols(
                        target_labels_index=indexes, labels=current_votes)
                    R = np.abs(valid_votes - valid_threshold)
                    print(
                        f"mean R: {np.mean(R)}, min R: {np.min(R)}, "
                        f"median R: {np.median(R)}, max R: {np.max(R)}")
                all_preds = np.array(noisy_votes > threshold)
                # all_preds = current_votes > args.multilabel_prob_threshold
                # balanced_acc, balanced_acc_detailed = self.get_multilabel_balanced_acc(
                #     all_targets=current_targets, all_preds=all_preds)
                metrics = compute_metrics_multilabel_from_preds_targets(
                    targets=current_targets, preds=all_preds, args=args)
                # print(data_size, ',', universal_constant, ',', balanced_acc)
                # print(universal_constant, ',', balanced_acc)

        indices_answered = np.arange(0, data_size, 1)

        results = {
            result.predictions: all_preds,
            result.indices_answered: indices_answered,
            metric.gaps_mean: None,
            metric.gaps_detailed: None,
            metric.acc: metrics[metric.acc],
            metric.acc_detailed: metrics[metric.acc_detailed],
            metric.balanced_acc: metrics[metric.balanced_acc],
            metric.balanced_acc_detailed: metrics[metric.balanced_acc_detailed],
            metric.auc: metrics[metric.auc],
            metric.auc_detailed: metrics[metric.auc_detailed],
            metric.map: metrics[metric.map],
            metric.map_detailed: metrics[metric.map_detailed],
        }

        return results

    def query_multilabel_counting_gaussian(
            self, queryloader, args, indices_queried, all_votes):
        """Query a noisy ensemble model."""
        indices_queried = np.array(indices_queried)
        data_size_all = len(indices_queried)
        num_labels = args.num_classes
        all_targets = get_all_targets_numpy(dataloader=queryloader, args=args)
        np.save('query_multilabel_counting_gaussian_targets.npy', all_targets)
        # print('number of queries,accuracy')
        print('universal constant,balanced accuracy')
        # data_size = data_size_all
        # data_size = 3000
        data_size = 50
        universal_constants = [1.0]
        # for data_size in range(1, data_size_all + 1, 10):
        # for data_size in [1000]:
        # for data_size in [data_size_all]:
        # universal_constants = np.linspace(start=1.0, stop=0.0, num=10)
        # universal_constants = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
        #                        0.01]
        # universal_constants = [0.2, 0.1, 0.01]
        for universal_constant in universal_constants:
            # Bounded Noise Mechanism
            current_targets = all_targets[:data_size]
            current_votes = all_votes[:data_size]
            epsilon = args.budget / data_size
            num_labels = current_votes.shape[1]
            # delta = np.exp(-num_labels / (np.log(num_labels) ** 8))
            delta = args.delta
            noise_matrix, sigma = sample_gaussian_noise(
                epsilon=epsilon, delta=delta, num_labels=num_labels,
                universal_constant=universal_constant,
                shape=current_votes.shape, num_users=args.num_models)
            noisy_votes = current_votes + noise_matrix
            all_preds = np.array(noisy_votes > args.multilabel_prob_threshold)
            # all_preds = current_votes > args.multilabel_prob_threshold
            balanced_acc, balanced_acc_detailed = self.get_multilabel_balanced_acc(
                all_targets=current_targets, all_preds=all_preds)
            # print(data_size, ',', balanced_acc)
            print(universal_constant, ',', balanced_acc)

        indices_answered = np.arange(0, data_size, 1)

        results = {
            result.predictions: all_preds,
            result.indices_answered: indices_answered,
            metric.gaps_mean: None,
            metric.gaps_detailed: None,
            metric.balanced_acc: balanced_acc,
            metric.balanced_acc_detailed: balanced_acc_detailed
        }

        return results

    def query_multilabel_counting_laplace(self, queryloader, args,
                                          indices_queried,
                                          all_votes):
        """Query a noisy ensemble model."""
        indices_queried = np.array(indices_queried)
        data_size_all = len(indices_queried)
        num_labels = args.num_classes
        all_targets = get_all_targets(dataloader=queryloader).numpy()
        np.save('query_multilabel_counting_laplace_targets.npy', all_targets)
        # print('number of queries,accuracy')
        print('universal constant,balanced accuracy')
        # data_size = data_size_all
        # data_size = 3000
        data_size = 50
        universal_constants = [1.0, 0.5]
        # for data_size in range(1, data_size_all + 1, 10):
        # for data_size in [1000]:
        # for data_size in [data_size_all]:
        # universal_constants = np.linspace(start=1.0, stop=0.0, num=10)
        # universal_constants = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,
        #                        0.01]
        # universal_constants = [0.2, 0.1, 0.01]
        for universal_constant in universal_constants:
            # Bounded Noise Mechanism
            current_targets = all_targets[:data_size]
            current_votes = all_votes[:data_size]
            epsilon = args.budget / data_size
            num_labels = current_votes.shape[1]
            # delta = np.exp(-num_labels / (np.log(num_labels) ** 8))
            delta = args.delta
            noise_matrix, sigma = sample_laplace_noise(
                epsilon=epsilon, delta=delta, num_labels=num_labels,
                universal_constant=universal_constant,
                shape=current_votes.shape, num_users=args.num_models)
            noisy_votes = current_votes + noise_matrix
            all_preds = np.array(noisy_votes > args.multilabel_prob_threshold)
            # all_preds = current_votes > args.multilabel_prob_threshold
            balanced_acc, balanced_acc_detailed = self.get_multilabel_balanced_acc(
                all_targets=current_targets, all_preds=all_preds)
            # print(data_size, ',', balanced_acc)
            print(universal_constant, ',', balanced_acc)

        gaps_mean, gaps_detailed = None, None
        indices_answered = np.arange(0, data_size, 1)

        results = {
            result.predictions: all_preds,
            result.indices_answered: indices_answered,
            metric.gaps_mean: None,
            metric.gaps_detailed: None,
            metric.balanced_acc: balanced_acc,
            metric.balanced_acc_detailed: balanced_acc_detailed
        }

        return results

    def query_multilabel_tau_single_noise(
            self, queryloader, args, indices_queried, all_votes, targets=None):
        """Query a noisy ensemble model."""
        indices_queried = np.array(indices_queried)
        sigma = args.sigma_gnmax

        preds_pos = np.copy(all_votes)
        # Add noise.
        if sigma > 0:
            num_votes = preds_pos.shape[0]
            num_labels = preds_pos.shape[1]
            preds_pos += sigma * np.random.randn(num_votes, num_labels)

        preds = (2 * preds_pos > args.num_models).astype(np.int)

        if targets is None:
            targets = get_all_targets_numpy(dataloader=queryloader, args=args)

        # balanced_acc, balanced_acc_detailed = self.get_multilabel_balanced_acc(
        #     all_targets=targets, all_preds=preds)
        metrics = compute_metrics_multilabel_from_preds_targets(
            targets=targets, preds=preds, args=args)

        # Gap between the ensemble votes of the two most probable classes.
        gaps = np.abs(2 * preds_pos - args.num_models)
        assert (gaps >= 0).all()  # some might be np.nan

        gaps_mean = gaps.mean()
        gaps_detailed = gaps.mean(axis=-1)
        # print("gaps detailed: ",
        #       ",".join([str(x) for x in gaps.flatten()]))

        indices_answered = indices_queried
        labels_answered = np.ones_like(all_votes)
        count_answered = np.sum(labels_answered)

        results = {
            result.predictions: preds,
            result.indices_answered: indices_answered,
            result.labels_answered: labels_answered,
            result.count_answered: count_answered,
            metric.gaps_mean: gaps_mean,
            metric.gaps_detailed: gaps_detailed,
            metric.acc: metrics[metric.acc],
            metric.acc_detailed: metrics[metric.acc_detailed],
            metric.balanced_acc: metrics[metric.balanced_acc],
            metric.balanced_acc_detailed: metrics[metric.balanced_acc_detailed],
            metric.auc: metrics[metric.auc],
            metric.auc_detailed: metrics[metric.auc_detailed],
            metric.map: metrics[metric.map],
            metric.map_detailed: metrics[metric.map_detailed],
        }

        return results

    def query_multilabel_tau(self, queryloader, args, indices_queried,
                             all_votes, targets=None):
        """Query a noisy ensemble model."""
        indices_queried = np.array(indices_queried)
        sigma = args.sigma_gnmax

        preds_pos = np.copy(all_votes)
        # Create the corresponding negative predictions.
        preds_neg = args.num_models * np.ones_like(preds_pos) - preds_pos

        # Add noise.
        if sigma > 0:
            num_votes = preds_pos.shape[0]
            num_labels = preds_pos.shape[1]
            preds_pos += sigma * np.random.randn(num_votes, num_labels)
            preds_neg += sigma * np.random.randn(num_votes, num_labels)

        preds = (preds_pos > preds_neg).astype(np.int)

        if targets is None:
            targets = get_all_targets_numpy(dataloader=queryloader, args=args)

        # balanced_acc, balanced_acc_detailed = self.get_multilabel_balanced_acc(
        #     all_targets=targets, all_preds=preds)
        metrics = compute_metrics_multilabel_from_preds_targets(
            targets=targets, preds=preds, args=args)

        # Gap between the ensemble votes of the two most probable classes.
        gaps = np.abs(preds_pos - preds_neg)
        assert (gaps >= 0).all()  # some might be np.nan

        gaps_mean = gaps.mean()
        gaps_detailed = gaps.mean(axis=-1)
        # print("gaps detailed: ",
        #       ",".join([str(x) for x in gaps.flatten()]))

        indices_answered = indices_queried
        labels_answered = np.ones_like(all_votes)
        count_answered = np.sum(labels_answered)

        results = {
            result.predictions: preds,
            result.indices_answered: indices_answered,
            result.labels_answered: labels_answered,
            result.count_answered: count_answered,
            metric.gaps_mean: gaps_mean,
            metric.gaps_detailed: gaps_detailed,
            metric.acc: metrics[metric.acc],
            metric.acc_detailed: metrics[metric.acc_detailed],
            metric.balanced_acc: metrics[metric.balanced_acc],
            metric.balanced_acc_detailed: metrics[metric.balanced_acc_detailed],
            metric.auc: metrics[metric.auc],
            metric.auc_detailed: metrics[metric.auc_detailed],
            metric.map: metrics[metric.map],
            metric.map_detailed: metrics[metric.map_detailed],
        }

        return results

    def get_multilabel_balanced_acc(
            self, all_targets: np.ndarray, all_preds: np.ndarray) -> (
            np.float, np.ndarray):
        balanced_acc_detailed = []
        num_labels = all_targets.shape[1]
        for task in range(num_labels):
            task_targets = all_targets[:, task]
            task_preds = all_preds[:, task]
            no_nans = ~np.isnan(task_targets)
            task_targets = task_targets[no_nans].astype(np.float)
            task_preds = task_preds[no_nans].astype(np.float)
            assert not np.any(np.isnan(task_targets))
            if len(task_targets) > 0:
                balanced_acc = metrics.balanced_accuracy_score(
                    y_true=task_targets, y_pred=task_preds)
                balanced_acc_detailed.append(balanced_acc)
            else:
                balanced_acc_detailed.append(np.nan)

        balanced_acc_detailed = np.array(balanced_acc_detailed)
        index = ~np.isnan(balanced_acc_detailed)
        detailed = balanced_acc_detailed[index]
        if len(detailed) > 0:
            balanced_acc = np.mean(detailed)
        else:
            balanced_acc = 'N/A'
        return balanced_acc, balanced_acc_detailed

    def get_multilabel_balanced_acc_from_votes(
            self, votes: np.ndarray, targets: np.ndarray, args):
        all_preds = self.get_preds(
            votes=votes, class_type=args.class_type,
            threshold=args.multilabel_prob_threshold)
        return self.get_multilabel_balanced_acc(
            all_preds=all_preds, all_targets=targets)

    def get_multilabel_accuracy(self, preds, targets):
        if np.any(preds == np.nan):
            not_answered = preds == np.nan
            preds = preds.astype(np.int64)
            preds[not_answered] = -1
        else:
            preds = preds.astype(np.int64)

        targets = targets.astype(np.int64)

        num_labels = targets.shape[1]
        correct = np.zeros(num_labels, dtype=np.int64)
        wrong = np.zeros(num_labels, dtype=np.int64)

        for label, pred in zip(targets, preds):
            # Iterate through all the labels for a given data sample.
            for j in range(len(label)):
                if pred[j] == -1:
                    # Skip the unanswered query.
                    continue
                if label[j] == pred[j]:
                    correct[j] += 1
                else:
                    wrong[j] += 1
        total = correct.sum() + wrong.sum()

        acc = 100. * correct.sum() / total
        total_per_label = correct + wrong
        detailed_acc = 100. * correct / total_per_label

        return acc, detailed_acc

    def get_votes_multilabel_powerset(self, dataloader, args) -> np.ndarray:
        """
        Get votes for the multilabel powerset method.

        :param dataloader: data loader
        :param args: program params
        :return: array of dicts, with dict per sample that for each class
        returns the count of the multilabel class.
        """
        _, _, model_votes = self.get_votes_probs(dataloader=dataloader,
                                                 args=args, save_all_votes=True)
        model_votes = np.swapaxes(model_votes, 0, 1)
        return model_votes

    def query_multilabel_powerset(self, queryloader, args, indices_queried,
                                  votes_all, targets=None):
        """Query a noisy ensemble model."""
        indices_queried = np.array(indices_queried)
        data_size = len(indices_queried)
        num_labels = votes_all.shape[2]
        votes_all = votes_all[indices_queried]
        vote_count, map_idx_label = get_vote_count_and_map_powerset(
            args=args, votes_all=votes_all)

        # Threshold mechanism.
        if args.sigma_threshold > 0:
            max_counts = vote_count.max(axis=-1)
            # print("max counts", max_counts)
            noise_threshold = np.random.normal(
                loc=0.0, scale=args.sigma_threshold, size=max_counts.shape)
            answered = (max_counts + noise_threshold) > args.threshold
            # print("answered", answered.shape)
            labels_answered = answered.astype(np.int64)
            # print("labels_answered", labels_answered)
            not_answered = np.invert(answered)
            count_answered = answered.sum()
            # print("count answered", count_answered)

            # print('number of not answered: ', not_answered.sum())
            # print('number of answered: ', answered.sum())
            # assert not_answered.sum() == 0
        else:
            # Do not use the threshold mechanism.
            labels_answered = np.ones((data_size, num_labels), dtype=bool)
            not_answered = np.zeros((data_size, num_labels), dtype=bool)
            count_answered = data_size * num_labels

        if targets is None:
            targets = get_all_targets_numpy(dataloader=queryloader,
                                            args=args)
            targets = pick_labels_general(labels=targets, args=args)

        are_gaps_computed = False

        if are_gaps_computed:
            # Gap between the ensemble votes of the two most probable classes.
            # Sort the votes in ascending order.
            sorted_votes = np.sort(vote_count, axis=-1)
            # Compute the gap between 2 votes with the largest counts.
            gaps = sorted_votes[:, -1] - sorted_votes[:, -2]
            assert (gaps >= 0).all()  # some might be np.nan
            gaps_mean = np.mean(gaps)
            gaps_detailed = np.mean(gaps, axis=-1)
            # print("gaps detailed: ", ",".join([str(x) for x in gaps.flatten()]))
        else:
            gaps_mean = None
            gaps_detailed = None

        # file_name = (
        #     f"evaluate_big_ensemble_{args.class_type}_seaborn_"
        #     f"dataset_{args.dataset}_"
        #     f"_private_tau_{args.private_tau}_"
        #     f"labels_{num_labels}_"
        #     f"{args.timestamp}.txt")

        file_name = f'labels_{args.class_type}_{args.dataset}_{num_labels}_labels.csv'

        preds = None
        metrics = None

        if args.command == 'evaluate_big_ensemble_model':
            sigma_gnmaxs = [
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
                18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 50, 55, 60]
        else:
            sigma_gnmaxs = [args.sigma_gnmax]
        print('Check sigma_gnmaxs: ')
        is_header = False
        for sigma_gnmax in sigma_gnmaxs:
            print('sigma_gnmax: ', sigma_gnmax)
            args.sigma_gnmax = sigma_gnmax
            num_samples, num_labels = vote_count.shape
            if num_labels <= 2 ** 20:
                # GNMax mechanism - Gaussian based Noisy (arg)max mechanism for DP.
                if args.sigma_gnmax > 0:
                    size = vote_count.shape
                    print('Generate noise: ')
                    start = time.time()
                    noise_gnmax = np.random.normal(
                        loc=0.0, scale=args.sigma_gnmax, size=size)
                    stop = time.time()
                    print(f"Elapsed time: {stop - start} (sec).")
                    noisy_votes = vote_count + noise_gnmax
                else:
                    noisy_votes = vote_count

                print('Find argmax: ')
                start = time.time()
                pred_indexes = noisy_votes.argmax(axis=-1)
                stop = time.time()
                print(f"Elapsed time: {stop - start} (sec).")
            else:
                pred_indexes = []
                for sample_idx in range(num_samples):
                    votes = vote_count[sample_idx]
                    if args.sigma_gnmax > 0:
                        noise_gnmax = np.random.normal(
                            loc=0.0, scale=args.sigma_gnmax, size=num_labels)
                        noisy_vote = votes + noise_gnmax
                    else:
                        noisy_vote = votes
                    pred_indexes.append(noisy_vote.argmax())

            preds = []
            for pred_idx in pred_indexes:
                # Map to a binary string that represents all the labels assigned
                # to a query.
                pred = map_idx_label[pred_idx]
                # map to binary vector
                pred = from_str(a=pred)
                preds.append(pred)
            preds = np.array(preds)

            if not_answered.sum() > 1:
                targets[not_answered] = np.nan
                preds[not_answered] = np.nan

            # balanced_acc, balanced_acc_detailed = self.get_multilabel_balanced_acc(
            #     all_targets=targets, all_preds=preds)
            # preds = pick_labels_general(labels=preds, args=args)

            metrics = compute_metrics_multilabel_from_preds_targets(
                targets=targets, preds=preds, args=args)

            with open(file_name, "a") as writer:
                if is_header is False:
                    is_header = True
                    writer.write('sigma,metric,value\n')

                writer.write(
                    f"{args.sigma_gnmax},ACC,{metrics[metric.acc]}\n")
                writer.write(
                    f"{args.sigma_gnmax},AUC,{metrics[metric.auc]}\n")
                writer.write(
                    f"{args.sigma_gnmax},MAP,{metrics[metric.map]}\n")

        # If any label for a given query is answered then the whole query
        # is answered.
        is_query_answered = np.any(labels_answered, axis=1)
        indices_answered = indices_queried[is_query_answered]

        if preds is not None and metrics is not None:
            results = {
                result.predictions: preds,
                result.indices_answered: indices_answered,
                result.labels_answered: labels_answered,
                result.count_answered: count_answered,
                metric.gaps_mean: gaps_mean,
                metric.gaps_detailed: gaps_detailed,
                metric.acc: metrics[metric.acc],
                metric.acc_detailed: metrics[metric.acc_detailed],
                metric.balanced_acc: metrics[metric.balanced_acc],
                metric.balanced_acc_detailed: metrics[
                    metric.balanced_acc_detailed],
                metric.auc: metrics[metric.auc],
                metric.auc_detailed: metrics[metric.auc_detailed],
                metric.map: metrics[metric.map],
                metric.map_detailed: metrics[metric.map_detailed],
            }
        else:
            results = None

        return results

    def query_multilabel_powerset_iterative(
            self, queryloader, args, indices_queried, votes_all, targets=None):
        """
        Query a noisy ensemble model. Do it iteratively with one sample at
        a time.
        """
        indices_queried = np.array(indices_queried)
        data_size = len(indices_queried)
        num_samples, num_models, num_labels = votes_all.shape

        class_labels, map_idx_label = get_class_labels_and_map_powerset(
            args=args, num_labels=num_labels)

        # Threshold mechanism.
        if args.sigma_threshold > 0:
            raise Exception(
                f"Unsupported sigma_threshold for iterative powerset.")
            # max_counts = vote_count.max(axis=-1)
            # # print("max counts", max_counts)
            # noise_threshold = np.random.normal(
            #     loc=0.0, scale=args.sigma_threshold, size=max_counts.shape)
            # answered = (max_counts + noise_threshold) > args.threshold
            # # print("answered", answered.shape)
            # labels_answered = answered.astype(np.int64)
            # # print("labels_answered", labels_answered)
            # not_answered = np.invert(answered)
            # count_answered = answered.sum()
            # print("count answered", count_answered)

            # print('number of not answered: ', not_answered.sum())
            # print('number of answered: ', answered.sum())
            # assert not_answered.sum() == 0
        else:
            # Do not use the threshold mechanism.
            labels_answered = np.ones((data_size, num_labels), dtype=bool)
            not_answered = np.zeros((data_size, num_labels), dtype=bool)
            count_answered = data_size * num_labels

        if targets is None:
            targets = get_all_targets_numpy(dataloader=queryloader,
                                            args=args)
            targets = pick_labels_general(labels=targets, args=args)

        gaps_mean = None
        gaps_detailed = None

        file_name = (
            f"evaluate_big_ensemble_{args.class_type}_seaborn_"
            f"dataset_{args.dataset}_"
            f"_private_tau_{args.private_tau}_"
            f"labels_{num_labels}_"
            f"{args.timestamp}.txt")

        preds = None
        metrics = None

        # sigma_gnmaxs = [args.sigma_gnmax]
        sigma_gnmaxs = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
            18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
            33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 50, 55, 60]
        print('Check sigma_gnmaxs: ')
        for sigma_gnmax in tqdm(sigma_gnmaxs):
            print('sigma_gnmax: ', sigma_gnmax)
            args.sigma_gnmax = sigma_gnmax
            pred_indexes = []
            for sample_idx in range(num_samples):
                votes = votes_all[sample_idx]
                votes = generate_histogram_powerset(input_votes=votes,
                                                    class_labels=class_labels)
                if args.sigma_gnmax > 0:
                    noise_gnmax = np.random.normal(
                        loc=0.0, scale=args.sigma_gnmax, size=num_labels)
                    noisy_votes = votes + noise_gnmax
                else:
                    noisy_votes = votes
                pred_indexes.append(noisy_votes.argmax())

            preds = []
            for pred_idx in pred_indexes:
                # Map to a binary string that represents all the labels assigned
                # to a query.
                pred = map_idx_label[pred_idx]
                # map to binary vector
                pred = from_str(a=pred)
                preds.append(pred)
            preds = np.array(preds)

            if not_answered.sum() > 1:
                targets[not_answered] = np.nan
                preds[not_answered] = np.nan

            # balanced_acc, balanced_acc_detailed = self.get_multilabel_balanced_acc(
            #     all_targets=targets, all_preds=preds)
            # preds = pick_labels_general(labels=preds, args=args)

            metrics = compute_metrics_multilabel_from_preds_targets(
                targets=targets, preds=preds, args=args)

            with open(file_name, "a") as writer:
                writer.write(
                    f"{args.sigma_gnmax},ACC,{metrics[metric.acc]}\n")
                writer.write(
                    f"{args.sigma_gnmax},AUC,{metrics[metric.auc]}\n")
                writer.write(
                    f"{args.sigma_gnmax},MAP,{metrics[metric.map]}\n")

        # If any label for a given query is answered then the whole query
        # is answered.
        is_query_answered = np.any(labels_answered, axis=1)
        indices_answered = indices_queried[is_query_answered]

        if preds is not None and metrics is not None:
            results = {
                result.predictions: preds,
                result.indices_answered: indices_answered,
                result.labels_answered: labels_answered,
                result.count_answered: count_answered,
                metric.gaps_mean: gaps_mean,
                metric.gaps_detailed: gaps_detailed,
                metric.acc: metrics[metric.acc],
                metric.acc_detailed: metrics[metric.acc_detailed],
                metric.balanced_acc: metrics[metric.balanced_acc],
                metric.balanced_acc_detailed: metrics[
                    metric.balanced_acc_detailed],
                metric.auc: metrics[metric.auc],
                metric.auc_detailed: metrics[metric.auc_detailed],
                metric.map: metrics[metric.map],
                metric.map_detailed: metrics[metric.map_detailed],
            }
        else:
            results = None

        return results
