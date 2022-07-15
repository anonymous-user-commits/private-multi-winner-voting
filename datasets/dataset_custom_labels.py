from torch.utils.data import Dataset


class DatasetLabels(Dataset):
    r"""
    Subset of a dataset at specified indices and with specific labels.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels
        self.correct = 0
        self.total = 0

    def __getitem__(self, idx):
        data, raw_label = self.dataset[idx]
        label = self.labels[idx]
        # print('labels: ', label, raw_label)
        if raw_label == label:
            self.correct += 1
        self.total += 1
        return data, label

    def __len__(self):
        return len(self.labels)
