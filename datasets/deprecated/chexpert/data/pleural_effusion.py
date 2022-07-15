import numpy as np
from torch.utils.data import Dataset
import cv2
import os
from PIL import Image
from datasets.deprecated.chexpert.data.imgaug import GetTransforms
from datasets.deprecated.chexpert.data.data_utils import transform

np.random.seed(0)


class PleuralEffusionDataset(Dataset):
    def __init__(self, in_csv_path, cfg, mode='train'):
        self.cfg = cfg
        self._label_header = None
        self._image_paths = []
        self._labels = []
        self._mode = mode
        # self.dict = {'1.0': 1, '': 0, '-1.0': 2, '0.0': 3}
        self.dict = cfg.class_mapping
        with open(cfg.data_path + in_csv_path) as f:
            header = f.readline().strip('\n').split(',')
            assert header[15] == 'Pleural Effusion'
            self._label_header = [header[15]]
            for line in f:
                labels = []
                fields = line.strip('\n').split(',')
                image_path = fields[0]

                # get full path to the image
                data_path_split = cfg.data_path.split('/')
                image_path_split = image_path.split('/')
                data_path_split = [x for x in data_path_split if x != '']
                image_path_split = [x for x in image_path_split if x != '']
                if data_path_split[-1] == image_path_split[0]:
                    full_image_path = data_path_split[:-1] + image_path_split
                    full_image_path = "/" + "/".join(full_image_path)
                else:
                    full_image_path = cfg.data_path + image_path

                self._image_paths.append(full_image_path)
                assert os.path.exists(full_image_path), full_image_path

                # get the label
                labels.append(self.dict.get(fields[15]))
                self._labels.append(labels)

        self._num_image = len(self._image_paths)

    def sample_counts_per_class(self):
        if not hasattr(self, '_sample_counts_per_class'):
            self._sample_counts_per_class = self._get_sample_counts_per_class()
        return self._sample_counts_per_class

    def _get_sample_counts_per_class(self):
        labels = np.array(self._labels).reshape(-1)
        uniques = np.unique(labels)
        counts = {u: 0 for u in uniques}
        for u in labels:
            counts[u] += 1
        return counts

    def __len__(self):
        return self._num_image

    def __getitem__(self, idx):
        image = cv2.imread(self._image_paths[idx], 0)
        image = Image.fromarray(image)
        if self._mode == 'train':
            image = GetTransforms(image, type=self.cfg.use_transforms_type)
        image = np.array(image)
        image = transform(image, self.cfg)
        label = np.array(self._labels[idx]).astype(np.float32)

        path = self._image_paths[idx]

        if self._mode == 'train' or self._mode == 'dev':
            return (image, label)
        elif self._mode == 'test':
            return (image, path)
        elif self._mode == 'heatmap':
            return (image, path, label)
        else:
            raise Exception('Unknown mode : {}'.format(self._mode))


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from utils import get_cfg

    cfg = get_cfg('../config/pleural_effusion_small.json')
    train_dataset = PleuralEffusionDataset(
        in_csv_path=cfg.train_csv, cfg=cfg,
        mode='train')
    dataloader_train = DataLoader(
        train_dataset,
        batch_size=40,
        num_workers=1,
        drop_last=True, shuffle=False)
    steps = len(dataloader_train)
    dataiter = iter(dataloader_train)
    targets = []
    for step in range(3):
        image, target = next(dataiter)
        print('image shape: ', image.shape, image.device, image.dtype)
        print('target: ', target, target.device, target.dtype, target.shape)
        targets += list(target.squeeze().squeeze().numpy())

    len_targets = len(targets)
    print('targets len: ', len_targets)
    targets = np.array(targets)
    uniques = np.unique(targets)
    print('uniques len: ', len(uniques))
    print('uniques: ', uniques)
    counts = {u: 0 for u in uniques}
    for u in targets:
        counts[u] += 1
    print('counts: ', counts)
    print('count values: ', counts.values())
    sum_counts = sum(counts.values())
    print('sum_counts: ', sum_counts)
    assert sum_counts == len_targets

    # counts = count_samples_per_class(dataloader=dataloader_train)
    counts = train_dataset.sample_counts_per_class()
    print('counts on the whole dataset: ', counts)

    from utils.misc import class_wise_loss_reweighting

    samples_per_cls = list(counts.values())
    weights = class_wise_loss_reweighting(
        beta=0.9999, samples_per_cls=samples_per_cls)
    print('weights: ', weights)
