import numpy as np
from torch.utils.data import Dataset
import cv2
import os
from PIL import Image
from datasets.deprecated.chexpert.data.imgaug import GetTransforms
from datasets.deprecated.chexpert.data.data_utils import transform

np.random.seed(0)


class PosDiseaseDataset(Dataset):
    def __init__(self, in_csv_path, cfg, mode='train'):
        self.cfg = cfg
        self._label_header = None
        self._image_paths = []
        self._labels = []
        self._mode = mode
        self.dict = [{'1.0': '1', '': '0', '0.0': '0', '-1.0': '0'},
                     {'1.0': '1', '': '0', '0.0': '0', '-1.0': '1'}, ]
        with open(cfg.data_path + in_csv_path) as f:
            header = f.readline().strip('\n').split(',')
            self._label_header = [
                header[7],
                header[10],
                header[11],
                header[13],
                header[15]]
            cfg._label_header = self._label_header
            for line in f:
                labels = []
                fields = line.strip('\n').split(',')
                image_path = fields[0]
                flg_enhance = False
                is_disease = 0
                for index, value in enumerate(fields[5:]):
                    if index == 5 or index == 8:
                        label = self.dict[1].get(value)
                        if label == '1':
                            is_disease = 1
                            if self.cfg.enhance_index.count(index) > 0:
                                flg_enhance = True
                    elif index == 2 or index == 6 or index == 10:
                        label = self.dict[0].get(value)
                        if label == '1':
                            is_disease = 1
                            if self.cfg.enhance_index.count(index) > 0:
                                flg_enhance = True

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
                self._labels.append(is_disease)

                if flg_enhance and self._mode == 'train':
                    for i in range(self.cfg.enhance_times):
                        self._image_paths.append(full_image_path)
                        self._labels.append(is_disease)
        self._num_image = len(self._image_paths)

    def __len__(self):
        return self._num_image

    def __getitem__(self, idx):
        image = cv2.imread(self._image_paths[idx], 0)
        image = Image.fromarray(image)
        if self._mode == 'train':
            image = GetTransforms(image, type=self.cfg.use_transforms_type)
        image = np.array(image)
        image = transform(image, self.cfg)
        # labels = np.array(self._labels[idx]).astype(np.float32)
        labels = float(self._labels[idx])

        path = self._image_paths[idx]

        if self._mode == 'train' or self._mode == 'dev':
            return (image, labels)
        elif self._mode == 'test':
            return (image, path)
        elif self._mode == 'heatmap':
            return (image, path, labels)
        else:
            raise Exception('Unknown mode : {}'.format(self._mode))


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from utils import get_cfg
    from utils import class_wise_loss_reweighting

    cfg = get_cfg('../config/capc_multi_disease_small.json')
    train_dataset = PosDiseaseDataset(
        in_csv_path=cfg.train_csv,
        cfg=cfg,
        mode='train')

    total_len = len(train_dataset)
    print('total len: ', total_len)
    total_positive_len = sum(train_dataset._labels)
    print('positive len: ', total_positive_len,
          total_positive_len / total_len * 100, '%')
    neg_len = total_len - total_positive_len
    print('negative len: ', neg_len, ' ', neg_len / total_len * 100, '%')

    dataloader_train = DataLoader(
        train_dataset,
        batch_size=40,
        num_workers=1,
        drop_last=True,
        shuffle=False)
    steps = len(dataloader_train)
    dataiter = iter(dataloader_train)
    targets = []

    total_len = 0
    total_positive_len = 0  # a diseaes is present
    for step in range(steps):
        _, target = next(dataiter)
        # print('target: ', target, target.device, target.dtype, target.shape)
        len = target.shape[0]
        total_len += len
        # print('total_len: ', total_len)
        # target - is there any disease present
        target = target.max(axis=1)[0]
        # print('target after max: ', target)
        positive_len = target.sum().item()
        # print('positive len: ', positive_len)
        total_positive_len += positive_len

    print('total len: ', total_len)
    print('positive len: ', total_positive_len,
          total_positive_len / total_len * 100, '%')
    neg_len = total_len - total_positive_len
    print('negative len: ', neg_len, ' ', neg_len / total_len * 100, '%')

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

    samples_per_cls = list(counts.values())
    weights = class_wise_loss_reweighting(
        beta=0.9999, samples_per_cls=samples_per_cls)
    print('weights: ', weights)
