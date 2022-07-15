import os
import numpy as np
import torch
import torch.utils.data as data
import imageio
from PIL import Image
import xml.etree.ElementTree as ET


class DataLoader(data.Dataset):
    def __init__(self, data_path="", trainval='trainval', transform=None):
        self.data_path = data_path
        self.transform = transform
        self.trainval = trainval

        self.__init_classes()
        self.names, self.labels = self.__dataset_info()

    def __getitem__(self, index):
        x = imageio.imread(
            self.data_path + 'JPEGImages/' + self.names[index] + '.jpg',
            pilmode='RGB')
        x = Image.fromarray(x)
        if self.transform != None:
            x = self.transform(x)
        y = self.labels[index]
        y = torch.tensor(y)
        # print(x,y)
        return x, y

    def __len__(self):
        return len(self.names)

    def __dataset_info(self):
        # annotation_files = os.listdir(self.data_path+'/Annotations')
        with open(
                self.data_path + 'ImageSets/Main/' + self.trainval + '.txt') as f:
            annotations = f.readlines()
        annotations = [n[:-1] for n in annotations]
        names = []
        labels = []
        for af in annotations:
            filename = os.path.join(self.data_path, 'Annotations', af)
            tree = ET.parse(filename + '.xml')
            objs = tree.findall('object')
            num_objs = len(objs)

            boxes_cl = np.zeros((num_objs), dtype=np.int32)

            for ix, obj in enumerate(objs):
                cls = self.class_to_ind[obj.find('name').text.lower().strip()]
                boxes_cl[ix] = cls

            lbl = np.zeros(self.num_classes)
            lbl[boxes_cl] = 1
            labels.append(lbl)
            names.append(af)

        return np.array(names), np.array(labels).astype(np.float32)

    def __init_classes(self):
        self.classes = ('aeroplane', 'bicycle', 'bird', 'boat',
                        'bottle', 'bus', 'car', 'cat', 'chair',
                        'cow', 'diningtable', 'dog', 'horse',
                        'motorbike', 'person', 'pottedplant',
                        'sheep', 'sofa', 'train', 'tvmonitor')
        self.num_classes = len(self.classes)
        self.class_to_ind = dict(zip(self.classes, range(self.num_classes)))


if __name__ == '__main__':
    pascal_path = os.path.join('/home/nicolas/data', 'VOC2012/')
    data = DataLoader(pascal_path, 'small_train_0')
    augmented_dataloader = torch.utils.data.DataLoader(data,
                                                       batch_size=32,
                                                       shuffle=True)
    x, y = data.__getitem__(3)
    print(x, y)
    print(augmented_dataloader.dataset.labels)
