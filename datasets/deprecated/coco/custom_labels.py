import os
import torch
import urllib
from PIL import Image
from torchvision import datasets as datasets

from general_utils.save_load import load_obj

label_files_url = {
    'smallval2017': "https://bit.ly/3p13xP1",
    'test2017': "https://bit.ly/39FtAVe",
    'val2017': "https://bit.ly/38V5ZRn",
    'train2017': "https://bit.ly/3o7G3GN",
    'unlabeled2017': "https://bit.ly/2M53piK",
}


class CocoDatasetCustom(datasets.coco.CocoDetection):

    def __init__(self, root, label_file, transform=None, target_transform=None):
        self.root = root
        if not os.path.isfile(label_file):
            key = label_file.split('/')[-1].split('.')[0]
            url = label_files_url[key]
            urllib.request.urlretrieve(url, label_file)
        try:
            self.labels = load_obj(label_file)
        except Exception as err:
            print(err)
            print(f'Delete the file with labels: {label_file} and try '
                  f'downloading it again. It might be malformatted.')
            exit(1)
        self.ids = list(range(len(self.labels)))
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        label = self.labels[index]
        img_path = label[0]
        target = torch.tensor(label[1])
        img = Image.open(os.path.join(self.root, img_path)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


def main():
    print('Test custom labels.')

    from torchvision import transforms
    from getpass import getuser
    import os

    user = getuser()
    normalize = transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
    coco_image_size = 448
    data_name = f"smallval2017"
    data_path = f"/home/{user}/data/coco"
    data_path = os.path.join(data_path, data_name)
    cwd = os.getcwd()
    print('curr dir: ', cwd)
    label_path = os.path.join(cwd, f'labels/{data_name}.pkl')
    dataset = CocoDatasetCustom(
        data_path,
        label_path,
        transforms.Compose([
            transforms.Resize(
                (coco_image_size, coco_image_size)),
            transforms.ToTensor(),
            normalize,
        ]))

    print("len(val_dataset)): ", len(dataset))
    for i in range(10):
        _, target = dataset[i]
        print('target: ', target)


if __name__ == "__main__":
    main()
