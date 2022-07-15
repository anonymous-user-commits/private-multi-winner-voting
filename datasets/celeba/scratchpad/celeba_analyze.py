from getpass import getuser
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets.celeba.celeba_dataset import CelebADataset

len_celeba_all = 202599
len_celeba_train = 162770
len_celeba_valid = 19867
len_celeba_test = 19962
celeba_H = 218
celeba_W = 178
num_classes = 40

"""
split:  all
self.root:  /home/nicolas/code/capc-learning/celeba
self.base_folder:  celeba
len all:  202599
batch_index:  0
img shape:  torch.Size([4, 3, 218, 178])
label:  tensor(
[[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0,
         1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0,
         1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
         1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1]])
split:  train
self.root:  /home/nicolas/code/capc-learning/celeba
self.base_folder:  celeba
len train:  162770
"""


def main():
    kwargs = {'num_workers': 1,
              'pin_memory': True}
    user = getuser()
    dataset_path = f'/home/{user}/code/capc-learning/celeba'
    # dataset_type = 'native'
    dataset_type = 'custom'

    for split in ['all', 'train', 'valid', 'test']:
        print('split: ', split)

        if dataset_type == 'native':
            dataset_extractor = datasets.CelebA
        elif dataset_type == 'custom':
            dataset_extractor = CelebADataset
        else:
            raise Exception(f"Unknown dataset_type: {dataset_type}.")

        dataset_all = dataset_extractor(
            dataset_path,
            split=split,
            target_type='attr',
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]),
            download=False)
        print(f'len {split}: ', len(dataset_all))
        dataloader = DataLoader(
            dataset_all,
            batch_size=4,
            shuffle=False,
            **kwargs)
        max_count = 1
        for batch_index, (img, label) in enumerate(dataloader):
            print('batch_index: ', batch_index)
            print('img shape: ', img.shape)
            print('label: ', label)
            if batch_index + 1 == max_count:
                break


if __name__ == "__main__":
    main()
