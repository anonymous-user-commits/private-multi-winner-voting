import torch
from PIL import Image
from torchvision import datasets as datasets

from general_utils.save_load import save_obj


class CocoDetectionWithImgPath(datasets.coco.CocoDetection):
    def __init__(self, root, annFile, transform=None, target_transform=None):
        from pycocotools.coco import COCO
        self.root = root
        self.coco = COCO(annFile)

        self.ids = list(self.coco.imgToAnns.keys())
        self.transform = transform
        self.target_transform = target_transform
        self.cat2cat = dict()
        for cat in self.coco.cats.keys():
            self.cat2cat[cat] = len(self.cat2cat)
        # print(self.cat2cat)

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        output = torch.zeros((3, 80), dtype=torch.long)
        for obj in target:
            if obj['area'] < 32 * 32:
                output[0][self.cat2cat[obj['category_id']]] = 1
            elif obj['area'] < 96 * 96:
                output[1][self.cat2cat[obj['category_id']]] = 1
            else:
                output[2][self.cat2cat[obj['category_id']]] = 1
        target = output
        # For each of 3 areas, set one for category_id if at least one of the
        # area is set to 1 there.
        target = target.max(dim=0)[0]

        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, path

if __name__ == "__main__":
    from torchvision import transforms
    from getpass import getuser
    import os
    import time

    start = time.time()
    user = getuser()
    normalize = transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
    coco_image_size = 448
    # data_name = f"val2017"
    data_name = f"train2017"
    data_path = f"/home/{user}/data/coco"
    instances_path = os.path.join(
        data_path, f'annotations/instances_{data_name}.json')
    data_path = os.path.join(data_path, data_name)
    dataset = CocoDetectionWithImgPath(
        data_path,
        instances_path,
        transforms.Compose([
            transforms.Resize(
                (coco_image_size, coco_image_size)),
            transforms.ToTensor(),
            normalize,
        ]))

    print("len(val_dataset)): ", len(dataset))

    labels = []
    data_len = len(dataset)
    # data_len = 10
    for i in range(data_len):
        _, target, path = dataset[i]
        target = list(target.detach().cpu().numpy())
        labels.append((path, target))

    save_obj(file=f'./labels/{data_name}.pkl', obj=labels)
    stop = time.time()
    elapsed_time = stop - start
    print('elapsed time (sec): ', elapsed_time)