import numpy as np

from datasets.deprecated.coco.helper_functions.helper_functions import coco_classes_list

class_ratios = [59.23, 3.05, 11.46, 3.27, 2.79, 3.72, 3.35, 5.7, 2.77, 3.82,
                1.53, 1.58, 0.64, 5.19, 3.03, 3.86, 4.13, 2.78, 1.44, 1.81,
                1.96, 0.87, 1.77, 2.39, 5.04, 3.68, 6.23, 3.54, 1.97, 2.05,
                2.79, 1.54, 4.02, 2.07, 2.36, 2.48, 3.29, 3.26, 3.15, 7.61,
                2.16, 7.53, 2.68, 3.38, 2.68, 5.3, 1.39, 0.97, 1.99, 1.01,
                0.91, 0.91, 0.89, 2.16, 1.02, 2.09, 11.74, 4.13, 4.03, 3.35,
                9.62, 1.6, 4.18, 2.98, 1.5, 2.74, 1.68, 4.45, 1.24, 2.23,
                0.18, 3.15, 2.17, 4.61, 4.29, 2.67, 0.6, 1.43, 0.14, 0.9]

threshold_9_99 = ['person', 'car', 'chair']  # size 3



threshold_4_99 = ['person', 'car', 'truck', 'bench', 'backpack', 'handbag', 'bottle', 'cup',
 'bowl', 'chair', 'dining table'] # size 11

threshold_3_99 = ['person', 'car', 'truck', 'bench', 'dog', 'backpack',
                  'handbag',
                  'sports ball', 'bottle', 'cup', 'bowl', 'chair', 'couch',
                  'potted plant',
                  'dining table', 'tv', 'cell phone', 'book',
                  'clock']  # size 19

threshold_0_99 = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                  'train',
                  'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
                  'bench',
                  'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant',
                  'zebra',
                  'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                  'suitcase',
                  'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
                  'baseball bat',
                  'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                  'bottle',
                  'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                  'banana',
                  'sandwich', 'orange', 'pizza', 'donut', 'cake', 'chair',
                  'couch',
                  'potted plant', 'bed', 'dining table', 'toilet', 'tv',
                  'laptop', 'mouse',
                  'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                  'sink',
                  'refrigerator', 'book', 'clock', 'vase',
                  'teddy bear']  # size 70


def main():
    classes = coco_classes_list
    ratios = class_ratios
    assert len(classes) == len(ratios)
    threshold = 5.05
    classes_threshold = []
    for i in range(len(ratios)):
        if ratios[i] > threshold:
            print(classes[i], ratios[i])
            classes_threshold.append(classes[i])

    classes_threshold_str = np.array2string(
        np.array(classes_threshold), precision=2, separator=', ')
    print(classes_threshold_str)
    print('size: ', len(classes_threshold))


if __name__ == "__main__":
    main()
