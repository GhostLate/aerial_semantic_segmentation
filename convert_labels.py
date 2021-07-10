import os
import cv2
import numpy as np
import argparse
from PIL import Image
from tqdm import tqdm

native_classes = (
    'unlabeled', 'paved-area', 'dirt', 'grass',
    'gravel', 'water', 'rocks', 'pool',
    'vegetation', 'roof', 'wall', 'window',
    'door', 'fence', 'fence-pole', 'person',
    'dog', 'car', 'bicycle', 'tree',
    'bald-tree', 'ar-marker', 'obstacle', 'conflicting'
)
native_palette = [
    [0, 0, 0], [128, 64, 128], [130, 76, 0], [0, 102, 0],
    [112, 103, 87], [28, 42, 168], [48, 41, 30], [0, 50, 89],
    [107, 142, 35], [190, 153, 153], [102, 102, 156], [254, 228, 12],
    [254, 148, 12], [70, 70, 70], [153, 153, 153], [255, 22, 96],
    [102, 51, 0], [9, 143, 150, ], [119, 11, 32], [51, 51, 0],
    [190, 250, 190], [112, 150, 146], [2, 135, 115], [255, 0, 0]
]

new_classes = (
    'background', 'road', 'ground', 'water',
    'vegetation', 'construction', 'person', 'dog',
    'car', 'bicycle', 'obstacle', 'conflicting'
)

convert_map = [[0], [1], [2, 3, 4, 6],
       [5, 7], [8, 19, 20], [9, 10, 11, 12, 13, 14],
       [15], [16], [17], [18],
       [21, 22], [23]]


def parse_args():
    parser = argparse.ArgumentParser(description='Convert and resize labels from 24 classes of DroneDataset_24 to num classes you choose. '
                                                 'It also supports conversion from RGB labels to GrayScale')
    parser.add_argument(
        '--labels-dir', help="Dir to native labels", default="./dataset/label_images")
    parser.add_argument(
        '--save-dir', help='Dir where converted labels will be saved')
    parser.add_argument(
        '--label-res', help='Resolution of converted labels', default=(768, 512))
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if not os.path.exists(args.labels_dir):
        raise FileNotFoundError

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    file_names = sorted(os.listdir(args.labels_dir))

    for file_name in tqdm(file_names):
        label = cv2.imread(os.path.join(args.labels_dir, file_name), cv2.IMREAD_UNCHANGED)
        label = cv2.resize(label, (768, 512), interpolation=cv2.INTER_NEAREST)

        if len(label.shape) == 3:  # If true, convert RGB label to GrayScale
            label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
            gray_label = np.zeros((label.shape[:2]), dtype=np.int8)
            for i in range(len(native_palette)):
                gray_label[(label == np.array(native_palette[i])).all(axis=2)] = i
            label = gray_label

        label_seg = np.zeros((label.shape[:2]), dtype=np.int8)
        for i in range(len(convert_map)):
            for index in convert_map[i]:
                label_seg[label == index] = i

        cv2.imwrite(os.path.join(args.save_dir, file_name), label_seg)


if __name__ == '__main__':
    main()