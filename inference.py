import argparse
import os

import cv2
import numpy as np

from utils import MediaFile
from model import Model
from dataset import DroneDataset_12, DroneDataset_24

model_data = {"model_type": ["SWIN", "ResNet", "HRNetV2_W48", "SWIN_24"],

              "config_path": ['./configs/config_SWIN_12.py',
                              './configs/config_ResNet.py',
                              './configs/config_HRNetV2_W48.py'
                              './configs/config_SWIN_24.py'],

              "checkpoint_path": ['./work_dir/output_12_SWIN/iter_13000.pth',
                                  './work_dir/output_12_ResNet/iter_13000.pth',
                                  './work_dir/output_12_HR48/iter_9000.pth',
                                  './work_dir/output_24_SWIN/iter_13000.pth']}


def parse_args():
    parser = argparse.ArgumentParser(description='Inference model')
    parser.add_argument(
        '--variant', help="the model's variants", default="SWIN")
    parser.add_argument(
        '--config', help='the train config file path')
    parser.add_argument(
        '--model-path', help='the checkpoint file path')
    parser.add_argument(
        '--data', help='inference the media files or dir', default="./test")
    parser.add_argument(
        '--save-dir', help='folder to save result frames', default="./test")
    parser.add_argument(
        '--window-size', help='size of window in pixels', default=(1900, 1000))

    args = parser.parse_args()
    return args


def run_viewer(data_path: str, model: Model, args):
    (width, height) = args.window_size
    count = 0

    media_file = MediaFile(data_path, model)

    while True:
        image, seg_map = media_file.read()
        result_image = np.hstack([image, seg_map])

        c = max(result_image.shape[1] / width, result_image.shape[0] / height)
        result_image = cv2.resize(result_image, (round(result_image.shape[1] / c), round(result_image.shape[0] / c)))

        cv2.imshow("Result", result_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('a'):  # Next media file from dir
            media_file.reload(data_path, -1)

        elif key == ord('d'):  # Previous media file from dir
            media_file.reload(data_path, 1)

        elif key == ord('s'):  # Save frame as JPEG file
            count += 1
            cv2.imwrite(f"{args.save_dir}frame_{count}.jpg", image)
            print(f'Frame was saved at {args.save_dir}frame_{count}.jpg')

        elif key == ord('w'):  # Turn On/Off segmentation
            media_file.load_seg = not media_file.load_seg
            media_file.reload(data_path, 0)

        elif key == ord('e'):  # Turn On/Off segmentation mode
            media_file.scale = not media_file.scale
            media_file.reload(data_path, 0)

        elif key == ord('z'):
            if media_file.scale:
                media_file.resize_f = 1.0
                media_file.scale_f -= 0.1
                print(media_file.scale_f)
                media_file.reload(data_path, 0)
            else:
                if media_file.resize_f >= 1.0:
                    media_file.resize_f -= 0.1
                    print(media_file.resize_f)
                    media_file.reload(data_path, 0)


        elif key == ord('x'):
            if media_file.scale:
                media_file.resize_f = 1.0
                media_file.scale_f += 0.1
                print(media_file.scale_f)
                media_file.reload(data_path, 0)
            else:
                media_file.resize_f += 0.1
                print(media_file.resize_f)
                media_file.reload(data_path, 0)

        elif key == ord('q'):
            exit()


def main():
    config_file, checkpoint_file = '', ''

    args = parse_args()

    if (args.model_path and args.config) is not None:
        config_file = args.config
        checkpoint_file = args.model_path

    elif args.variant in model_data["model_type"]:
        config_file = model_data["config_path"][model_data["model_type"].index(args.variant)]
        checkpoint_file = model_data["checkpoint_path"][model_data["model_type"].index(args.variant)]

    if not os.path.exists(args.data):
        raise FileNotFoundError

    model = Model(config_file, checkpoint_file)
    run_viewer(args.data, model, args)


if __name__ == '__main__':
    main()
