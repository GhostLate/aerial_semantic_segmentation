import mimetypes
import os
import cv2
import numpy as np

from model import Model


class MediaFile:
    image: np.array
    seg_map: np.array
    is_video: bool = False
    file_path: str
    idx: int = 0
    vid_capture: cv2.VideoCapture
    load_seg: bool = True
    model: Model
    scale: bool = False
    scale_f: float = 1.0
    resize_f: float = 1.0

    def __init__(self, data_path: str, model: Model):
        self.model = model
        self.reload(data_path, 0)

    def reload(self, file_path: str, change: int):
        if os.path.isfile(file_path):
            file_dir = os.path.dirname(file_path)
            file_names = sorted(os.listdir(file_dir))
            self.idx = file_names.index(os.path.basename(file_path))
        else:
            file_dir = file_path
            file_names = sorted(os.listdir(file_dir))

        new_idx = self.idx

        while True:
            new_idx += change

            if new_idx <= -1:
                new_idx = len(file_names) - 1
            elif new_idx >= len(file_names):
                new_idx = 0

            file_path = os.path.join(file_dir, file_names[new_idx])
            media_type = mimetypes.guess_type(file_path)[0]

            if media_type.startswith('video') or media_type.startswith('image'):
                if media_type.startswith('video'):
                    self.vid_capture = cv2.VideoCapture(file_path)
                else:
                    self.image = cv2.cvtColor(cv2.imread(file_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                    self.image = cv2.resize(self.image, (int(self.image.shape[1]/self.resize_f), int(self.image.shape[0]/self.resize_f)))
                    if self.load_seg:
                        self.seg_map = self.model.get_seg(self.image, self.scale, self.scale_f)
                    else:
                        self.seg_map = self.image
                self.file_path = file_path
                self.idx = new_idx
                break
            elif new_idx == self.idx:
                print("There isn't other media file in folder!")
                break

    def read(self):
        if mimetypes.guess_type(self.file_path)[0].startswith('video'):
            self.image = self.vid_capture.read()[1]
            self.image = cv2.resize(self.image, (int(self.image.shape[1]/self.resize_f), int(self.image.shape[0]/self.resize_f)))
            if self.load_seg:
                self.seg_map = self.model.get_seg(self.image, self.scale, self.scale_f)
            else:
                self.seg_map = self.image
        return self.image, self.seg_map