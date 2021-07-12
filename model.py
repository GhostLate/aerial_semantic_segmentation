import typing
import mmcv
import numpy as np
from mmseg.apis import init_segmentor, inference_segmentor
from mmseg.datasets import build_dataset
from mmseg.models.segmentors import base
import cv2

from dataset import DroneDataset_12, DroneDataset_24  # Register datasets


class Model:
    checkpoint: str
    model: base.BaseSegmentor
    cfg: mmcv.Config
    dataset: None
    crop_size: typing.Tuple = (512, 515)

    def __init__(self, config_file: str, checkpoint_file: str):
        self.cfg = mmcv.Config.fromfile(config_file)
        self.dataset = build_dataset(self.cfg.data.test)
        self.checkpoint = checkpoint_file
        self.model = init_segmentor(self.cfg, checkpoint_file, device='cuda:0')

    def get_seg(self, image, scale: bool, scale_f: float = 1.0) -> np.array:
        if scale:
            pipeline = next((pipeline for pipeline in self.model.cfg.data.train.pipeline if "crop_size" in pipeline), None)
            if pipeline:
                self.crop_size = pipeline["crop_size"]
            self.crop_size = (int(self.crop_size[0] * scale_f), int(self.crop_size[1] * scale_f))
            for i, pipeline in enumerate(self.model.cfg.data.test.pipeline):
                if "img_scale" in pipeline:
                    self.model.cfg.data.test.pipeline[i].img_scale = self.crop_size

            crop_w = np.int(np.ceil(image.shape[1] / self.crop_size[0]))
            crop_h = np.int(np.ceil(image.shape[0] / self.crop_size[1]))

            seg_img = np.empty(shape=image.shape, dtype=np.uint8)
            for w in range(crop_w):
                for h in range(crop_h):
                    img = image[h * self.crop_size[1]:(h + 1) * self.crop_size[1],
                                w * self.crop_size[0]:(w + 1) * self.crop_size[0]]
                    i = self.perform_seg(img)
                    seg_img[h * self.crop_size[1]:(h + 1) * self.crop_size[1],
                    w * self.crop_size[0]:(w + 1) * self.crop_size[0]] = i
            return seg_img
        else:
            pipeline = next(
                (pipeline for pipeline in self.model.cfg.data.train.pipeline if "img_scale" in pipeline), None)
            if pipeline:
                for i, test_pipeline in enumerate(self.model.cfg.data.test.pipeline):
                    if "img_scale" in test_pipeline:
                        self.model.cfg.data.test.pipeline[i].img_scale = pipeline["img_scale"]
            return self.perform_seg(image)

    def perform_seg(self, image) -> np.array:
        result = inference_segmentor(self.model, image)
        if hasattr(self.model, 'module'):
            self.model = self.model.module
        return self.model.show_result(image, result, palette=self.dataset.PALETTE, show=False, opacity=0.5)