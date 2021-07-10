import mmcv
import numpy as np
from mmseg.apis import init_segmentor, inference_segmentor
from mmseg.datasets import build_dataset
from mmseg.models.segmentors import base

from dataset import DroneDataset_12, DroneDataset_24  # Register datasets


class Model:
    checkpoint: str
    model: base.BaseSegmentor
    cfg: mmcv.Config
    dataset: None

    def __init__(self, config_file: str, checkpoint_file: str):
        self.cfg = mmcv.Config.fromfile(config_file)
        self.dataset = build_dataset(self.cfg.data.test)
        self.checkpoint = checkpoint_file
        self.model = init_segmentor(self.cfg, checkpoint_file, device='cuda:0')

    def get_seg(self, image) -> np.array:
        result = inference_segmentor(self.model, image)
        if hasattr(self.model, 'module'):
            self.model = self.model.module
        return self.model.show_result(image, result, palette=self.dataset.PALETTE, show=False, opacity=0.5)
