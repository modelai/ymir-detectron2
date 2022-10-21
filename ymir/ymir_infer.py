import numpy as np
import torch
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures import Boxes, RotatedBoxes

import logging
import os.path as osp
import sys
import warnings
from easydict import EasyDict as edict
from nptyping import NDArray, Shape
from typing import Any, List
from ymir.utils import CV_IMAGE, YmirStage, get_weight_file
from ymir_exc import dataset_reader as dr
from ymir_exc import env, monitor
from ymir_exc import result_writer as rw
from ymir_exc.util import YmirStage, get_merged_config, write_ymir_monitor_process

DETECTION_RESULT = NDArray[Shape['*,5'], Any]


def get_config_file(cfg):
    model_dir = cfg.ymir.input.models_dir
    config_file = osp.join(model_dir, 'config.yaml')

    if osp.exists(config_file):
        return config_file
    else:
        raise Exception(
            f'no config_file config.yaml found in {model_dir}')


class YmirModel(object):
    def __init__(self, ymir_cfg: edict):
        self.ymir_cfg = ymir_cfg
        self.class_names = ymir_cfg.param.class_names

        # Specify the path to model config and checkpoint file
        config_file = get_config_file(ymir_cfg)
        checkpoint_file = get_weight_file(ymir_cfg)
        self.conf = float(ymir_cfg.param.conf_threshold)

        logging.info(f"use {checkpoint_file} with confidence threshold {self.conf}")

        cfg_node = get_cfg()
        cfg_node.merge_from_file(config_file)

        # TODO cfg_node.merge_from_list(cfg.param.opts)
        cfg_node.MODEL.WEIGHTS = checkpoint_file

        # Set score_threshold for builtin models
        cfg_node.MODEL.RETINANET.SCORE_THRESH_TEST = self.conf
        cfg_node.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.conf
        cfg_node.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = self.conf
        cfg_node.freeze()
        self.predictor = DefaultPredictor(cfg_node)

    def infer(self, img: CV_IMAGE) -> List[rw.Annotation]:
        """
        boxes: Nx4 of XYXY_ABS --> instances.pred_boxes
        scores --> instances.scores
        classes --> instances.pred_classes.tolist()
        """
        predictions = self.predictor(img)
        instances = predictions['instances'].to(torch.device('cpu'))
        scores = instances.scores
        classes = instances.pred_classes
        boxes = instances.pred_boxes
        if isinstance(boxes, Boxes) or isinstance(boxes, RotatedBoxes):
            boxes = boxes.tensor.detach().numpy()
        else:
            boxes = np.asarray(boxes)
        anns = []

        for i in range(len(instances)):
            score = scores[i]
            if score <= self.conf:
                warnings.warn(f'score={score} < {self.conf}')
                continue
            cls = classes[i]
            xmin, ymin, xmax, ymax = boxes[i]
            if int(xmax - xmin) == 0 or int(ymax - ymin) == 0:
                continue
            ann = rw.Annotation(class_name=self.class_names[int(cls) - 1],
                                score=float(score),
                                box=rw.Box(x=int(xmin),
                                           y=int(ymin),
                                           w=int(xmax - xmin),
                                           h=int(ymax - ymin)))

            anns.append(ann)
        return anns


def main():
    cfg = get_merged_config()

    model = YmirModel(cfg)
    write_ymir_monitor_process(cfg, task='infer', naive_stage_percent=1.0, stage=YmirStage.PREPROCESS)

    N = dr.items_count(env.DatasetType.CANDIDATE)
    infer_result = dict()

    idx = -1

    monitor_gap = max(1, N // 1000)
    for asset_path, _ in tqdm(dr.item_paths(dataset_type=env.DatasetType.CANDIDATE)):
        # img = cv2.imread(asset_path)
        img = read_image(asset_path, format="BGR")
        result = model.infer(img)
        infer_result[asset_path] = result
        idx += 1

        if idx % monitor_gap == 0:
            write_ymir_monitor_process(cfg, task='infer', naive_stage_percent=idx / N, stage=YmirStage.TASK)

    rw.write_infer_result(infer_result=infer_result)
    write_ymir_monitor_process(cfg, task='infer', naive_stage_percent=1.0, stage=YmirStage.POSTPROCESS)

    return 0


if __name__ == '__main__':
    sys.exit(main())
