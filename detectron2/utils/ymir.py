"""
utils function for ymir and yolov5
"""
import os
import os.path as osp
import shutil
from enum import IntEnum
from typing import Any, List, Tuple

import numpy as np
import torch
import yaml
import json
from easydict import EasyDict as edict
from nptyping import NDArray, Shape, UInt8
from ymir_exc import env
from ymir_exc import result_writer as rw
import imagesize

class YmirStage(IntEnum):
    PREPROCESS = 1  # convert dataset
    TASK = 2    # training/mining/infer
    POSTPROCESS = 3  # export model


BBOX = NDArray[Shape['*,4'], Any]
CV_IMAGE = NDArray[Shape['*,*,3'], UInt8]


def get_ymir_process(stage: YmirStage, p: float) -> float:
    # const value for ymir process
    PREPROCESS_PERCENT = 0.1
    TASK_PERCENT = 0.8
    POSTPROCESS_PERCENT = 0.1

    if p < 0 or p > 1.0:
        raise Exception(f'p not in [0,1], p={p}')

    if stage == YmirStage.PREPROCESS:
        return PREPROCESS_PERCENT * p
    elif stage == YmirStage.TASK:
        return PREPROCESS_PERCENT + TASK_PERCENT * p
    elif stage == YmirStage.POSTPROCESS:
        return PREPROCESS_PERCENT + TASK_PERCENT + POSTPROCESS_PERCENT * p
    else:
        raise NotImplementedError(f'unknown stage {stage}')


def get_merged_config() -> edict:
    """
    merge ymir_config and executor_config
    """
    merged_cfg = edict()
    # the hyperparameter information
    merged_cfg.param = env.get_executor_config()

    # the ymir path information
    merged_cfg.ymir = env.get_current_env()
    return merged_cfg

def get_weight_file(cfg: edict) -> str:
    """
    return the weight file path by priority
    find weight file in cfg.param.model_params_path or cfg.param.model_params_path
    """
    if cfg.ymir.run_training:
        model_params_path = cfg.param.pretrained_model_paths
    else:
        model_params_path = cfg.param.model_params_path

    model_dir = osp.join(cfg.ymir.input.root_dir,
                         cfg.ymir.input.models_dir)
    model_params_path = [p for p in model_params_path if osp.exists(osp.join(model_dir, p))]

    # choose weight file by priority, best.pt > xxx.pt
    if 'best.pt' in model_params_path:
        return osp.join(model_dir, 'best.pt')
    else:
        for f in model_params_path:
            if f.endswith('.pt'):
                return osp.join(model_dir, f)

    return ""


# def download_weight_file(model_name):
#     weights = attempt_download(f'{model_name}.pt')
#     return weights


# class YmirYolov5():
#     """
#     used for mining and inference to init detector and predict.
#     """

#     def __init__(self, cfg: edict):
#         self.cfg = cfg
#         device = select_device(cfg.param.get('gpu_id', 'cpu'))

#         self.model = self.init_detector(device)
#         self.device = device
#         self.class_names = cfg.param.class_names
#         self.stride = self.model.stride
#         self.conf_thres = float(cfg.param.conf_thres)
#         self.iou_thres = float(cfg.param.iou_thres)

#         img_size = int(cfg.param.img_size)
#         imgsz = (img_size, img_size)
#         imgsz = check_img_size(imgsz, s=self.stride)

#         self.model.warmup(imgsz=(1, 3, *imgsz), half=False)  # warmup
#         self.img_size = imgsz

#     def init_detector(self, device: torch.device) -> DetectMultiBackend:
#         weights = get_weight_file(self.cfg)

#         data_yaml = osp.join(self.cfg.ymir.output.root_dir, 'data.yaml')
#         model = DetectMultiBackend(weights=weights,
#                                    device=device,
#                                    dnn=False,  # not use opencv dnn for onnx inference
#                                    data=data_yaml)  # dataset.yaml path

#         return model

#     def predict(self, img: CV_IMAGE) -> NDArray:
#         """
#         predict single image and return bbox information
#         img: opencv BGR, uint8 format
#         """
#         # preprocess: padded resize
#         img1 = letterbox(img, self.img_size, stride=self.stride, auto=True)[0]

#         # preprocess: convert data format
#         img1 = img1.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
#         img1 = np.ascontiguousarray(img1)
#         img1 = torch.from_numpy(img1).to(self.device)

#         img1 = img1 / 255  # 0 - 255 to 0.0 - 1.0
#         img1.unsqueeze_(dim=0)  # expand for batch dim
#         pred = self.model(img1)

#         # postprocess
#         conf_thres = self.conf_thres
#         iou_thres = self.iou_thres
#         classes = None  # not filter class_idx in results
#         agnostic_nms = False
#         max_det = 1000

#         pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

#         result = []
#         for det in pred:
#             if len(det):
#                 # Rescale boxes from img_size to img size
#                 det[:, :4] = scale_coords(img1.shape[2:], det[:, :4], img.shape).round()
#                 result.append(det)

#         # xyxy, conf, cls
#         if len(result) > 0:
#             tensor_result = torch.cat(result, dim=0)
#             numpy_result = tensor_result.data.cpu().numpy()
#         else:
#             numpy_result = np.zeros(shape=(0, 6), dtype=np.float32)

#         return numpy_result

#     def infer(self, img: CV_IMAGE) -> List[rw.Annotation]:
#         anns = []
#         result = self.predict(img)

#         for i in range(result.shape[0]):
#             xmin, ymin, xmax, ymax, conf, cls = result[i, :6].tolist()
#             ann = rw.Annotation(class_name=self.class_names[int(cls)], score=conf, box=rw.Box(
#                 x=int(xmin), y=int(ymin), w=int(xmax - xmin), h=int(ymax - ymin)))

#             anns.append(ann)

#         return anns


def convert_ymir_to_coco(cfg: edict) -> None:
    """
    convert ymir format dataset to coco format
    generate coco_{train/val/test}.json for training/mining/infer
    view follow line for detail:
    - https://cocodataset.org/#format-data
    - https://github.com/facebookresearch/detectron2/blob/main/datasets/README.md
    - https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html
    """
    out_dir = cfg.ymir.output.root_dir
    # os.environ.setdefault('DETECTRON2_DATASETS', out_dir)
    ymir_dataset_dir = osp.join(out_dir,'ymir_dataset')
    os.makedirs(ymir_dataset_dir, exist_ok=True)


    for split, prefix in zip(['train', 'val'], ['training', 'val']):
        src_file = getattr(cfg.ymir.input, f'{prefix}_index_file')
        with open(src_file) as fp:
            lines = fp.readlines()

        img_id=0
        ann_id=0
        data = dict(images=[],
                annotations=[],
                categories=[],
                licenses=[],
                info='convert from ymir')

        for id,name in enumerate(cfg.param.class_names):
            data['categories'].append(dict(id=id,
                name=name,
                supercategory="none"))

        for line in lines:
            img_file, ann_file = line.strip().split()
            width, height = imagesize.get(img_file)
            img_info=dict(file_name=img_file,
                height=height,
                width=width,
                id=img_id)

            data['images'].append(img_info)

            if osp.exists(ann_file):
                for ann_line in open(ann_file,'r').readlines():
                    ann_strlist = ann_line.strip().split(',')
                    class_id, x1, y1, x2, y2 = [int(s) for s in ann_strlist[0:5]]
                    bbox_width = x2-x1
                    bbox_height = y2-y1
                    bbox_area = bbox_width * bbox_height
                    bbox_quality = float(ann_strlist[5]) if len(ann_strlist)>5 and ann_strlist[5].isnumeric() else 1
                    ann_info = dict(bbox=[x1,y1,bbox_width,bbox_height],   # x,y,width,height
                        area=bbox_area,
                        score=1.0,
                        bbox_quality=bbox_quality,
                        iscrowd=0,
                        segmentation=[[x1,y1,x1,y2,x2,y2,x2,y1]],
                        category_id=class_id,   # start from 0
                        id=ann_id,
                        image_id=img_id)
                    data['annotations'].append(ann_info)
                    ann_id+=1

            img_id+=1

        with open(osp.join(ymir_dataset_dir, f'ymir_{split}.json'), 'w') as fw:
            json.dump(data, fw)


def write_ymir_training_result(cfg: edict, results: Tuple, maps: NDArray, rewrite=False) -> int:
    """
    cfg: ymir config
    results: (mp, mr, map50, map, loss)
    maps: map@0.5:0.95 for all classes
    rewrite: set true to ensure write the best result
    """
    if not rewrite:
        training_result_file = cfg.ymir.output.training_result_file
        if osp.exists(training_result_file):
            return 0

    model = cfg.param.model
    class_names = cfg.param.class_names
    mp = results[0]  # mean of precision
    mr = results[1]  # mean of recall
    map50 = results[2]  # mean of ap@0.5
    map = results[3]  # mean of ap@0.5:0.95

    # use `rw.write_training_result` to save training result
    rw.write_training_result(model_names=[f'{model}.yaml', 'best.pt', 'last.pt', 'best.onnx'],
                             mAP=float(map),
                             mAP50=float(map50),
                             precision=float(mp),
                             recall=float(mr),
                             classAPs={class_name: v
                                       for class_name, v in zip(class_names, maps.tolist())})
    return 0
