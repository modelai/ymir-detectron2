"""
utils function for ymir and yolov5
"""
import yaml

from detectron2.config import CfgNode
from detectron2.data.datasets import register_coco_instances

import glob
import imagesize
import json
import os
import os.path as osp
from easydict import EasyDict as edict
from enum import IntEnum
from nptyping import NDArray, Shape, UInt8
from packaging.version import Version
from typing import Any, List, Optional
from ymir_exc import env
from ymir_exc import result_writer as rw


class YmirStage(IntEnum):
    PREPROCESS = 1  # convert dataset
    TASK = 2    # training/mining/infer
    POSTPROCESS = 3  # export model


BBOX = NDArray[Shape['*,4'], Any]
CV_IMAGE = NDArray[Shape['*,*,3'], UInt8]


def get_ymir_process(stage: YmirStage, p: float, task_idx: int = 0, task_num: int = 1) -> float:
    """
    stage: pre-process/task/post-process
    p: percent for stage
    task_idx: index for multiple tasks like mining (task_idx=0) and infer (task_idx=1)
    task_num: the total number of multiple tasks.
    """
    # const value for ymir process
    PREPROCESS_PERCENT = 0.1
    TASK_PERCENT = 0.8
    POSTPROCESS_PERCENT = 0.1

    if p < 0 or p > 1.0:
        raise Exception(f'p not in [0,1], p={p}')

    init = task_idx * 1.0 / task_num
    ratio = 1.0 / task_num
    if stage == YmirStage.PREPROCESS:
        return init + PREPROCESS_PERCENT * p * ratio
    elif stage == YmirStage.TASK:
        return init + (PREPROCESS_PERCENT + TASK_PERCENT * p) * ratio
    elif stage == YmirStage.POSTPROCESS:
        return init + (PREPROCESS_PERCENT + TASK_PERCENT + POSTPROCESS_PERCENT * p) * ratio
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
    ymir_dataset_dir = osp.join(out_dir, 'ymir_dataset')
    os.makedirs(ymir_dataset_dir, exist_ok=True)

    for split, prefix in zip(['train', 'val'], ['training', 'val']):
        src_file = getattr(cfg.ymir.input, f'{prefix}_index_file')
        with open(src_file) as fp:
            lines = fp.readlines()

        img_id = 0
        ann_id = 0
        data = dict(images=[],
                    annotations=[],
                    categories=[],
                    licenses=[],
                    info='convert from ymir'
                    )

        cat_id_start = 1
        for id, name in enumerate(cfg.param.class_names):
            data['categories'].append(dict(id=id + cat_id_start,
                                           name=name,
                                           supercategory="none"))

        for line in lines:
            img_file, ann_file = line.strip().split()
            width, height = imagesize.get(img_file)
            img_info = dict(file_name=img_file,
                            height=height,
                            width=width,
                            id=img_id)

            data['images'].append(img_info)

            if osp.exists(ann_file):
                for ann_line in open(ann_file, 'r').readlines():
                    ann_strlist = ann_line.strip().split(',')
                    class_id, x1, y1, x2, y2 = [int(s) for s in ann_strlist[0:5]]
                    bbox_width = x2 - x1
                    bbox_height = y2 - y1
                    bbox_area = bbox_width * bbox_height
                    bbox_quality = float(ann_strlist[5]) if len(
                        ann_strlist) > 5 and ann_strlist[5].isnumeric() else 1
                    ann_info = dict(bbox=[x1, y1, bbox_width, bbox_height],   # x,y,width,height
                                    area=bbox_area,
                                    score=1.0,
                                    bbox_quality=bbox_quality,
                                    iscrowd=0,
                                    segmentation=[[x1, y1, x1, y2, x2, y2, x2, y1]],
                                    category_id=class_id + cat_id_start,   # start from cat_id_start
                                    id=ann_id,
                                    image_id=img_id)
                    data['annotations'].append(ann_info)
                    ann_id += 1

            img_id += 1

        with open(osp.join(ymir_dataset_dir, f'ymir_{split}.json'), 'w') as fw:
            json.dump(data, fw)


def modify_detectron2_config(detectron_cfg: CfgNode) -> CfgNode:
    """
    - modify dataset config
    - modify model output channel
    - modify epochs, checkpoint, tensorboard config
    """
    # config_file = ymir_cfg.param.config_file
    # num_gpus = len(ymir_cfg.param.gpu_id.split(','))
    # args_options = ymir_cfg.param.args_options

    ymir_cfg = get_merged_config()

    # register ymir_dataset train/val in detectron2
    convert_ymir_to_coco(ymir_cfg)
    register_coco_instances("ymir_dataset_train", {}, "/out/ymir_dataset/ymir_train.json", "/in")
    register_coco_instances("ymir_dataset_val", {}, "/out/ymir_dataset/ymir_val.json", "/in")

    models_dir = ymir_cfg.ymir.output.models_dir
    num_classes = len(ymir_cfg.param.class_names)
    batch_size = int(ymir_cfg.param.batch_size)
    max_iter = int(ymir_cfg.param.max_iter)
    learning_rate = float(ymir_cfg.param.learning_rate)

    detectron_cfg.DATASETS.TRAIN = ('ymir_dataset_train',)
    detectron_cfg.DATASETS.TEST = ('ymir_dataset_val',)
    detectron_cfg.OUTPUT_DIR = models_dir
    detectron_cfg.MODEL.RETINANET.NUM_CLASSES = num_classes
    detectron_cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    detectron_cfg.SOLVER.IMS_PER_BATCH = batch_size
    # detectron_cfg.YMIR=CfgNode()
    detectron_cfg.YMIR.TENSORBOARD_DIR = ymir_cfg.ymir.output.tensorboard_dir
    # modify iters, checkpoint, tensorboard config
    if max_iter > 0:
        # use dynamic default value if max_iter <= 0
        detectron_cfg.SOLVER.MAX_ITER = max_iter

    if learning_rate > 0:
        # use dynamic default value if learning_rate <= 0
        detectron_cfg.SOLVER.BASE_LR = learning_rate

    return detectron_cfg


def get_weight_file(cfg: edict) -> str:
    """
    return the weight file path by priority
    find weight file in cfg.param.pretrained_model_params or cfg.param.model_params_path
    """
    model_params_path: List = []
    if cfg.ymir.run_training:
        model_params_path = cfg.param.get('pretrained_model_params', [])
    else:
        model_params_path = cfg.param.get('model_params_path', [])

    model_dir = cfg.ymir.input.models_dir
    model_params_path = [
        osp.join(model_dir, p) for p in model_params_path if osp.exists(osp.join(model_dir, p)) and p.endswith(('.pth', '.pt'))]

    if len(model_params_path) > 0:
        return max(model_params_path, key=os.path.getctime)

    return ""


def write_ymir_training_result(last: bool = False):
    EVAL_TMP_FILE = os.getenv('EVAL_TMP_FILE')
    if EVAL_TMP_FILE is None:
        raise Exception(
            'please set valid environment variable EVAL_TMP_FILE to write result into json file')

    with open(EVAL_TMP_FILE, 'r') as f:
        eval_result = json.load(f)

    map50 = eval_result.get('AP50', 0) / 100
    YMIR_VERSION = os.environ.get('YMIR_VERSION', '1.2.0')
    if Version(YMIR_VERSION) >= Version('1.2.0'):
        write_latest_ymir_training_result(last, map50)
    elif last:
        write_ancient_ymir_training_result(map50)


def write_latest_ymir_training_result(last: bool = False, map: float = 0):
    WORK_DIR = os.getenv('YMIR_MODELS_DIR')
    if WORK_DIR is None or not osp.isdir(WORK_DIR):
        raise Exception(
            f'please set valid environment variable YMIR_MODELS_DIR, invalid directory {WORK_DIR}')

    # assert only one model config file in work_dir
    result_files = [osp.basename(f) for f in glob.glob(
        osp.join(WORK_DIR, '*')) if osp.basename(f) != 'result.yaml' and osp.isfile(f)]

    if last:
        # save all output file
        rw.write_model_stage(files=result_files,
                             mAP=float(map),
                             stage_name='last')
    else:
        # save newest weight file in format model_0001234.pth
        if 'model_final.pth' in result_files:
            result_files.remove('model_final.pth')
        weight_files = [osp.join(WORK_DIR, f)
                        for f in result_files if f.startswith('model_') and f.endswith('.pth')]

        if len(weight_files) > 0:
            newest_weight_file = osp.basename(
                max(weight_files, key=os.path.getctime))

            stage_name = osp.splitext(newest_weight_file)[0]
            training_result_file = osp.join(WORK_DIR, 'result.yaml')
            if osp.exists(training_result_file):
                with open(training_result_file, 'r') as f:
                    training_result = yaml.safe_load(f)
                    model_stages = training_result.get('model_stages', {})
            else:
                model_stages = {}

            if stage_name not in model_stages:
                config_files = [osp.join(WORK_DIR, 'config.yaml')]
                rw.write_model_stage(files=[newest_weight_file] + config_files,
                                     mAP=float(map),
                                     stage_name=stage_name)


def write_ancient_ymir_training_result(map: float = 0):
    WORK_DIR = os.getenv('YMIR_MODELS_DIR')
    if WORK_DIR is None or not osp.isdir(WORK_DIR):
        raise Exception(
            f'please set valid environment variable YMIR_MODELS_DIR, invalid directory {WORK_DIR}')

    # assert only one model config file in work_dir
    result_files = [osp.basename(f) for f in glob.glob(
        osp.join(WORK_DIR, '*')) if osp.basename(f) != 'result.yaml' and osp.isfile(f)]

    training_result_file = osp.join(WORK_DIR, 'result.yaml')
    if osp.exists(training_result_file):
        with open(training_result_file, 'r') as f:
            training_result = yaml.safe_load(f)

        training_result['model'] = result_files
        training_result['map'] = max(map, training_result['map'])
    else:
        training_result = dict(model=result_files, map=map)

    with open(training_result_file, 'w') as f:
        yaml.safe_dump(training_result, f)
