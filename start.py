import logging
import os
import os.path as osp
import shutil
import subprocess
import sys

import cv2
from easydict import EasyDict as edict
from ymir_exc import dataset_reader as dr
from ymir_exc import env, monitor
from ymir_exc import result_writer as rw

from detectron2.utils.ymir import get_merged_config, get_ymir_process, YmirStage, convert_ymir_to_coco

def start() -> int:
    cfg = get_merged_config()

    logging.info(f'merged config: {cfg}')

    if cfg.ymir.run_training:
        _run_training(cfg)
    elif cfg.ymir.run_mining:
        _run_mining(cfg)
    elif cfg.ymir.run_infer:
        _run_infer(cfg)
    else:
        logging.warning('no task running')

    return 0


def _run_training(cfg: edict) -> None:
    # convert ymir dataset to coco format
    convert_ymir_to_coco(cfg)
    monitor.write_monitor_logger(percent=get_ymir_process(stage=YmirStage.PREPROCESS, p=1.0))

    config_file = cfg.param.config_file
    num_gpus = len(cfg.param.gpu_id.split(','))
    models_dir = cfg.ymir.output.models_dir
    args_options = cfg.param.args_options
    num_classes = len(cfg.param.class_names)
    batch_size = int(cfg.param.batch_size)
    command = f'python3 tools/train_net.py --config-file {config_file}' + \
        f' --num-gpus {num_gpus}'

    if config_file.endswith('.py'):
        command += f" train.output_dir={models_dir} MODEL.RETINANET.NUM_CLASSES={num_classes}" + \
                f" MODEL.ROI_HEADS.NUM_CLASSES={num_classes}" + \
                f" SOLVER.IMS_PER_BATCH={batch_size}"
    else:
        command += f" OUTPUT_DIR {models_dir} MODEL.RETINANET.NUM_CLASSES {num_classes}" + \
            f" MODEL.ROI_HEADS.NUM_CLASSES {num_classes}" + \
                f" SOLVER.IMS_PER_BATCH {batch_size}"

    if args_options:
        command += f" {args_options}"
    logging.info(f'start training: {command}')

    subprocess.run(command.split(), check=True)
    monitor.write_monitor_logger(percent=get_ymir_process(stage=YmirStage.TASK, p=1.0))

    # if task done, write 100% percent log
    monitor.write_monitor_logger(percent=1.0)


def _run_mining(cfg: edict()) -> None:
    # generate data.yaml for mining

    command = 'python3 mining/mining_cald.py'
    logging.info(f'mining: {command}')
    subprocess.run(command.split(), check=True)
    monitor.write_monitor_logger(percent=1.0)


def _run_infer(cfg: edict) -> None:
    # generate data.yaml for infer

    N = dr.items_count(env.DatasetType.CANDIDATE)
    infer_result = dict()
    model = None
    idx = -1

    monitor_gap = max(1, N // 100)
    for asset_path, _ in dr.item_paths(dataset_type=env.DatasetType.CANDIDATE):
        img = cv2.imread(asset_path)
        result = model.infer(img)
        infer_result[asset_path] = result
        idx += 1

        if idx % monitor_gap == 0:
            percent = get_ymir_process(stage=YmirStage.TASK, p=idx / N)
            monitor.write_monitor_logger(percent=percent)

    rw.write_infer_result(infer_result=infer_result)
    monitor.write_monitor_logger(percent=1.0)


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout,
                        format='%(levelname)-8s: [%(asctime)s] %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        level=logging.INFO)

    os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
    sys.exit(start())
