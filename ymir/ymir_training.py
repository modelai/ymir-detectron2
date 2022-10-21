import glob
import logging
import os
import os.path as osp
import shutil
import subprocess
import sys
from easydict import EasyDict as edict
from ymir.utils import (
    convert_ymir_to_coco,
    write_ymir_training_result,
)
from ymir_exc import monitor
from ymir_exc.util import YmirStage, get_merged_config, write_ymir_monitor_process


def main(cfg: edict) -> int:
    # convert ymir dataset to coco format
    convert_ymir_to_coco(cfg)
    write_ymir_monitor_process(cfg, task='training', naive_stage_percent=1.0, stage=YmirStage.PREPROCESS)

    config_file = cfg.param.config_file
    num_gpus = len(cfg.param.gpu_id.split(','))
    args_options = cfg.param.get('args_options', '')
    command = f'python3 tools/train_net.py --config-file {config_file}' + \
        f' --num-gpus {num_gpus}'

    if args_options:
        command += f" {args_options}"
    logging.info(f'start training: {command}')

    subprocess.run(command.split(), check=True)
    write_ymir_monitor_process(cfg, task='training', naive_stage_percent=1.0, stage=YmirStage.TASK)

    write_ymir_training_result(last=True)
    # if task done, write 100% percent log
    monitor.write_monitor_logger(percent=1.0)

    return 0


if __name__ == '__main__':
    cfg = get_merged_config()
    os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
    os.environ.setdefault('EVAL_TMP_FILE', osp.join(cfg.ymir.output.models_dir, 'eval_tmp.json'))
    os.environ.setdefault('YMIR_MODELS_DIR', cfg.ymir.output.models_dir)
    os.environ.setdefault('TENSORBOARD_DIR', cfg.ymir.output.tensorboard_dir)
    sys.exit(main(cfg))
