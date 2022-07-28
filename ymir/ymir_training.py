import glob
import logging
import os
import os.path as osp
import shutil
import subprocess
import sys
from easydict import EasyDict as edict
from ymir.utils import (
    YmirStage,
    convert_ymir_to_coco,
    get_merged_config,
    get_ymir_process,
    write_ymir_training_result,
)
from ymir_exc import monitor


def main(cfg: edict) -> int:
    # convert ymir dataset to coco format
    convert_ymir_to_coco(cfg)
    monitor.write_monitor_logger(percent=get_ymir_process(stage=YmirStage.PREPROCESS, p=1.0))

    config_file = cfg.param.config_file
    num_gpus = len(cfg.param.gpu_id.split(','))
    models_dir = cfg.ymir.output.models_dir
    args_options = cfg.param.get('args_options','')
    num_classes = len(cfg.param.class_names)
    batch_size = int(cfg.param.batch_size)
    command = f'python3 tools/train_net.py --config-file {config_file}' + \
        f' --num-gpus {num_gpus}'

    if args_options:
        command += f" {args_options}"
    logging.info(f'start training: {command}')

    subprocess.run(command.split(), check=True)
    monitor.write_monitor_logger(percent=get_ymir_process(stage=YmirStage.TASK, p=1.0))

    # copy tensorboard log to ymir tensorboard directory
    tensorboard_log_files = glob.glob(osp.join(models_dir,'events.out.tfevents.*'))
    for log_file in tensorboard_log_files:
        shutil.copy(log_file, cfg.ymir.output.tensorboard_dir, follow_symlinks=True)

    write_ymir_training_result(last=True)
    # if task done, write 100% percent log
    monitor.write_monitor_logger(percent=1.0)

    return 0

if __name__ == '__main__':
    cfg = get_merged_config()
    os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')
    os.environ.setdefault('EVAL_TMP_FILE', osp.join(cfg.ymir.output.models_dir, 'eval_tmp.json'))
    os.environ.setdefault('YMIR_MODELS_DIR', cfg.ymir.output.models_dir)
    sys.exit(main(cfg))
