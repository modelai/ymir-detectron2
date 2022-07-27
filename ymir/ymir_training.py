import os
import sys
import subprocess
import logging

from ymir.utils import (get_merged_config, 
    get_ymir_process, YmirStage, convert_ymir_to_coco, 
    write_ymir_training_result)
from ymir_exc import monitor

def main() -> int:
    cfg = get_merged_config()

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

    write_ymir_training_result(last=True)
    # if task done, write 100% percent log
    monitor.write_monitor_logger(percent=1.0)

    return 0

if __name__ == '__main__':
    sys.exit(main())
