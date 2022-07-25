# ymir change log

## dataset
```
# register ymir_dataset train/val in detectron2
convert_ymir_to_coco(ymir_cfg)
register_coco_instances("ymir_dataset_train", {}, "/out/ymir_dataset/ymir_train.json", "/in")
register_coco_instances("ymir_dataset_val", {}, "/out/ymir_dataset/ymir_val.json", "/in")
```

## tensorboard

```
trainer.register_hooks(
    [
        hooks.IterationTimer(),
        hooks.PeriodicWriter([CommonMetricPrinter(max_iter)]),
        hooks.TorchProfiler(
            lambda trainer: trainer.iter == max_iter - 1, cfg.OUTPUT_DIR, save_tensorboard=True
        ),
    ]
)
```

## args_options

```
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
```

## learning_rate

- learning_rate too large may cause training NAN problem

[FloatingPointError: Loss became infinite or NaN at iteration=556](https://github.com/facebookresearch/detectron2/issues/550)
