# detectron2-ymir

- [detectron2](./README_DETECTRON2.md)
- [ymir](https://github.com/IndustryEssentials/ymir)

# ymir docker image

```
docker pull youdaoyzbx/ymir-executor:ymir1.0.0-detectron2-tmi
```

# build executor

```
docker build -t ymir/ymir-executor:ymir1.0.0-cuda111-detectron2-tmi . -f cu111.dockerfile --build-arg SERVER_MODE=dev --build-arg YMIR=1.0.0
```

## todo 
- [ ] do not support small batch size (=2) with large learning rates (>0.001). 
```
FloatPointError: Loss became infinite or NaN at iteration=902!
loss_dict = {'loss_cls': nan, 'loss_box_reg': nan}
```

## change log 

- add folder `ymir` for utils, train, infer and mining 

- modify `detectron2/engine/defaults.py default_writers` to change tensorboard logging directory

- modify `detectron2/engine/hooks.py EvalHook` to write `monitor.txt` and `result.yaml`

- modify `tools/train_net.py` to modify training configuration

- modify `detectron2/evaluation/coco_evaluation.py` to save EVAL_TMP_FILE