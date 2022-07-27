# detectron2-ymir

- [detectron2](./README_DETECTRON2.md)
- [ymir](https://github.com/IndustryEssentials/ymir)

# build executor

```
docker build -t ymir/ymir-executor:ymir1.2.0-cuda111-detectron2-tmi . -f cu111.dockerfile --build-arg SERVER_MODE=dev
```

## todo 
- [ ] do not support small batch size (=2) with large learning rates (>0.001). 
```
FloatPointError: Loss became infinite or NaN at iteration=902!
loss_dict = {'loss_cls': nan, 'loss_box_reg': nan}
```

- [ ] do not support multi-stage ymir models

- [ ] save model weight and config file