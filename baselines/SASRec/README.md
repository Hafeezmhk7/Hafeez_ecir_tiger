## Train
```
python main.py --dataset=Beauty --train_dir=default --maxlen=200 --dropout_rate=0.2 --device=cuda
```

## Inference
```
python main.py --full_data --inference_only=true --dataset=Beauty --state_dict_path='models/Beauty_default/SASRec.epoch=1000.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth' --train_dir=default --maxlen=200 --device=cuda 
```

## Citation
- [Official Tensorflow Implementation](https://github.com/kang205/SASRec)
- [Adapted Pytorch Implementation](https://github.com/pmixer/SASRec.pytorch)


### Original Paper

```
@inproceedings{kang2018self,
  title={Self-attentive sequential recommendation},
  author={Kang, Wang-Cheng and McAuley, Julian},
  booktitle={2018 IEEE International Conference on Data Mining (ICDM)},
  pages={197--206},
  year={2018},
  organization={IEEE}
}
```

### Adaption
```
@software{Huang_SASRec_pytorch,
author = {Huang, Zan},
title = {PyTorch implementation for SASRec},
url = {https://github.com/pmixer/SASRec.pytorch},
year={2020}
}
```
