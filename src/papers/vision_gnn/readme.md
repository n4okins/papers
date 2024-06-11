# please get the model from [here](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/vig_pytorch)

### evaluate.pyは、VisionGNNの実装を見るための、ImageNetのvalで評価するだけの単体ファイルです。

`models/`下に、VisionGNNのモデルをダウンロードして、`evaluate.py`を実行してください。

## example:
### ViG-S
```
python evaluate.py sample_figs/Data/CLS-LOC/ --model_type vig --model_path ./models/vig_s_80.6.pth
```
あるいは
```
python evaluate.py /path/to/ImageNet/ --model_type vig --model_path ./models/vig_s_80.6.pth
```


### PyramidViG-S
```
python evaluate.py sample_figs/Data/CLS-LOC/ --model_type pvig --model_path ./models/pvig_s_82.1.pth
```
あるいは
```
python evaluate.py /path/to/ImageNet/ --model_type pvig --model_path ./models/pvig_s_82.1.pth
```

など。