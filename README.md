# Batched Kronecker Product Test
cupyで性能差が見られなかったので生CUDAで書いてみようと

## ファイル
- kp.py : BatchedKroneckerProductのPythonのコード 
- main.cu : BatchedKroneckerProductの生CUDAのコード


## 実験
### 環境
- GF GTX 1080

### 結果
#### バッチ処理
```
行列サイズ : 100 x 100
バッチサイズ : 5000
計算回数 : 100
計算時間 : 34.81[ms]
```
#### forループ
```
行列サイズ : 100 x 100
バッチサイズ : 5000
計算回数 : 100
計算時間 : 46.97[ms]
```

だいたいこんな感じ  
生だとバッチ処理が期待通り速いっぽい

### 高速化率
[グラフ](./speedup.pdf)

## 思ったこと
- 計算する行列のサイズが小さいと沢山並列できるのかな
- streamの上限はGPUごとに違いそう
