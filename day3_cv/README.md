# Day3 画像認識

## 環境
（ここに実行環境）

<!-- 演習を追加するときはこのブロックをコピーする
## 演習4.x.x タイトル
```sh
実行したコマンド
```
- 変更点: 変更したファイルと箇所（なければ「なし」）
- 結果: 数値や出力（図があればパスも）
-->

## 演習4.2.1 MLPでMNIST学習
```sh
python train_mnist_mlp.py -g 0 -o result/mlp
```
- 変更点: 
- 結果: 検証精度 xx% (図のパス)

## 演習4.2.x MLPのテスト
```sh
python test_mnist_mlp.py -g 0 -m result/mlp/model_final -i image.png
```
- 変更点: 
- 結果: 

## 演習4.2.6 CNN実装
```sh
python train_mnist_cnn.py -g 0 -o result/cnn
```
- 変更点: 
- 結果: 検証精度 xx% (図のパス)

## 演習4.x.x CIFAR-10の学習
```sh
python train_cifar10.py -g 0 -o result/cifar10 -d data/mini_cifar
```
- 変更点: 
- 結果: 検証精度 xx% (図のパス)

## 演習4.x.x CIFAR-10のテスト
```sh
python test_cifar10.py -g 0 -m result/cifar10/model_final -d data/mini_cifar
```
- 変更点: 
- 結果: テスト精度 xx%

## 演習4.3.x 深層特徴データベースの作成
```sh
python create_db.py -g 0 -d データセットのパス
```
- 変更点: 
- 結果: 

## 演習4.3.x 深層特徴検索
```sh
python search.py -g 0 -i 入力画像のパス -f result/feature.pt -p result/path.pt -k 5
```
- 変更点: 
- 結果: 
