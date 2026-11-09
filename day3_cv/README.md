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
- 変更点: なし
- 設定: デフォルト（epoch 20, unit 1000, batchsize 100）
- 実行環境: Google Colab GPU（Tesla T4）
- 結果: 20エポック目の精度（グラフからの読み取り値）と学習時間（`time` の real）

| train | validation | かかった時間 |
|---|---|---|
| 約 95.0% | 約 94.5% | 2分51.7秒 |

![MLPの学習曲線](figures/4.2.1_mlp_accuracy.png)

## 演習4.2.2 手書き数字画像の作成
```sh
なし（ブラウザの描画ページでマウスで作成）
```
- 変更点: なし
- 結果: 黒地に白文字で数字の 8 を描いた画像（280×280、RGBA。透明部分なし）

![手書き数字の8](my_digit.png)

## 演習4.2.3 手書き数字の認識
```sh
python test_mnist_mlp.py -g -1 -m result/mlp/model_20 -i my_digit.png
```
- 変更点: test_mnist_mlp.py の `torch.load` に `map_location='cpu'` を追加（GPU で学習したモデルを Mac の CPU で読み込むため）
- 実行環境: Mac の CPU（モデルは Colab の GPU で学習した20エポック目のもの）
- 注意: `result/mlp/model_20` は git 管理外なので、演習4.2.1 の手順で作る。乱数シードを固定していないため、学習し直すと予測が変わる可能性がある
- 結果: 

| 入力 | 正解 | 予測 | 判定 |
|---|---|---|---|
| my_digit.png | 8 | 3 | 誤り |

## 演習4.2.4 予測の上位3つの表示
```sh
python test_mnist_mlp.py -g -1 -m result/mlp/model_20 -i my_digit.png --topk 3
```
- 変更点: test_mnist_mlp.py（`--topk 3` で次を切り替える。既定の `--topk 1` は演習4.2.3 と同じ表示）
  - 予測: `torch.topk` で確信度が高い順に上位 `args.topk` 個の数字を取り出して表示する
- 実行環境: Mac の CPU（モデルは Colab の GPU で学習した20エポック目のもの）
- 結果: 

| 入力 | 正解 | 1位 | 2位 | 3位 |
|---|---|---|---|---|
| my_digit.png | 8 | 3 | 2 | 5 |

## 演習4.2.5 中間ノード数の変更
```sh
python train_mnist_mlp.py -g 0 -u 3 -f 100 -o result/mlp_u3
python train_mnist_mlp.py -g 0 -u 10000 -f 100 -o result/mlp_u10000
```
- 変更点: train_mnist_mlp.py（学習終了時に最終精度を表示する1行を追加。学習の動作は変わらない）
- 設定: epoch 20, batchsize 100, unit のみ変更（`-f 100` は途中のモデルを保存しない指定）
- 実行環境: Google Colab GPU（T4）
- 結果: 20エポック目の精度と学習時間（`time` の real）

| 中間ノード数 | train | validation | かかった時間 |
|---|---|---|---|
| 3 | 68.36% | 68.37% | 2分49.4秒 |
| 10000 | 96.29% | 94.97% | 7分23.7秒 |

- 比較: 中間ノード数 10000 は 3 より validation の精度が 26.60 ポイント高く、学習時間は約 2.6 倍だった。

## 演習4.2.6 CNN実装
```sh
python train_mnist_cnn.py -g 0 -o result/cnn
```
- 変更点:
  - network.py の `MnistCNN.__init__`（66-67行目）: 畳み込み層2つを実装
    - `conv1`: `nn.Conv2d(1, 5, 5, stride=2, padding=2)`（1 × 28 × 28 → 5 × 14 × 14）
    - `conv2`: `nn.Conv2d(5, 10, 5, stride=2, padding=2)`（5 × 14 × 14 → 10 × 7 × 7）
    - stride と padding は、カーネルサイズ 5 で縦横が半分になるよう、どちらの層も stride 2、padding 2 を選んだ
  - train_mnist_cnn.py（35-36行目）: `net = MLP(...)` をコメントアウトし、`net = MnistCNN(10)` に切り替え
  - train_mnist_cnn.py: 学習終了時に最終精度を表示する1行を追加
- 設定: デフォルト（epoch 20, batchsize 100）
- 実行環境: Google Colab GPU（Tesla T4）
- 結果: 20エポック目の精度と学習時間（`time` の real）。演習4.2.1 の MLP の精度はグラフからの読み取り値

| モデル | train | validation | かかった時間 |
|---|---|---|---|
| MLP（演習4.2.1、unit 1000） | 約 95.0% | 約 94.5% | 2分51.7秒 |
| CNN（演習4.2.6） | 96.96% | 96.80% | 2分54.4秒 |

- 比較: CNN は MLP より validation の精度が約 2.3 ポイント高く、学習時間は約 2.6 秒長かった。

![CNNの学習曲線](figures/4.2.6_cnn_accuracy.png)

## 演習4.2.7 CIFAR-10の学習
```sh
python train_cifar10.py -g 0 -o result/cifar10 -d data/mini_cifar
```
- 変更点: dataset.py の `MyCifarDataset` を穴埋め
  - `classes`: クラス名（フォルダ名）から 0〜9 の番号への辞書
  - `img_path_list`: `glob` で `cls_dir` の中の png のパスを取得
  - `cls_label`: `classes[cls]` でクラスの番号を取得
  - `__getitem__`: `self.all_data[idx]` からパスとラベルを取り出す
- 設定: デフォルト（epoch 20, batchsize 100）、データは mini_cifar（学習用 1500 枚を学習:検証 = 9:1 に分割）
- 実行環境: Google Colab GPU（Tesla T4）
- 結果: 20エポック目の精度（グラフからの読み取り値）

| train | validation |
|---|---|
| 約 12% | 約 8% |

![CIFAR-10の学習曲線](figures/4.2.7_cifar_accuracy.png)

## 演習4.2.8 CIFAR-10のテスト
```sh
python test_cifar10.py -g 0 -b 64 -m result/cifar10/model_18 -d data/mini_cifar
```
- 変更点: なし
- 設定: `-b 64`（テストは 201 枚なので、バッチサイズ 100 だと最後のバッチが1枚になり、`.squeeze()` のため `IndexError` になる。精度は1枚ずつ数えるのでバッチサイズの影響を受けない）
- 使用モデル: `model_18`（演習4.2.7 のグラフで validation が最も高いエポックのうち、最も後のもの。グラフからの読み取り）
- 実行環境: Google Colab GPU（Tesla T4）
- 結果: テスト精度 12.438%

| クラス | 精度 |
|---|---|
| airplane | 15% |
| automobile | 5% |
| bird | 0% |
| cat | 0% |
| deer | 10% |
| dog | 85% |
| frog | 0% |
| horse | 0% |
| ship | 9% |
| truck | 0% |

## 演習4.2.9 CIFAR-10 の精度向上（発展課題）
```sh
python train_cifar10.py -g 0 -o result/cifar10_exp0 -d /content/mini_cifar --seed 0 -f 100
python train_cifar10.py -g 0 -e 100 -o result/cifar10_expA -d /content/mini_cifar --seed 0 -f 200
python train_cifar10.py -g 0 --optimizer adam -o result/cifar10_expB -d /content/mini_cifar --seed 0 -f 100
python train_cifar10.py -g 0 -e 100 --optimizer adam -o result/cifar10_expB2 -d /content/mini_cifar --seed 0 -f 200
python train_cifar10.py -g 0 -e 100 --optimizer adam --augment -o result/cifar10_expF -d /content/mini_cifar --seed 0 -f 200
python train_cifar10.py -g 0 -e 100 --optimizer adam --augment --ch1 32 --ch2 64 -o result/cifar10_expC -d /content/mini_cifar --seed 0 -f 200
```
- 変更点: train_cifar10.py に次の引数と表示を追加
  - `--seed`（既定は固定しない）とエポックごとの精度表示（学習の動作は変わらない）
  - `--optimizer`（既定は `sgd` で従来どおり。`adam` で `optim.Adam(net.parameters(), lr=0.001)` に切り替える）
  - `--augment`（既定はなしで従来どおり。付けると学習データにだけ `RandomHorizontalFlip` と `RandomCrop(32, padding=4)` をかける。検証データは拡張なしのまま）
  - `--ch1`, `--ch2`（既定は 6, 16 で従来どおり。network.py の `CifarCNN` の conv1, conv2 の出力チャンネル数を変え、全結合層の入力も `ch2 × 5 × 5` に合わせる）
- 設定: epoch 20（実験A, B2, F, C は 100）, batchsize 100, seed 0, データは mini_cifar（Colab のローカルディスクにコピーして使用）。`-f` は途中のモデルを保存しない指定
- 実行環境: Google Colab GPU（Tesla T4）
- 結果: 学習時間は `time` の real

| 実験 | 変えたこと | パラメータ数 | 最終エポックの学習精度 | 検証精度の最大値（何エポック目） | 最後の20エポックの検証精度の平均 | 学習時間 |
|---|---|---|---|---|---|---|
| 0 | 変更なし（ベースライン） | 6,882 | 15.11% | 13.33%（2〜4エポック目） | — | 19.8秒 |
| A | エポック数 20→100 | 6,882 | 38.30% | 46.67%（100エポック目） | — | 1分15.0秒 |
| B | 最適化手法 SGD→Adam（lr=0.001）※以降はBを基準とする | 6,882 | 44.59% | 42.67%（16エポック目） | — | 19.9秒 |
| B2 | Bからエポック数 20→100 | 6,882 | 72.89% | 53.33%（84エポック目） | 48.07%（81〜100エポック目） | 1分13.9秒 |
| F | B2にデータ拡張（左右反転 + RandomCrop(32, padding=4)）を追加 | 6,882 | 48.00% | 58.00%（71, 76エポック目） | 54.47%（81〜100エポック目） | 1分23.7秒 |
| C | Fからチャンネル数を増やす（conv1 6→32、conv2 16→64） | 69,706 | 65.70% | 56.67%（95エポック目） | 51.23%（81〜100エポック目） | 1分28.7秒 |

実験F は 100エポック目でもまだゆるやかに上がっている（検証精度の平均は 61〜80エポック目 53.70% → 81〜100エポック目 54.47%、loss は 80エポック目 20.957 → 100エポック目 19.839）。

実験0

![実験0の学習曲線](figures/4.2.9_exp0_accuracy.png)

実験A

![実験Aの学習曲線](figures/4.2.9_expA_accuracy.png)

実験B

![実験Bの学習曲線](figures/4.2.9_expB_accuracy.png)

実験B2

![実験B2の学習曲線](figures/4.2.9_expB2_accuracy.png)

実験F

![実験Fの学習曲線](figures/4.2.9_expF_accuracy.png)

実験C

![実験Cの学習曲線](figures/4.2.9_expC_accuracy.png)

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
