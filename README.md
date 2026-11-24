# 人工知能演習 基本課題・自由課題

- 基本課題: day2〜day4 の各フォルダ（下の「構成」）
- 自由課題: [free_task/](free_task/)（下の「自由課題」）

## 環境
- ローカル: macOS (arm64), Python 3.12, PyTorch 2.9.1（CPUで動作確認）
- 学習: Google Colab (GPU)
- 依存パッケージ: requirements.txt（Colab での実行時の一覧は各 result/packages.txt）

## 構成
| フォルダ | 内容 | 該当演習 |
|---|---|---|
| day2_nlp | RNN/LSTM言語モデル, 英日翻訳 | 2.7.x |
| day2_rl | 迷路のQ学習, DQN, CartPole | 3.4.x |
| day3_cv | MNIST MLP/CNN, CIFAR-10, 深層特徴検索 | 4.2.x, 4.3.x |
| day4_speech | 声質変換 | 5.4.x |
| [free_task](free_task/) | 自由課題（ノイズを足すデータ拡張の強さと、壊れ方ごとのトレードオフ） | – |

## データの準備
データは容量の都合でリポジトリに含めていない。
人工知能演習ウェブサイトの配布フォルダから、各 day フォルダの同じ位置に `data/` などを置く。
自由課題のデータ（CIFAR-10、CIFAR-10-C）は、`free_task/download_data.sh` で取得する。

## 実行方法
各フォルダの README を参照。Colab では colab/run.ipynb を上から実行する。

## 自由課題
ガウスノイズを足すデータ拡張の最大の強さ σ を 0〜0.3 の5段階に変えて、day3 の CNN を CIFAR-10 で学習した。壊れた画像（CIFAR-10-C）への強さがどう変わるかと、どの細かさの縞に敏感かを調べた。

- 説明・実行手順・結果: [free_task/README.md](free_task/README.md)
- 学習環境: Azure VM（Tesla T4）。基本課題の Colab とは別
