# Day4 音声処理

Colab では [vc_exercise.ipynb](vc_exercise.ipynb) を上から実行する。

## 環境
（ここに実行環境）

<!-- 演習を追加するときはこのブロックをコピーする
## 演習5.4.x タイトル
```sh
実行したコマンド（ノートブックならセル名）
```
- 変更点: 変更したファイルと箇所（なければ「なし」）
- 結果: 数値や出力（音声や図があればパスも）
-->

## 演習5.4.x 特徴量の分析
```sh
vc_exercise.ipynb「特徴量の分析」のセル
```
- 変更点: 
- 結果: 

## 演習5.4.x フレーム毎時間アラインメント
```sh
vc_exercise.ipynb「フレーム毎時間アラインメント」のセル
```
- 変更点: 
- 結果: 

## 演習5.4.x 音声変換モデルの学習
```sh
vc_exercise.ipynb「音声変換モデルの学習」のセル
```
- 変更点: 
- 結果: 

## 演習5.4.x 学習したモデルによる音声の変換
```sh
vc_exercise.ipynb「学習したモデルによる音声の変換」のセル
```
- 変更点: 
- 結果: 

## 演習5.4.1 基本周波数（F0）の変換
```sh
cd day4_speech
# 前提（5.4.4〜5.4.7）: 特徴量抽出 → DTW → 学習 → mgc 変換
../.venv/bin/python -u src/make_acoustic_features.py
../.venv/bin/python -u src/frame_alignment.py
../.venv/bin/python -u src/train.py -g -1
../.venv/bin/python -u src/conversion.py
# 演習5.4.1
../.venv/bin/python -u src/convert_f0.py
```
- 実行環境: Mac (arm64, CPU)。ログは result/f0/run.log
- 変更点: src/make_acoustic_features.py の `np.float` を `float` に変更（NumPy 2 で np.float が削除されたため。ノートブックと同じ）。convert_f0.py は変更なし
- 出力: result/wav/atr503_a46_f0.wav 〜 atr503_a50_f0.wav（声色・ap は SF のまま、F0 のみ TF の平均・標準偏差に合わせて変換）
- 結果（聴き比べ）: ②（F0変換）は①（元のSF）と抑揚は同じで、声の高さだけ少し変わったと分かる程度。大きな変化はなく、③（TF）に近づいた感じはしない（両話者とも女性のため変化が小さいと考えられる）
