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

## 演習5.4.2 声道特徴と基本周波数の両方の変換
```sh
cd day4_speech
../.venv/bin/python -u src/convert_both.py
```
- 実行環境: Mac (arm64, CPU)。前提として演習5.4.1 の欄のコマンド（conversion.py, convert_f0.py）を実行済み
- 変更点: src/convert_both.py を新規作成（convert_f0.py をコピーし、mgc の読み込み元を data/SF/mgc → data/SF-TF/mgc（conversion.py が保存した変換済み mgc）、出力名を {}_both.wav に変更）
- 出力: result/wav/atr503_a46_both.wav 〜 atr503_a50_both.wav（mgc・F0 は変換後、ap は SF のまま）
- 結果（聴き比べ）: ②（両方変換）は①（元のSF）より音質が悪く感じた。声の高さも声色も少し変わった気がするが、③（TF）に近づいた感じはしなかった

## 演習5.4.3 差分スペクトル法
```sh
cd day4_speech
../.venv/bin/python -u src/convert_differential.py
```
- 実行環境: Mac (arm64, CPU)。前提として演習5.4.1 の欄のコマンド（conversion.py まで）を実行済み
- 変更点: src/convert_differential.py
  - 29行目 `np.float` → `float`（NumPy 2 対応）
  - 32行目 `diff_mgc = np.zeros(...)` をコメントアウト（31行目の差分 conv_mgc − src_mgc が 0 で上書きされ、無変換になっていたため）
- 出力: result/wav/atr503_a46_diff.wav 〜 atr503_a50_diff.wav（SF の入力音声に差分フィルタを適用。F0 は SF のまま）
- 結果（聴き比べ）: ②（差分スペクトル法）は声の高さは①（元のSF）のまま。声色は①とも③（TF）ともどちらとも言えない微妙な感じ。音質は5.4.2（両方変換）より向上したと感じた
