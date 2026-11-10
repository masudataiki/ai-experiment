# Day4 音声処理

## 環境
| 環境 | 内容 | 使った演習 |
|---|---|---|
| Mac（CPU） | MacBook Air（Apple M1）、macOS 15.5、Python 3.12.16、PyTorch 2.9.1、NumPy 2.5.4、SciPy 1.18.1、pyworld 0.3.5、pysptk 1.0.1、dtw 1.4.1（pollen-robotics 版） | 5.4.4〜5.4.7 の手順、演習5.4.1〜5.4.4 のすべて |

- **Colab は使わず、すべて手元の Mac（CPU）で実行した。** 実験書は Colab（[vc_exercise.ipynb](vc_exercise.ipynb)）を前提としているが、ノートブックのセルと同じ内容の `src/*.py` を `day4_speech/` で実行した。学習も CPU（`-g -1`）で行った。
- パッケージは `.venv/bin/python -m pip install pyworld pysptk "git+https://github.com/pollen-robotics/dtw"` で入れた。pyworld と pysptk は `pkg_resources` を使うため、setuptools は 81 未満（80.10.2）にしている。一覧はリポジトリ直下の `requirements.txt` を参照。
- 配布コードはスクリプトの上から順に処理が書かれていて、関数やクラスは使っていない（`train.py` の `VCDNN`, `get_dataset`, `main` だけが例外）。そのため、変更箇所は「どのスクリプトの、どの処理か」で示す。
- 聴き比べでは、①元の SF（`data/SF/wav/`）、②変換した音声（`result/wav/`）、③本物の TF（`data/TF/wav/`）を同じ文（主に atr503_a46）で比べた。

<!-- 演習を追加するときはこのブロックをコピーする
## 演習5.4.x タイトル
```sh
実行したコマンド
```
- 変更点: 変更したファイルと箇所（なければ「なし」）。変更したコードを数行貼る
- 結果: 数値や出力（音声や図があればパスも）
-->

## 前提の手順（5.4.4〜5.4.7：特徴量の分析 → 時間アラインメント → 学習 → 変換）
```sh
cd day4_speech
../.venv/bin/python -u src/make_acoustic_features.py   # 5.4.4 WORLD で mgc / f0 / ap を抽出
../.venv/bin/python -u src/frame_alignment.py          # 5.4.5 DTW で SF と TF のフレームを対応づけ、conf/train.list・eval.list を作成
../.venv/bin/python -u src/train.py -g -1              # 5.4.6 VCDNN を 50 エポック学習（CPU）
../.venv/bin/python -u src/conversion.py               # 5.4.7 評価用5文の mgc を変換して合成
```
- 実行環境: Mac（CPU）。演習5.4.1 の `convert_f0.py` まで続けて実行し、合計 5分16.6秒（`time` の total）。ログは `result/f0/run.log`
- 変更点: `src/make_acoustic_features.py` の、wav ごとに WORLD 分析をするループの中。NumPy 2 で `np.float` が削除されて動かないため、`float` に変えた（ノートブックと同じ書き方）。
  ```python
  fs, data = wavfile.read("data/{}/wav/{}".format(s,wf))
  data = data.astype(float)   # 元は data.astype(np.float)
  ```
- 結果:
  - 学習用 45 文、評価用 5 文（atr503_a46〜a50）
  - 学習の loss: 1 エポック目 0.2792 → 50 エポック目 0.0408
  - 出力: `model/vcmodel.model`、`data/SF-TF/mgc/*.mgc`（変換後の mgc）、`result/wav/atr503_a46.wav` 〜 `a50.wav`

## 演習5.4.1 基本周波数（F0）の変換
```sh
cd day4_speech
../.venv/bin/python -u src/convert_f0.py
```
- 実行環境: Mac（CPU）
- 変更点: なし（配布の `src/convert_f0.py` をそのまま実行）。中心となるのは、評価用の各文の F0 を変換するループの中の次の1行で、有声フレーム（F0 > 0）の log₂F0 を、SF の平均・標準偏差から TF の平均・標準偏差に合わせる（5.3.2節）。
  ```python
  f0[f0 > 0.0] = np.power(2, (np.log2(f0[f0 > 0.0]) - src_lf0_mean) * tgt_lf0_std / src_lf0_std + tgt_lf0_mean)
  ```
- 出力: `data/SF-TF/f0/*.f0`（変換後の F0）、`result/wav/atr503_a46_f0.wav` 〜 `a50_f0.wav`（mgc・ap は SF のまま、F0 だけ変換）
- 結果（聴き比べ）: ②（F0変換）は①（元のSF）と抑揚は同じで、声の高さだけ少し変わったと分かる程度。大きな変化はなく、③（TF）に近づいた感じはしない（両話者とも女性のため変化が小さいと考えられる）

## 演習5.4.2 声道特徴と基本周波数の両方の変換
```sh
cd day4_speech
../.venv/bin/python -u src/convert_both.py
```
- 実行環境: Mac（CPU）。前提として `conversion.py`（変換後の mgc）と `convert_f0.py`（変換後の F0）を実行済み
- 変更点: `src/convert_both.py` を新規作成した。`convert_f0.py` をコピーし、評価用の各文を WORLD で合成するループの中で、mgc の読み込み元と出力名を変えた。
  ```python
  outfile = "result/wav/{}_both.wav".format(datalist[i])                 # 元は {}_f0.wav
  with open("data/SF-TF/mgc/{}.mgc".format(datalist[i]), "rb") as f:     # 元は data/SF/mgc（SF の元の mgc）
      mgc = np.fromfile(f, dtype="<f8", sep="")
  with open("data/SF-TF/f0/{}.f0".format(datalist[i]), "rb") as f:       # F0 は元から変換後のものを読んでいる
      f0 = np.fromfile(f, dtype="<f8", sep="")
  ```
- 出力: `result/wav/atr503_a46_both.wav` 〜 `a50_both.wav`（mgc・F0 は変換後、ap は SF のまま）
- 結果（聴き比べ）: ②（両方変換）は①（元のSF）より音質が悪く感じた。声の高さも声色も少し変わった気がするが、③（TF）に近づいた感じはしなかった

## 演習5.4.3 差分スペクトル法
```sh
cd day4_speech
../.venv/bin/python -u src/convert_differential.py
```
- 実行環境: Mac（CPU）。前提として `conversion.py`（変換後の mgc）を実行済み
- 変更点: `src/convert_differential.py` の、評価用の各文に差分フィルタをかけるループの中の2か所。
  - 差分 `conv_mgc − src_mgc` を計算した直後に、それを 0 で上書きする行があり、フィルタが何も変えないものになっていた。この行をコメントアウトした。
  - NumPy 2 対応のため、`np.float` を `float` に変えた。
  ```python
  fs, data = wavfile.read("data/SF/wav/{}.wav".format(datalist[i]))  # 入力音声そのもの
  data = data.astype(float)                                          # 元は data.astype(np.float)

  diff_mgc = conv_mgc - src_mgc  # 差分のフィルタ（変換後の mgc − SF の mgc）
  #diff_mgc = np.zeros(shape=conv_mgc.shape)   # コメントアウトした行

  b = np.apply_along_axis(sptk.mc2b, 1, diff_mgc, alpha)
  synthesizer = Synthesizer(MLSADF(order=dim-1, alpha=alpha), 80)
  owav = synthesizer.synthesis(data, b)   # 入力音声に差分フィルタをかける
  ```
- 出力: `result/wav/atr503_a46_diff.wav` 〜 `a50_diff.wav`（SF の入力音声に差分フィルタを適用。F0 は SF のまま）
- 結果（聴き比べ）: ②（差分スペクトル法）は声の高さは①（元のSF）のまま。声色は①とも③（TF）ともどちらとも言えない微妙な感じ。音質は5.4.2（両方変換）より向上したと感じた

## 演習5.4.4 F0 変換音声への差分フィルタの適用
```sh
cd day4_speech
../.venv/bin/python -u src/convert_differential_f0.py
```
- 実行環境: Mac（CPU）。前提として `convert_f0.py`（F0 だけ変換した音声）を実行済み
- 変更点: `src/convert_differential_f0.py` を新規作成した。`convert_differential.py` をコピーし、評価用の各文に差分フィルタをかけるループの中で、出力名と、差分フィルタをかける入力音声を変えた。入力音声を元の SF の録音から、演習5.4.1 で作った F0 だけ変換した音声に替えている。
  ```python
  outfile = "result/wav/{}_diff_f0.wav".format(datalist[i])            # 元は {}_diff.wav
  ...
  fs, data = wavfile.read("result/wav/{}_f0.wav".format(datalist[i]))  # 元は data/SF/wav/{}.wav
  ```
- 出力: `result/wav/atr503_a46_diff_f0.wav` 〜 `a50_diff_f0.wav`
- 結果（聴き比べ）: ②（F0変換＋差分フィルタ）は声の高さが③（TF）に近く、5.4.3 の _diff.wav より③に近くなった。音質は耳を澄ますと粗があるように聞こえる。声色は5.4.3と同様に①（元のSF）とも③とも言えない微妙な感じ
