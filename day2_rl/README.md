# Day2 強化学習

配布コードの説明は [README_original.md](README_original.md) を参照。
演習の解答（文章で答えるもの）は [answer_sheet.md](answer_sheet.md) に記入。

## 環境
macOS (arm64)、CPU のみ、Python 3.12.16。gymnasium 1.4.0、torch 2.9.1、numpy 2.5.4、six 1.17.0。
コマンドは `day2_rl` フォルダで実行する。

<!-- 演習を追加するときはこのブロックをコピーする
## 演習3.4.x タイトル
```sh
実行したコマンド
```
- 変更点: 変更したファイルと箇所（なければ「なし」）
- 結果: 数値や出力（図があればパスも）
-->

## 演習3.4.1 準備
```sh
pip install gymnasium six
python main.py | less
```
- 変更点: なし
- 結果: ランダムに動くエージェントの迷路が、step ごとに表示されることを確認した

また、`random_agent.py`と`easymaze_env.py`が配布ファイルのどこにあるのか、以下のように確認した。
  - `random_agent.py`: `agents/random_agent.py`
  - `easymaze_env.py`: `gym_easymaze/envs/easymaze_env.py`

## 演習3.4.2 相互作用の理解
- 変更点: なし（main.py を読んで確認した）
- 結果: 解答は [answer_sheet.md](answer_sheet.md) の「演習 3.4.2」に記入

## 演習3.4.3 ランダムエージェントの平均step数
```sh
python main.py --quiet
```
- 変更点: main.py
  - `--quiet` を追加: 毎 step の迷路の出力を止める（引数なしなら従来どおり表示する）
  - 平均step数の出力を追加: エピソードごとのstep数を合計し、終わったエピソード数で割って `T:` に表示する
  - 300 steps でゴールできなかったエピソードは、300 steps として平均に入る
- 結果: train 56.76 steps / test 50.63 steps。解答は answer_sheet.md の「演習 3.4.3」にも記入
  - 100 episodes の平均（`episode: 100` の行の `T:`）
  - 乱数シードを固定していないので、実行ごとに値は変わる

## 演習3.4.4 ルールベースエージェント
```sh
python main.py --agent rulebase --quiet
```
- 変更点:
  - agents/rulebase_agent.py の act() を実装: 一番下の行（`y = 2`）に着くまで下、着いたら右に進む
  - main.py に `--agent` を追加: 使うエージェントを切り替える（引数なしなら従来どおり random）
- 結果: train 5.0 steps / test 5.0 steps
  - スタートからゴールまでの最短 step 数（下に2マス、右に3マス）と一致した
  - ルールに乱数がないので、実行ごとに値は変わらない

## 演習3.4.5 ルールベースエージェントがゴールできない迷路
```sh
python main.py --env EasyMazeModified-v0 --agent rulebase --quiet
python main.py --env EasyMazeModified-v0 --agent tableq --quiet
```
- 変更点:
  - gym_easymaze/envs/easymaze_env.py: `(y, x) = (2, 2)` を通れないマスにした迷路を追加（`variant='modified'` のときだけ使う）
  - gym_easymaze/__init__.py: 変更後の迷路を `EasyMazeModified-v0` として登録
  - main.py に `--env` を追加（引数なしなら従来どおり元の迷路 `EasyMaze-v0`）
- 結果:

| エージェント | train | test | 平均 reward |
|---|---|---|---|
| ルールベース | 300.0 steps | 300.0 steps | 0.0 |
| Table-Q（別のルートでのゴールの確認） | 8.05 steps | 7.47 steps | 10.0 |

  - ルールベースは `(2, 1)` で止まり、全エピソードが 300 steps で打ち切られる
  - Table-Q は上の行を右に進み、右端を下りる別のルートでゴールできる

## 演習3.4.6 Table-Q エージェントの実装
```sh
python main.py --agent tableq --quiet
```
- 変更点:
  - agents/table_q_agent.py の穴を埋めた
    - train(): 次の状態のQ値の最大値 `max_q` を求め、式(3.3)でQ値を更新する
    - epsilon_greedy(): 確率 ε でランダムな行動、確率 1-ε でQ値が最大の行動を選ぶ
  - main.py の `--agent` に `tableq` を追加（引数なしなら従来どおり random）
- 結果: train 8.77 steps / test 7.18 steps
  - 乱数シードを固定していないので、実行ごとに値は変わる

## 演習3.4.7 Table-Q エージェントの平均step数
```sh
python main.py --agent tableq --quiet
```
- 変更点: main.py
  - train の最後に、最初と最後の10エピソードの平均step数の出力を追加
- 結果: 解答は [answer_sheet.md](answer_sheet.md) の「演習 3.4.7」にも記入

| train | test | train 最初の10エピソード | train 最後の10エピソード |
|---|---|---|---|
| 6.97 steps | 6.81 steps | 8.4 steps | 6.2 steps |

- 乱数シードを固定していないので、実行ごとに値は変わる

## 演習3.4.8 Q値の観察
```sh
python main.py --agent tableq --quiet --show-q
```
- 変更点: main.py
  - `--show-q` を追加: train 中に10エピソードごとに `q_table_to_str()` の結果を表示する（`--agent tableq` のときだけ使える）

- 迷路: 元の迷路（`EasyMaze-v0`、実験書の図3.5）

```text
    x=0 x=1 x=2 x=3
y=0  S   .   .   .
y=1  .   .   #   .
y=2  .   .   .   G
```

  - `S`: スタート `(y, x) = (0, 0)`、`G`: ゴール `(y, x) = (2, 3)`、`#`: 通れないマス、`.`: 通れるマス
  - 報酬はゴールに着いたときの 10 だけ
  - 行動は上下左右の4つで、壁や `#` に向かって動くとその場にとどまる

- 結果: Q値の変化

| エピソード | `(1, 3)` の「下」（ゴールの直前） | `(0, 3)` の「下」 | `(0, 0)` の「右」（スタート） |
|---|---|---|---|
| 10 | 6.13 | 2.14 | 0.01 |
| 100 | 10.00 | 9.50 | 7.92 |

- 観察
  - ゴール付近のQ値から先に増え、学習が進むとスタート側のQ値も増えていく（式(3.3)の更新で、γ を掛けた値が手前のマスに伝わるため）
  - ゴールに近いマスほど、ゴールへ向かう行動のQ値が大きくなる

## 演習3.4.9 test 時の greedy 行動
```sh
python main.py --agent tableq --quiet --greedy-test
```
- 変更点:
  - agents/table_q_agent.py の act(): `greedy_test` が真なら、Q値が最大の行動を選ぶ（train は ε-greedy のまま）
    - train で訪れなかった状態では、ランダムに動く
  - main.py に `--greedy-test` を追加（引数なしなら従来どおり test も ε-greedy）
- 結果: 解答は [answer_sheet.md](answer_sheet.md) の「演習 3.4.9」にも記入

| train | test |
|---|---|
| 7.34 steps | 5.0 steps |

- 比較
  - test は最短の 5 steps で、ランダムな遠回りがなくなった
  - train は確率 ε = 0.3 でランダムに動くことと、学習の初期にQ値が定まっていないことから、step 数が多くなる
  - 乱数シードを固定していないので、train の値は実行ごとに変わる

## 演習3.4.10 DQN エージェント
```sh
python main.py --agent dqn --quiet
python main.py --agent dqn --dqn-model linear --quiet
python main.py --agent dqn --dqn-model tanh --quiet
```
- 変更点:
  - main.py の `--agent` に `dqn` を追加（引数なしなら従来どおり random）
  - main.py に `--dqn-model` を追加: DQN の Q 関数モデルを切り替える（引数なしなら配布どおりの `mlp`、`--agent dqn` のときだけ使える）
  - agents/dqn_agent.py: `model_type` を受け取り、`policy_net` と `target_net` の両方に渡す
  - agents/models/dqn_model.py: `model_type` でモデルを分岐
    - `mlp`: 配布どおり（7→5→5→4、ReLU）
    - `linear`: 線形回帰（7→4 の `nn.Linear` 1 層、活性化関数なし）
    - `tanh`: `mlp` と同じ構成で、活性化関数を tanh に変えた
- 結果:

| モデル | train | test | train 最初の10エピソード | train 最後の10エピソード |
|---|---|---|---|---|
| mlp | 13.51 steps | 5.0 steps | 33.5 steps | 9.7 steps |
| linear | 15.9 steps | 5.0 steps | 94.8 steps | 6.5 steps |
| tanh | 18.83 steps | 300.0 steps | 71.4 steps | 12.4 steps |

  - mlp と linear は test で最短の 5 steps でゴールできた。tanh はこの実行ではゴールできず、300 steps で打ち切られた
  - 乱数シードを固定していないので、実行ごとに値は変わる（test が 300 steps で、ゴールできないこともある）

## 演習3.4.11 CartPole
```sh
python main.py --agent dqn --env CartPole-v0 --quiet
```
- 変更点: main.py の `--env` に `CartPole-v0` を追加（引数なしなら従来どおり `EasyMaze-v0`）
  - agents/dqn_agent.py と agents/models/dqn_model.py は変更していない
- 結果:

| train | test | train 最初の10エピソード | train 最後の10エピソード |
|---|---|---|---|
| 24.18 点 | 50.92 点 | 10.4 点 | 37.4 点 |

  - 各 step の報酬が 1 なので、step 数と点数は同じ値になる
  - 点数の最大値は 200 で、学習で点数は上がったが、200 点には届かなかった
  - 乱数シードを固定していないので、実行ごとに値は変わる

## 演習3.4.12 CartPole の高得点化

### 目的
配布の DQN（3.4.11 と同じ設定）は、CartPole で test 平均 30.92 点（シード 0, 1, 2 の 3 回の平均。最大 200 点）だった。設定を変えて点数を上げることを目指した。

### 方法
DQN の結果は乱数によって大きく変わるため、各設定をシード 0, 1, 2 で 3 回実行し、test の平均で比較した。次の 3 つを順に変えた。

- B：train のエピソード数を 100 → 500。3.4.11 では train の最初の 10 エピソード 10.4 点から最後の 10 エピソード 37.4 点へ点数が上がっている途中で学習が終わっており、学習量が足りないと考えた
- C：B に加えて、隠れ層を 5×2 → 64×2 にした。5 ユニットでは Q 関数を表しきれないと考えた。64×2 で学習率 0.01 のままでは点数が低かったため（1 回の試行で 35.65 点）、学習率を 0.001 に下げた
- D：C に加えて、ε を 0.3 から 0.01 まで 250 エピソードかけて減らした。学習後半はランダムな行動を減らす方がよいと考えた

実行したコマンドは次のとおり（各設定をシード 0, 1, 2 で実行。結果は results.csv に記録される）。

```sh
# A（3.4.11 と同じ設定）
for s in 0 1 2; do python main.py --agent dqn --env CartPole-v0 --quiet --seed $s; done
# B
for s in 0 1 2; do python main.py --agent dqn --env CartPole-v0 --quiet --n-train-episodes 500 --seed $s; done
# C
for s in 0 1 2; do python main.py --agent dqn --env CartPole-v0 --quiet --n-train-episodes 500 --hidden-sizes 64 64 --lr 0.001 --seed $s; done
# D
for s in 0 1 2; do python main.py --agent dqn --env CartPole-v0 --quiet --n-train-episodes 500 --hidden-sizes 64 64 --lr 0.001 --eps-decay 250 --seed $s; done
```

C の学習率を決めるときの 1 回の試行（35.65 点）は、シードを固定せずに次のコマンドで実行した。

```sh
python main.py --agent dqn --env CartPole-v0 --quiet --n-train-episodes 500 --hidden-sizes 64 64
```

### 結果
| 設定 | test（シード 0 / 1 / 2） | test 平均 | train 平均 |
|---|---|---|---|
| A（3.4.11 と同じ） | 49.02 / 34.48 / 9.26 | 30.92 | 14.79 |
| B（エピソード 500） | 55.85 / 200.0 / 97.78 | 117.88 | 93.79 |
| C（B ＋ 64×2・学習率 0.001） | 86.07 / 96.11 / 135.24 | 105.81 | 66.39 |
| D（C ＋ ε を減らす） | 116.64 / 90.16 / 102.05 | 102.95 | 114.43 |

### 考察
- エピソード数を増やすと点数は大きく上がった。A の最高（49.02 点）より B の最低（55.85 点）の方が高く、100 エピソードでは学習量が足りていなかった
- 隠れ層を大きくしても点数は上がらなかった。64×2 で学習率 0.01 のままでは点数が低かった（1 回の試行で 35.65 点）ため、学習率を 0.001 に下げて比べたが、C の平均（105.81 点）は B（117.88 点）を上回らなかった。この差（約 12 点）は、同じ設定でもシードによって出る差（B では 55.85〜200.0 点）より小さく、はっきりした差とは言えない
- ε を減らすと train の平均は上がったが（66.39 → 114.43 点）、test は上がらなかった。train では ε の確率でランダムに行動するため、ε を減らせば train の点数は上がる。一方、greedy に行動する test が変わらなかったことから、学習された行動の質は変わらなかったと考えられる

### 結論
最も効果があったのは学習量（エピソード数）を増やすことで、test 平均 30.92 点から 117.88 点に上がった。隠れ層を大きくすること（学習率も合わせて調整）と、ε を減らすことは効果が見られなかった。また、同じ設定でもシードによって 55.85〜200.0 点と結果が大きく異なった。

### 実装メモ
- main.py: 次の引数を追加（引数なしなら 3.4.11 までと同じ値）

| 引数 | 変えるもの | 既定値 |
|---|---|---|
| `--n-train-episodes` | train の episode 数（10 以上） | 100 |
| `--hidden-sizes` | 隠れ層のユニット数と枚数（例: `--hidden-sizes 64 64`） | 5 5 |
| `--lr` | 学習率 | 0.01 |
| `--eps` | ε-greedy の ε | 0.3 |
| `--replay-buffer-size` | リプレイバッファの大きさ | 1000000 |
| `--target-update-interval` | `target_net` を `policy_net` と同期する間隔（step 数） | 100 |
| `--eps-decay` | ε を `--eps` の値から 0.01 まで直線的に小さくする episode 数 | 指定なし（ε は固定） |
| `--seed` | `random`・NumPy・PyTorch の乱数と、train / test それぞれ最初の `env.reset` のシードを固定し、実行の最後に results.csv へ 1 行追記する | 指定なし（固定せず、CSV も書かない） |

- agents/dqn_agent.py: 上の値を引数で受け取る。`eps_decay` を指定すると、train の 1 episode が終わるごとに ε を更新する
- agents/models/dqn_model.py: `mlp` と `tanh` の隠れ層の数と大きさを `hidden_sizes` で決める
