# Day2 強化学習

配布コードの説明は [README_original.md](README_original.md) を参照。
演習の解答（文章で答えるもの）は [answer_sheet.md](answer_sheet.md) に記入。

## 環境
（ここに実行環境）

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
- 結果: train 56.76 steps / test 50.63 steps 解答は answer_sheet.md の「演習 3.4.3」にも記入
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
  - 上の行を右に進み、右端を下りる別のルートでゴールできる

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

## 演習3.4.x DQNエージェント
```sh
python main.py
```
- 変更点: 
- 結果: train xx steps / test xx steps

## 演習3.4.x CartPole
```sh
python main.py
```
- 変更点: 
- 結果: 
