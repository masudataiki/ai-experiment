# 演習解答欄

## 演習 3.4.2

作成している行（配布時の main.py の行番号）
agent: 11 `agent = agents.RandomAgent(env)`
env: 10 `env = gym.make(env_id)`
env: 37 `env = gym.make(env_id, render_mode=render_mode if prints_detail[interact_mode] else None)`

呼び出されている変数・メソッド

agent:
act_and_train
act
get_statistics
stop_episode_and_train
stop_episode

env:
metadata
reset
render
step
close

### 演習 3.4.3

平均step数

train:56.76
test:50.63

## 演習 3.4.7

平均step数

train:6.97
test:6.81

train first 10:8.4 steps
train last 10:6.2 steps

## 演習 3.4.9

平均step数

train:
test:
