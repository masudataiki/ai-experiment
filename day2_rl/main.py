import argparse
import csv
import os
import random
import numpy as np
import torch
import gymnasium as gym
import gym_easymaze
import agents
from print_buffer import PrintBuffer

parser = argparse.ArgumentParser()
parser.add_argument("--quiet", action="store_true", help="毎 step の迷路の出力を止める")
parser.add_argument("--agent", choices=["random", "rulebase", "tableq","dqn"], default="random", help="使う agent")
parser.add_argument("--env", choices=["EasyMaze-v0", "EasyMazeModified-v0","CartPole-v0"], default="EasyMaze-v0", help="使う環境")
parser.add_argument("--show-q", action="store_true", help="Q 値の表を出力する (tableq のみ)")
parser.add_argument("--greedy-test", action="store_true", help="test 時に greedy で行動する (tableq のみ)")
parser.add_argument("--dqn-model", choices=["mlp", "linear", "tanh"], default="mlp", help="DQN の Q 関数モデル (dqn のみ)")
parser.add_argument("--n-train-episodes", type=int, default=100, help="train の episode 数 (10 以上)")
parser.add_argument("--lr", type=float, default=0.01, help="DQN の学習率 (dqn のみ)")
parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[5, 5], help="DQN の隠れ層のユニット数。個数が層の枚数になる (dqn の mlp, tanh のみ)")
parser.add_argument("--eps", type=float, default=0.3, help="DQN の epsilon-greedy の epsilon (dqn のみ)")
parser.add_argument("--replay-buffer-size", type=int, default=10**6, help="DQN のリプレイバッファの大きさ (dqn のみ)")
parser.add_argument("--eps-decay", type=int, default=None, help="epsilon を --eps から 0.01 まで直線的に小さくする episode 数。指定しなければ固定 (dqn のみ)")
parser.add_argument("--target-update-interval", type=int, default=100, help="target_net を policy_net と同期する間隔 (step 数、dqn のみ)")
parser.add_argument("--seed", type=int, default=None, help="乱数シードを固定し、実行の最後に results.csv へ結果を 1 行追記する")
args = parser.parse_args()
if args.show_q and args.agent != "tableq":
    parser.error("--show-q は --agent tableq のときだけ使える")
if args.greedy_test and args.agent != "tableq":
    parser.error("--greedy-test は --agent tableq のときだけ使える")
if args.dqn_model != "mlp" and args.agent != "dqn":
    parser.error("--dqn-model は --agent dqn のときだけ使える")
if args.n_train_episodes < 10:
    parser.error("--n-train-episodes は 10 以上にする")
if args.agent != "dqn" and (args.lr != 0.01 or args.hidden_sizes != [5, 5] or args.eps != 0.3 or args.replay_buffer_size != 10**6):
    parser.error("--lr, --hidden-sizes, --eps, --replay-buffer-size は --agent dqn のときだけ使える")
if args.dqn_model == "linear" and args.hidden_sizes != [5, 5]:
    parser.error("--dqn-model linear には隠れ層がないので --hidden-sizes は使えない")
if args.eps_decay is not None and args.agent != "dqn":
    parser.error("--eps-decay は --agent dqn のときだけ使える")
if args.target_update_interval != 100 and args.agent != "dqn":
    parser.error("--target-update-interval は --agent dqn のときだけ使える")
if args.target_update_interval < 1:
    parser.error("--target-update-interval は 1 以上にする")
if args.eps_decay is not None and args.eps_decay < 1:
    parser.error("--eps-decay は 1 以上にする")

# 乱数シードを固定する (agent のモデルの初期値も固定するため、agent を作る前に行う)
if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


# 環境と agent を用意
env_id = args.env
# env_id = 'CartPole-v0'
env = gym.make(env_id)
if args.agent == "rulebase":
    agent = agents.RulebaseAgent(env)
elif args.agent == "tableq":
    agent = agents.TableQAgent(env, greedy_test=args.greedy_test)
elif args.agent == "dqn":
    agent = agents.DQNAgent(env, model_type=args.dqn_model, lr=args.lr, hidden_sizes=args.hidden_sizes,
                            eps=args.eps, replay_buffer_size=args.replay_buffer_size, eps_decay=args.eps_decay,
                            target_update_interval=args.target_update_interval)
else:
    agent = agents.RandomAgent(env)
# agent = agents.DQNAgent(env)


# 描画設定
# train時・test時に各 step を描画するかどうか
prints_detail = {'train': not args.quiet, 'test': not args.quiet}
# 何 episode ごとに統計情報を出力するか
every_print_statistics = {'train': 10, 'test': 10}
# 描画モード
# human: 動画をウィンドウで出す ansi: 文字列をstrで返す
supported_render_modes = env.metadata.get('render_modes', [])
render_mode = 'human' if 'human' in supported_render_modes else 'ansi'
# 描画バッファの用意
render_buffer = PrintBuffer()

# 学習設定
# 行うepisode 数
n_episode = {'train': args.n_train_episodes, 'test': 100}
# 何step 経ったらepisode を強制終了するか
n_max_time = {'train': 300, 'test': 300}
# results.csv に書く結果
average_steps_of_mode = {}
first_ten_steps = last_ten_steps = None
# ゲームを繰り返す
for interact_mode in ['train', 'test']:  # 一周目: train, 二周目: test
    # 描画するかどうかに合わせて、描画モードを指定して環境を作り直す
    env = gym.make(env_id, render_mode=render_mode if prints_detail[interact_mode] else None)
    sum_of_all_rewards = 0.0  # episode ごとの average reward を出すために、総計を覚えておく
    sum_of_all_steps = 0
    array_sum_of_steps = [0]

    for i_episode in range(n_episode[interact_mode]):  # 各episode について
        render_buffer.prints('-------------------------------')
        render_buffer.prints('episode: {0} / {1}'.format(i_episode, n_episode[interact_mode]))
        # 環境を初期化して新たな episode を開始する
        reward, done, sum_of_rewards,sum_of_steps = 0.0, False, 0.0, 0
        # env は train と test でそれぞれ作り直すので、それぞれの最初の reset にシードを渡す
        obs, info = env.reset(seed=args.seed if i_episode == 0 else None)
        for time in range(n_max_time[interact_mode]):  # 各 step について
            if prints_detail[interact_mode] and render_mode == 'ansi':
                # 環境を描画 (human のときは reset や step の中で自動的に描画される)
                render_buffer.prints(env.render())
            # agentに観測から行動を選択してもらう
            if interact_mode == 'train':
                # 学習もしてもらう
                action = agent.act_and_train(obs, reward, done)
            else:
                action = agent.act(obs)
            # 環境に行動を与えて1ステップ進ませる
            obs, reward, terminated, truncated, info = env.step(action)
            # ゲームが終わった場合も、時間切れで打ち切られた場合も episode を終える
            done = terminated or truncated
            sum_of_rewards += reward
            sum_of_steps += 1
            render_buffer.prints(interact_mode, 'episode:', i_episode, 'T:', time,
                                 'R:', sum_of_rewards, 'statistics:', agent.get_statistics())
            if prints_detail[interact_mode]:
                render_buffer.flush()  # 表示
            else:
                render_buffer.clear()  # 表示しない
            if done:
                # ゲーム終了時(ゲームクリア) の処理
                if prints_detail[interact_mode]:
                    print('Episode finished.')
                break
        
        # ゲーム終了時(時間切れ) の処理
        if prints_detail[interact_mode] and not done:
            print('{0} steps have past, but the agent could not reach the goal.'.format(time))
        # episode が終了したことを agent に伝える
        if interact_mode == 'train':
            # 最後に学習もしてもらう
            agent.stop_episode_and_train(obs, reward, done)
        else:
            agent.stop_episode()
        sum_of_all_rewards += sum_of_rewards
        sum_of_all_steps += sum_of_steps
        array_sum_of_steps.append(array_sum_of_steps[len(array_sum_of_steps) - 1] + sum_of_steps)
        # 数 episodes に一回、統計情報を表示
        if (i_episode + 1) % every_print_statistics[interact_mode] == 0 or prints_detail[interact_mode]:
            average_rewards = sum_of_all_rewards / (i_episode + 1)
            average_steps = sum_of_all_steps / (i_episode + 1)
            print(interact_mode, 'episode:', i_episode + 1, 'T:', average_steps,
                  'R:', average_rewards, 'statistics:', agent.get_statistics())
            if interact_mode == 'train' and args.show_q == True:
                print(agent.q_table_to_str())
            if interact_mode == 'train' and i_episode == n_episode['train'] - 1:
                #ハードコーディングなのでepisode数が10以上に大きくないと動かないので注意する
                first_ten_average_steps = (array_sum_of_steps[10] - array_sum_of_steps[0]) / 10
                last_ten_average_steps = (array_sum_of_steps[len(array_sum_of_steps) - 1] - array_sum_of_steps[len(array_sum_of_steps) - 11]) / 10
                print(f'first_ten_average_steps = {first_ten_average_steps} steps\nlast_ten_average_steps = {last_ten_average_steps} steps')

    env.close()  # human モードのウィンドウなどを閉じる
    print(interact_mode, 'finished.')
    average_steps_of_mode[interact_mode] = sum_of_all_steps / n_episode[interact_mode]
    if interact_mode == 'train':
        first_ten_steps = (array_sum_of_steps[10] - array_sum_of_steps[0]) / 10
        last_ten_steps = (array_sum_of_steps[-1] - array_sum_of_steps[-11]) / 10

# --seed を指定した実行だけ、設定と結果を results.csv に 1 行追記する
if args.seed is not None:
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.csv')
    write_header = not os.path.exists(results_path)
    with open(results_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['agent', 'env', 'dqn_model', 'n_train_episodes', 'hidden_sizes', 'lr', 'eps', 'eps_decay',
                             'replay_buffer_size', 'target_update_interval', 'seed', 'train_avg', 'test_avg', 'train_first_10', 'train_last_10'])
        writer.writerow([args.agent, args.env, args.dqn_model, args.n_train_episodes,
                         ' '.join(str(s) for s in args.hidden_sizes), args.lr, args.eps,
                         args.eps_decay if args.eps_decay is not None else '', args.replay_buffer_size, args.target_update_interval, args.seed,
                         average_steps_of_mode['train'], average_steps_of_mode['test'], first_ten_steps, last_ten_steps])
