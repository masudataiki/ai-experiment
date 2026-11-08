import argparse
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
args = parser.parse_args()
if args.show_q and args.agent != "tableq":
    parser.error("--show-q は --agent tableq のときだけ使える")
if args.greedy_test and args.agent != "tableq":
    parser.error("--greedy-test は --agent tableq のときだけ使える")
if args.dqn_model != "mlp" and args.agent != "dqn":
    parser.error("--dqn-model は --agent dqn のときだけ使える")


# 環境と agent を用意
env_id = args.env
# env_id = 'CartPole-v0'
env = gym.make(env_id)
if args.agent == "rulebase":
    agent = agents.RulebaseAgent(env)
elif args.agent == "tableq":
    agent = agents.TableQAgent(env, greedy_test=args.greedy_test)
elif args.agent == "dqn":
    agent = agents.DQNAgent(env, model_type=args.dqn_model)
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
n_episode = {'train': 100, 'test': 100}
# 何step 経ったらepisode を強制終了するか
n_max_time = {'train': 300, 'test': 300}
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
        obs, info = env.reset()
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
