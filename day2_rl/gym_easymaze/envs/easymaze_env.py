import gymnasium as gym
from gymnasium import error, spaces, utils
from gymnasium.utils import seeding
import copy
import numpy as np
from collections import OrderedDict


class EasyMazeEnv(gym.Env):
    """簡単な迷路の環境

    事前に形の決まっている迷路
    自分で環境を作ってみたい人のためのサンプルコード

    Actions
        directions (int): 移動方向 (右, 下, 左, 上)
    States
        maze: 迷路の全体像
        agent_pos: agentがどこにいるか
    Observes
        agent_pos
    Transitions
        actionで指定された方向にagent_posがそのまま移動
        移動先に障害物や壁があった場合は移動が起こらない
    Rewards
        Goal: 10点
    """

    # human, rgb_array, ansi の3つをサポートすることができる
    # 今回は文字列を返す ansi だけをサポート
    metadata = {'render_modes': ['ansi'], 'render_fps': 4}

    def __init__(self, render_mode=None):
        """コンストラクタ"""
        # 描画モードは環境を作るとき (gym.make) に指定する
        self.render_mode = render_mode
        # 初期設定
        self.move_directions = np.array(
            ((0, 1), (1, 0), (0, -1), (-1, 0)))  # 4方向 (y, x)
        # 迷路のかたち
        # 0は移動不可、1は移動可能
        self.initial_maze = [[1, 1, 1, 1],
                             [1, 1, 0, 1],
                             [1, 1, 1, 1]]
        self.start_pos, self.goal_pos = np.array(
            (0, 0)), np.array((2, 3))  # スタートとゴール (y, x)

        self.state = {}  # 状態 ゲーム中に変化するものはすべてstateとして扱うとよい
        self.world_size = (len(self.initial_maze), len(self.initial_maze[0]))

        # 観測空間と行動空間の計算
        # この2つは public なインタフェースとして提供される
        self.action_space = spaces.Discrete(len(self.move_directions))
        self.observation_space = spaces.Dict({'y': spaces.Discrete(self.world_size[0]),
                                              'x': spaces.Discrete(self.world_size[1])})

    def step(self, action):
        """与えられた行動を取ることでゲームの状態を1ステップ進め，観測や報酬を返す"""
        # 取る行動をベクトル表現にする
        move_direction = self.move_directions[action]

        # その行動を取ったらどこに行くかを計算
        agent_pos_next = self.state['agent_pos'] + move_direction

        # 移動先が移動可能場所か確認
        is_this_move_valid = True
        if np.any(agent_pos_next < [0, 0]) or np.any(self.world_size <= agent_pos_next):
            # 画面外なら
            is_this_move_valid = False
        elif self.state['maze'][agent_pos_next[0]][agent_pos_next[1]] == 0:
            # 障害マスなら
            is_this_move_valid = False

        # 移動可能なら実際に移動する
        if is_this_move_valid:
            self.state['agent_pos'] = agent_pos_next

        # 報酬計算
        reward = self.get_reward(self.state)

        # 観測計算
        observe = self.get_observation(self.state)

        # 終了判定
        # terminated: ゴールなどでゲームが終わったか
        # truncated: 時間切れなどで途中で打ち切られたか (この環境では打ち切らない)
        terminated = False
        if np.allclose(self.state['agent_pos'], self.goal_pos):
            terminated = True
        truncated = False

        # 追加データ
        info = {}

        # 観測, 報酬, 終了判定, 打ち切り判定, その他の任意情報 のタプルを返す
        return observe, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """ゲームを初期状態に戻す"""
        super().reset(seed=seed)  # 乱数の初期化などはスーパークラスがやってくれる
        self.state['maze'] = copy.deepcopy(self.initial_maze)
        self.state['agent_pos'] = copy.deepcopy(self.start_pos)
        # 観測, その他の任意情報 のタプルを返す
        return self.get_observation(self.state), {}

    def render(self):
        """gym.make で指定されたモードでゲームを描画する"""
        if self.render_mode == 'ansi':
            maze = self.state['maze']
            return_str = '------\n'
            # 現在の迷路の状況を描画
            return_str += self.maze_to_str(maze)
            return return_str

    def get_observation(self, state):
        pos = state['agent_pos']
        return OrderedDict(sorted([('y', int(pos[0])), ('x', int(pos[1]))]))

    def get_reward(self, state):
        # goal していたら報酬を返す
        if np.allclose(state['agent_pos'], self.goal_pos):
            return 10.0
        else:
            return 0.0

    def maze_to_str(self, maze):
        return_str = ''
        for i in range(-1, self.world_size[0] + 1):
            for j in range(-1, self.world_size[1] + 1):
                if 0 <= i < self.world_size[0] and 0 <= j < self.world_size[1]:
                    state_id = maze[i][j]
                    state_chr = ['#', '.'][state_id]
                    if np.allclose([i, j], self.state['agent_pos']):
                        state_chr = 'A'
                    elif np.allclose([i, j], self.start_pos):
                        state_chr = 'S'
                    elif np.allclose([i, j], self.goal_pos):
                        state_chr = 'G'
                else:
                    state_chr = '#'
                return_str += state_chr
            return_str += '\n'
        return return_str
