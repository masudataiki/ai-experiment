import torch
import torch.nn as nn


class DQNModel(nn.Module):
    """
    1次元ベクトルからQ値を返すQ関数モデル。
    Args:
        in_size: 入力される次元数 (例えば、EazyMazeの場合はx, y座標の2つ)
        out_size: 出力される次元数（actionの数 例えば、EazyMazeの場合は上下左右の4つ）
        model_type: モデルの種類 ('mlp' が配布どおり。main.py の --dqn-model で選ぶ)
    """

    def __init__(self, in_size, out_size, model_type='mlp'):
        """
        初期値を代入。
        """
        super(DQNModel, self).__init__()
        self.model_type = model_type
        if model_type == 'mlp':
            # 配布どおり: in_size -> 5 -> 5 -> out_size, 活性化関数 ReLU
            unit_sizes = [in_size, 5, 5]
            self.l_1 = nn.Linear(unit_sizes[0], unit_sizes[1])
            self.l_2 = nn.Linear(unit_sizes[1], unit_sizes[2])
            self.l_out = nn.Linear(unit_sizes[2], out_size)
        elif model_type == 'linear':
            self.W = nn.Linear(in_size,out_size)
        elif model_type == 'tanh':
            unit_sizes = [in_size, 5, 5]
            self.l_1 = nn.Linear(unit_sizes[0], unit_sizes[1])
            self.l_2 = nn.Linear(unit_sizes[1], unit_sizes[2])
            self.l_out = nn.Linear(unit_sizes[2], out_size)
        else:
            raise ValueError("unknown model_type: {}".format(model_type))

    def forward(self, x):
        """
        順伝搬の処理を行う
        """
        h = x
        if self.model_type == 'mlp':
            h = self.l_1(h)
            h = torch.relu(h)
            h = self.l_2(h)
            h = torch.relu(h)
            h = self.l_out(h)
        elif self.model_type == 'linear':
            h = self.W(h)
        elif self.model_type == 'tanh':
            h = self.l_1(h)
            h = torch.tanh(h)
            h = self.l_2(h)
            h = torch.tanh(h)
            h = self.l_out(h)
        return h
