import torch
import torch.nn as nn


class DQNModel(nn.Module):
    """
    1次元ベクトルからQ値を返すQ関数モデル。
    Args:
        in_size: 入力される次元数 (例えば、EazyMazeの場合はx, y座標の2つ)
        out_size: 出力される次元数（actionの数 例えば、EazyMazeの場合は上下左右の4つ）
        model_type: モデルの種類 ('mlp' が配布どおり。main.py の --dqn-model で選ぶ)
        hidden_sizes: 隠れ層のユニット数のリスト (個数が層の枚数。main.py の --hidden-sizes で選ぶ)
    """

    def __init__(self, in_size, out_size, model_type='mlp', hidden_sizes=(5, 5)):
        """
        初期値を代入。
        """
        super(DQNModel, self).__init__()
        self.model_type = model_type
        if model_type == 'mlp' or model_type == 'tanh':
            # 既定は配布どおりの in_size -> 5 -> 5 -> out_size (活性化関数は forward で切り替える)
            unit_sizes = [in_size] + list(hidden_sizes)
            self.hidden_layers = nn.ModuleList(
                [nn.Linear(unit_sizes[i], unit_sizes[i + 1]) for i in range(len(unit_sizes) - 1)])
            self.l_out = nn.Linear(unit_sizes[-1], out_size)
        elif model_type == 'linear':
            self.W = nn.Linear(in_size,out_size)
        else:
            raise ValueError("unknown model_type: {}".format(model_type))

    def forward(self, x):
        """
        順伝搬の処理を行う
        """
        h = x
        if self.model_type == 'linear':
            h = self.W(h)
        else:
            activation = torch.relu if self.model_type == 'mlp' else torch.tanh
            for layer in self.hidden_layers:
                h = activation(layer(h))
            h = self.l_out(h)
        return h
