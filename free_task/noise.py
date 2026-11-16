import torch


def gaussian_noise(images, s, generator=None):
	"""
	画素ごとに平均 0・標準偏差 s の正規分布の乱数を足し、0〜1 に収める（PLAN.md 4.2 節）

	Parameters
	----------
	images	: torch.Tensor
		0〜1 の画像（batchsize * channel * height * width）
	s	: float
		ノイズの標準偏差（0〜1 の画素値に対する値）
	generator	: torch.Generator
		ノイズ専用の乱数生成器（None なら PyTorch 共通の乱数を使う）
	"""
	# 全画像の全画素で別々の乱数（randn_like は generator を受け取れないので randn を使う）
	noise = torch.randn(images.shape, generator=generator,
						dtype=images.dtype, device=images.device)
	images = images + s * noise
	return images.clamp(0, 1)


def add_random_noise(images, sigma, generator=None):
	"""
	バッチごとに強さ s を 0〜sigma の一様分布から 1 回選び、ガウスノイズを足す（PLAN.md 4.2 節）
	sigma = 0 のときは何もしない

	generator にノイズ専用の乱数生成器を渡すと、s とノイズの値はすべてそこから取る。
	こうすると PyTorch 共通の乱数（データの並べ替え、反転・切り抜きに使う）を消費しないので、
	sigma が違っても画像の順番と反転・切り抜きは同じになる。

	Returns
	-------
	images	: torch.Tensor
		ノイズを足した画像
	s	: float
		このバッチで選んだ強さ
	"""
	if sigma <= 0:
		return images, 0.0
	s = torch.rand(1, generator=generator).item() * sigma	# このバッチの強さ（同じバッチの画像はすべて同じ s）
	return gaussian_noise(images, s, generator), s
