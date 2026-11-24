"""
自由課題: 周波数帯ごとの敏感さの分析（PLAN.md 12 節 ⑤）

テスト画像に、周波数帯だけを変えた「同じ大きさ（L2 ノルム ε）の邪魔」を足し、
各モデルの誤り率が元画像からどれだけ増えるかを測る。全モデルに同じ邪魔を入れる。

出力:
	result/freq_bands_<モデルのフォルダ名>.csv
	result/figures/freq_sensitivity.png   横軸 帯、縦軸 増えた分、モデルごとに 1 本
	result/figures/freq_examples.png      帯 1・5・9 の邪魔を足した画像と、邪魔そのもの

実行例（free_task/ で実行）:
	VM : python -u freq_bands.py -g 0                       # result/noise_sNNN を全部、ε = 4
	     python -u freq_bands.py -g 0 --eps 4 2             # ε = 2 も
	Mac: .venv/bin/python freq_bands.py --models result/noise_s000 --n-test 500
"""
import argparse
import csv
import glob
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets

from eval_cifar10c import resolve_model, to_tensor
from network import CifarCNN
from plot_results import SIGMA_STYLE, TEXT, label, style_axes

# 周波数帯（PLAN.md 12 節 ⑤ の表）。r = √(u² + v²)、u, v は 32 画素の中で何回上下するか
BANDS = [(1, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 14), (14, 16), (16, None)]
SIZE = 32


def radius():
	"""各 FFT 成分の周波数の大きさ r（32 * 32）。fftfreq(32) * 32 で -16〜15 の整数"""
	f = torch.fft.fftfreq(SIZE) * SIZE
	return torch.sqrt(f[:, None] ** 2 + f[None, :] ** 2)


def band_mask(lo, hi):
	"""lo ≤ r < hi の成分だけ True（hi = None は上限なし）。r = 0 は帯 1 の下限が 1 なので入らない"""
	r = radius()
	m = r >= lo
	if hi is not None:
		m &= r < hi
	return m


def make_perturbation(n, mask, eps, generator):
	"""
	指定した帯だけを含む邪魔を n 枚分作る（n * 3 * 32 * 32）
	1. 正規乱数 2. チャンネルごとに 2 次元 FFT し、帯の外を 0 3. 逆 FFT の実部
	4. 1 枚ごとに L2 ノルム（3,072 個の値）が eps になるよう拡大・縮小
	"""
	z = torch.randn(n, 3, SIZE, SIZE, generator=generator)
	d = torch.fft.ifft2(torch.fft.fft2(z) * mask).real
	norm = d.flatten(1).norm(dim=1).view(n, 1, 1, 1)
	return d / norm * eps


def check_perturbation(d, mask, eps):
	"""確認用: L2 ノルムが eps か、FFT の二乗和のうち帯の中にある割合"""
	norms = d.flatten(1).norm(dim=1)
	power = torch.fft.fft2(d).abs() ** 2
	inside = (power * mask).flatten(1).sum(1) / power.flatten(1).sum(1)
	return norms.min().item(), norms.max().item(), inside.min().item()


def sigma_of(folder):
	"""result/noise_sNNN -> σ = NNN / 100"""
	m = re.fullmatch(r'noise_s(\d{3})', os.path.basename(os.path.normpath(folder)))
	return int(m.group(1)) / 100 if m else None


def count_correct(net, x, y, device):
	"""1 バッチ（x: 画像, y: 正解ラベル）のうち、正しく分類できた枚数"""
	with torch.no_grad():
		return (net(x.to(device)).argmax(1) == y.to(device)).sum().item()


def plot_sensitivity(results, eps_list, path):
	"""図 1: 横軸 帯、縦軸 増えた分（ポイント）、モデルごとに 1 本。ε ごとに 1 枚"""
	fig, axes = plt.subplots(1, len(eps_list), figsize=(7 * len(eps_list), 4.5), squeeze=False)
	xt = ['{}\n{}-{}'.format(b + 1, lo, hi if hi is not None else '') for b, (lo, hi) in enumerate(BANDS)]
	for ax, eps in zip(axes[0], eps_list):
		for folder, rows in results.items():
			s = sigma_of(folder)
			ys = [r['increase'] for r in rows if r['eps'] == eps]
			ax.plot(range(1, len(BANDS) + 1), ys, markersize=6, linewidth=2,
					label=label(s) if s is not None else os.path.basename(folder), **SIGMA_STYLE.get(s, {}))
		ax.axhline(0, color='#8a8985', linewidth=1)
		ax.set_xticks(range(1, len(BANDS) + 1), xt)
		ax.set_xlabel('Frequency band (r range)  ←slow   fine→', color=TEXT)
		ax.set_ylabel('Error rate increase vs clean [points]', color=TEXT)
		ax.set_title('Perturbation L2 norm ε = {:g}'.format(eps), color=TEXT, loc='left')
		style_axes(ax)
		ax.legend(frameon=False)
	fig.tight_layout()
	fig.savefig(path, dpi=200)
	plt.close(fig)


def plot_examples(image, eps, seed, path, bands=(1, 5, 9)):
	"""図 2: 上の段 元画像と帯 1・5・9 の邪魔を足した画像、下の段 邪魔そのもの（0.5 中心に拡大）"""
	g = torch.Generator().manual_seed(seed)
	ds = [make_perturbation(1, band_mask(*BANDS[b - 1]), eps, g)[0] for b in bands]
	k = 0.5 / max(d.abs().max().item() for d in ds)	# 3 つ共通の倍率（帯どうしで大きさを比べられるように）
	fig, axes = plt.subplots(2, len(bands) + 1, figsize=(2.6 * (len(bands) + 1), 5.6))
	show = lambda ax, t: ax.imshow(t.permute(1, 2, 0).clamp(0, 1).numpy(), interpolation='nearest')
	show(axes[0, 0], image)
	axes[0, 0].set_title('Clean', color=TEXT)
	axes[1, 0].axis('off')
	axes[1, 0].text(0.5, 0.5, 'perturbation\n(0.5 + {:.1f} × d)'.format(k), ha='center', va='center',
					color=TEXT, transform=axes[1, 0].transAxes)
	for j, (b, d) in enumerate(zip(bands, ds), start=1):
		lo, hi = BANDS[b - 1]
		show(axes[0, j], image + d)
		axes[0, j].set_title('Band {} (r {}-{})'.format(b, lo, hi if hi is not None else ''), color=TEXT)
		show(axes[1, j], 0.5 + k * d)
	for ax in axes.flat:
		ax.set_xticks([])
		ax.set_yticks([])
	fig.suptitle('Same L2 norm ε = {:g}, different frequency bands'.format(eps), color=TEXT)
	fig.tight_layout()
	fig.savefig(path, dpi=200)
	plt.close(fig)


def main():
	"""各モデルについて、元画像と 9 帯 × ε の邪魔を足した画像の誤り率を測り、CSV 5 つと図 2 枚を保存する"""
	parser = argparse.ArgumentParser(description='Sensitivity to perturbations in each frequency band')
	parser.add_argument('--models', nargs='+', default=None,
						help='Result folders (default: all result/noise_sNNN with best_epoch.txt)')
	parser.add_argument('--eps', type=float, nargs='+', default=[4.0], help='L2 norm of the perturbation')
	parser.add_argument('--n-test', type=int, default=10000, help='Use the first N test images')
	parser.add_argument('--seed', type=int, default=0, help='Seed of the perturbations (same for all models)')
	parser.add_argument('--dataset', '-d', default='data', help='Root directory of CIFAR-10')
	parser.add_argument('--out-dir', default='result', help='Directory for CSVs (figures go to <out-dir>/figures)')
	parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU)')
	parser.add_argument('--batchsize', '-b', type=int, default=1000)
	args = parser.parse_args()

	if args.models is None:
		args.models = sorted(d for d in glob.glob('result/noise_s*')
							 if sigma_of(d) is not None and os.path.exists(os.path.join(d, 'best_epoch.txt')))
	device = 'cuda:' + str(args.gpu) if args.gpu >= 0 else 'cpu'

	# ---------- モデル（検証精度が最大のエポック） ----------
	nets = {}
	for folder in args.models:
		path = resolve_model(folder)
		net = CifarCNN(10, ch1=32, ch2=64, dropout=0.0)
		net.load_state_dict(torch.load(path, map_location='cpu'))
		nets[folder] = net.to(device).eval()
		print('# model: {}'.format(path))

	# ---------- テスト画像（0〜1、正規化なし） ----------
	testset = datasets.CIFAR10(root=args.dataset, train=False, download=True)
	x_all = to_tensor(testset.data[:args.n_test])
	y_all = torch.tensor(testset.targets[:args.n_test])
	n = len(y_all)
	batches = [(x_all[i:i + args.batchsize], y_all[i:i + args.batchsize]) for i in range(0, n, args.batchsize)]
	print('# test images: {}, eps: {}'.format(n, args.eps))

	clean_err = {}
	for folder, net in nets.items():
		correct = sum(count_correct(net, x, y, device) for x, y in batches)
		clean_err[folder] = 100 * (1 - correct / n)
		print('clean error {}: {:.2f}%'.format(folder, clean_err[folder]))

	# ---------- 帯ごと ----------
	results = {folder: [] for folder in nets}
	for eps in args.eps:
		for b, (lo, hi) in enumerate(BANDS, start=1):
			mask = band_mask(lo, hi)
			# 帯・ε ごとに種を決める。バッチを同じ順に作るので、全モデルが同じ邪魔を見る
			g = torch.Generator().manual_seed(args.seed * 1000 + b)
			correct = {folder: 0 for folder in nets}
			for i, (x, y) in enumerate(batches):
				d = make_perturbation(len(y), mask, eps, g)
				if i == 0:
					nmin, nmax, inside = check_perturbation(d, mask, eps)
				xp = (x + d).clamp(0, 1)
				for folder, net in nets.items():
					correct[folder] += count_correct(net, xp, y, device)
			msg = []
			for folder in nets:
				err = 100 * (1 - correct[folder] / n)
				results[folder].append({'band': b, 'r_low': lo, 'r_high': hi if hi is not None else '',
										'eps': eps, 'error': err, 'clean_error': clean_err[folder],
										'increase': err - clean_err[folder]})
				msg.append('{:+.2f}'.format(err - clean_err[folder]))
			print('[eps {:g} band {} r {}-{}] L2 norm {:.4f}-{:.4f}, in-band power >= {:.4f} | increase: {}'.format(
				eps, b, lo, hi if hi is not None else '', nmin, nmax, inside, ' '.join(msg)))

	# ---------- 出力 ----------
	os.makedirs(os.path.join(args.out_dir, 'figures'), exist_ok=True)
	for folder, rows in results.items():
		path = os.path.join(args.out_dir, 'freq_bands_{}.csv'.format(os.path.basename(os.path.normpath(folder))))
		with open(path, 'w', newline='') as f:
			writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
			writer.writeheader()
			for r in rows:
				writer.writerow({k: ('{:.2f}'.format(v) if k in ('error', 'clean_error', 'increase') else v)
								 for k, v in r.items()})
		print('# saved: {}'.format(path))
	plot_sensitivity(results, args.eps, os.path.join(args.out_dir, 'figures', 'freq_sensitivity.png'))
	plot_examples(x_all[0], args.eps[0], args.seed, os.path.join(args.out_dir, 'figures', 'freq_examples.png'))
	print('# saved figures to {}'.format(os.path.join(args.out_dir, 'figures')))


if __name__ == '__main__':
	main()
