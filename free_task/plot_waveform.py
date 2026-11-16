"""
自由課題: 1 行の波形の図（PLAN.md 5.3 節 3、12 節 ④）

CIFAR-10 テスト画像 1 枚について、元画像と CIFAR-10-C の 4 種類（同じ強さ）の画像を上に並べ、
その下に、真ん中の横 1 行の明るさ（グレースケール）を折れ線で描く。
各波形には、比べやすいよう元画像の波形を灰色の破線で重ねる。

実行例（free_task/ で実行。CIFAR-10-C は VM にしかない）:
	VM : python plot_waveform.py                 -> result/figures/waveform_i0_s3.png
	     python plot_waveform.py --index 7 --severity 3
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets

from plot_results import CORRUPTION_COLORS, NAMES, TEXT, style_axes

PANELS = ['clean', 'gaussian_noise', 'gaussian_blur', 'fog', 'contrast']
N_TEST = 10000
ROW_MARK = '#e34948'	# 画像の上に引く「この行」の印


def to_gray(img):
	"""uint8 の 32*32*3 -> 0〜1 のグレースケール（ITU-R BT.601 の重み）"""
	return (img.astype(np.float64) / 255) @ np.array([0.299, 0.587, 0.114])


def main():
	parser = argparse.ArgumentParser(description='Plot one row of pixels for clean and corrupted images')
	parser.add_argument('--index', '-i', type=int, default=0, help='Index of the CIFAR-10 test image (0-9999)')
	parser.add_argument('--severity', '-s', type=int, default=3, help='Severity of CIFAR-10-C (1-5)')
	parser.add_argument('--row', type=int, default=16, help='Row to plot (0-31, default: middle)')
	parser.add_argument('--dataset', '-d', default='data', help='Root directory of CIFAR-10')
	parser.add_argument('--c-dir', default='data/CIFAR-10-C', help='Directory of CIFAR-10-C .npy files')
	parser.add_argument('--out', '-o', default=None,
						help='Output PNG (default: result/figures/waveform_i<index>_s<severity>.png)')
	args = parser.parse_args()
	if args.out is None:
		args.out = 'result/figures/waveform_i{}_s{}.png'.format(args.index, args.severity)

	testset = datasets.CIFAR10(root=args.dataset, train=False, download=True)
	images = {'clean': testset.data[args.index]}
	# 強さ k の i 枚目は、(k-1)*10,000 + i 番目（PLAN.md 5.1 節）
	pos = (args.severity - 1) * N_TEST + args.index
	for name in PANELS[1:]:
		images[name] = np.load(os.path.join(args.c_dir, name + '.npy'), mmap_mode='r')[pos]
	labels = np.load(os.path.join(args.c_dir, 'labels.npy'))
	assert labels[pos] == testset.targets[args.index], 'CIFAR-10-C と CIFAR-10 で画像が対応していない'
	rows = {name: to_gray(img)[args.row] for name, img in images.items()}

	fig, axes = plt.subplots(2, len(PANELS), figsize=(3.0 * len(PANELS), 5.4),
							 gridspec_kw={'height_ratios': [1, 1]})
	x = np.arange(32)
	for j, name in enumerate(PANELS):
		# 上: 画像そのもの（拡大しても画素が見えるよう nearest）。描いた行に印を付ける
		ax = axes[0, j]
		ax.imshow(images[name], interpolation='nearest')
		ax.axhline(args.row, color=ROW_MARK, linewidth=1)
		ax.set_xticks([])
		ax.set_yticks([])
		title = NAMES[name] if name == 'clean' else '{} (sev. {})'.format(NAMES[name], args.severity)
		ax.set_title(title, color=TEXT, fontsize=11)

		# 下: その行の明るさ。元画像の波形を灰色の破線で重ねる
		ax = axes[1, j]
		if name != 'clean':
			ax.plot(x, rows['clean'], color='#8a8985', linestyle='--', linewidth=1.5, label='Clean')
		ax.plot(x, rows[name], color=CORRUPTION_COLORS[name], linewidth=2, label=NAMES[name])
		ax.set_xlim(0, 31)
		ax.set_ylim(0, 1)
		ax.set_xlabel('x (pixel)', color=TEXT)
		style_axes(ax)
		if j == 0:
			ax.set_ylabel('Brightness (gray, 0-1)', color=TEXT)
		else:
			ax.tick_params(labelleft=False)
			ax.legend(loc='upper right', frameon=False, fontsize=8)

	fig.suptitle('CIFAR-10 test #{} ({}), row y = {}'.format(
		args.index, testset.classes[testset.targets[args.index]], args.row), color=TEXT)
	fig.tight_layout()
	os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
	fig.savefig(args.out, dpi=200)
	plt.close(fig)
	print('# saved: {}'.format(args.out))


if __name__ == '__main__':
	main()
