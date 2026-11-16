"""
自由課題: 各モデルの評価 CSV から発表用の図を描く（PLAN.md 5.3 節、12 節 ③）

入力: <in-dir>/eval_noise_sNNN.csv（NNN = σ × 100。eval_cifar10c.py の出力）。ある分だけで描く。
出力: <out-dir>/heatmap_diff.png, sigma_lines.png, severity_lines.png

実行例（free_task/ で実行）:
	.venv/bin/python plot_results.py                  # result/ の CSV -> result/figures/
	.venv/bin/python plot_results.py --in-dir X --out-dir Y
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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# 横に並べる順番（PLAN.md 5.1 節の表の順）と、図に書く名前
COLUMNS = ['clean', 'gaussian_noise', 'gaussian_blur', 'fog', 'contrast', 'brightness']
NAMES = {'clean': 'Clean', 'gaussian_noise': 'Gaussian noise', 'gaussian_blur': 'Gaussian blur',
		 'fog': 'Fog', 'contrast': 'Contrast', 'brightness': 'Brightness'}
CORRUPTIONS = COLUMNS[1:]
# PLAN.md 4.3 節の ID
MODEL_ID = {0.0: 'M0', 0.05: 'M1', 0.1: 'M2', 0.2: 'M3', 0.3: 'M4'}

# --- 色（dataviz の参照パレット） ---
TEXT = '#0b0b0b'
TEXT_SECONDARY = '#52514e'
GRID = '#e4e3df'
# 壊れ方: 決まった順番の 5 色（元画像は基準なので黒）
CORRUPTION_COLORS = {'clean': TEXT, 'gaussian_noise': '#2a78d6', 'gaussian_blur': '#eb6834',
					 'fog': '#1baf7a', 'contrast': '#eda100', 'brightness': '#e87ba4'}
MARKERS = {'clean': 'o', 'gaussian_noise': 's', 'gaussian_blur': 'D', 'fog': '^',
		   'contrast': 'v', 'brightness': 'P'}
# モデル: 基準の M0 は灰色の破線、M1〜M4 は σ が大きいほど濃い青（σ ごとに固定。ある分だけ描いても色は変わらない）
SIGMA_STYLE = {
	0.0: {'color': '#8a8985', 'linestyle': '--', 'marker': 'o'},
	0.05: {'color': '#86b6ef', 'linestyle': '-', 'marker': 's'},
	0.1: {'color': '#3987e5', 'linestyle': '-', 'marker': 'D'},
	0.2: {'color': '#1c5cab', 'linestyle': '-', 'marker': '^'},
	0.3: {'color': '#0d366b', 'linestyle': '-', 'marker': 'v'},
}
# ヒートマップ: 青（良くなった、マイナス）<-> 灰色（0）<-> 赤（悪くなった、プラス）
DIVERGING = LinearSegmentedColormap.from_list(
	'blue_gray_red', ['#184f95', '#6da7ec', '#f0efec', '#ee8a89', '#b52a2a'])


def load(in_dir):
	"""{sigma: {(corruption, severity): error}} を返す。severity は 0（元画像）, 1〜5, 'mean'"""
	results = {}
	for path in sorted(glob.glob(os.path.join(in_dir, 'eval_noise_s*.csv'))):
		m = re.fullmatch(r'eval_noise_s(\d{3})\.csv', os.path.basename(path))
		if m is None:
			continue
		sigma = int(m.group(1)) / 100
		table = {}
		with open(path) as f:
			for row in csv.DictReader(f):
				sev = row['severity'] if row['severity'] == 'mean' else int(row['severity'])
				table[(row['corruption'], sev)] = float(row['error'])
		table[('clean', 'mean')] = table[('clean', 0)]	# 元画像も「平均」の列に並べるため
		results[sigma] = table
		print('# loaded: {} (sigma = {})'.format(path, sigma))
	return results


def label(sigma):
	mid = MODEL_ID.get(sigma)
	return '{} (σ={:g})'.format(mid, sigma) if mid else 'σ={:g}'.format(sigma)


def style_axes(ax):
	"""グリッドと軸線を控えめにする"""
	ax.grid(True, color=GRID, linewidth=0.8)
	ax.set_axisbelow(True)
	for side in ('top', 'right'):
		ax.spines[side].set_visible(False)
	for side in ('left', 'bottom'):
		ax.spines[side].set_color(TEXT_SECONDARY)
	ax.tick_params(colors=TEXT_SECONDARY)


def plot_heatmap(results, path):
	"""図 1: マスの値 = そのモデルの誤り率（平均）− M0 の誤り率（平均）"""
	sigmas = sorted(results)
	base = results[0.0]
	diff = np.array([[results[s][(c, 'mean')] - base[(c, 'mean')] for c in COLUMNS] for s in sigmas])
	vmax = max(5.0, np.abs(diff).max())	# 0 を真ん中にして左右対称（差が小さいときに色が強く出すぎないよう最低 ±5）

	fig, ax = plt.subplots(figsize=(8.5, 1.0 + 0.6 * len(sigmas)))
	im = ax.imshow(diff, cmap=DIVERGING, norm=TwoSlopeNorm(0, -vmax, vmax), aspect='auto')
	for i in range(len(sigmas)):
		for j in range(len(COLUMNS)):
			v = diff[i, j]
			color = 'white' if abs(v) > 0.6 * vmax else TEXT	# 濃いマスは白文字
			ax.text(j, i, '{:+.1f}'.format(v), ha='center', va='center', color=color, fontsize=10)
	ax.set_xticks(range(len(COLUMNS)), [NAMES[c] for c in COLUMNS])
	ax.set_yticks(range(len(sigmas)), [label(s) for s in sigmas])
	ax.tick_params(length=0, colors=TEXT)
	for spine in ax.spines.values():
		spine.set_visible(False)
	# マスの間に 2px 相当の白い区切り
	ax.set_xticks(np.arange(-0.5, len(COLUMNS)), minor=True)
	ax.set_yticks(np.arange(-0.5, len(sigmas)), minor=True)
	ax.grid(which='minor', color='white', linewidth=2)
	ax.tick_params(which='minor', length=0)
	cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
	cbar.set_label('Error rate change vs M0 [points]\n(blue: better, red: worse)', color=TEXT_SECONDARY)
	cbar.outline.set_visible(False)
	ax.set_title('Mean error rate (severity 1-5) minus M0', color=TEXT, loc='left')
	fig.tight_layout()
	fig.savefig(path, dpi=200)
	plt.close(fig)


def plot_sigma_lines(results, path):
	"""図 2: 横軸 σ、縦軸 誤り率（平均）、壊れ方ごとに 1 本"""
	sigmas = sorted(results)
	fig, ax = plt.subplots(figsize=(7, 4.5))
	for c in COLUMNS:
		ys = [results[s][(c, 'mean')] for s in sigmas]
		ax.plot(sigmas, ys, color=CORRUPTION_COLORS[c], marker=MARKERS[c], markersize=7,
				linewidth=2, linestyle='--' if c == 'clean' else '-', label=NAMES[c])
	ax.set_xticks(sigmas)
	ax.set_xlabel('Noise augmentation max strength σ', color=TEXT)
	ax.set_ylabel('Error rate, mean of severity 1-5 [%]', color=TEXT)
	ax.set_ylim(0, 100)
	style_axes(ax)
	ax.legend(loc='upper left', frameon=False, ncol=2)
	fig.tight_layout()
	fig.savefig(path, dpi=200)
	plt.close(fig)


def plot_severity_lines(results, path):
	"""図 3: 壊れ方ごとに 1 枚、横軸 強さ 1〜5、縦軸 誤り率、モデルごとに 1 本。90% に点線"""
	sigmas = sorted(results)
	fig, axes = plt.subplots(1, len(CORRUPTIONS), figsize=(3.0 * len(CORRUPTIONS), 3.6), sharey=True)
	for ax, c in zip(axes, CORRUPTIONS):
		ax.axhline(90, color=TEXT_SECONDARY, linestyle=':', linewidth=1.5)
		for s in sigmas:
			ax.plot(range(1, 6), [results[s][(c, sev)] for sev in range(1, 6)],
					markersize=6, linewidth=2, label=label(s), **SIGMA_STYLE.get(s, {}))
		ax.set_title(NAMES[c], color=TEXT)
		ax.set_xticks(range(1, 6))
		ax.set_xlabel('Severity', color=TEXT)
		ax.set_ylim(0, 100)
		style_axes(ax)
	axes[0].set_ylabel('Error rate [%]', color=TEXT)
	axes[0].annotate('chance (90%)', (1, 90), xytext=(0, 4), textcoords='offset points',
					 color=TEXT_SECONDARY, fontsize=8)
	handles, labels = axes[0].get_legend_handles_labels()
	fig.legend(handles, labels, loc='lower center', ncol=len(sigmas), frameon=False)
	fig.tight_layout(rect=(0, 0.08, 1, 1))
	fig.savefig(path, dpi=200)
	plt.close(fig)


def main():
	parser = argparse.ArgumentParser(description='Plot figures from eval CSVs')
	parser.add_argument('--in-dir', default='result', help='Directory of eval_noise_sNNN.csv')
	parser.add_argument('--out-dir', default='result/figures', help='Directory to save figures')
	args = parser.parse_args()

	results = load(args.in_dir)
	if not results:
		raise SystemExit('eval_noise_sNNN.csv が {} に見つかりません'.format(args.in_dir))
	os.makedirs(args.out_dir, exist_ok=True)

	if 0.0 in results:
		plot_heatmap(results, os.path.join(args.out_dir, 'heatmap_diff.png'))
	else:
		print('# M0（eval_noise_s000.csv）が無いので、ヒートマップは描きません')
	plot_sigma_lines(results, os.path.join(args.out_dir, 'sigma_lines.png'))
	plot_severity_lines(results, os.path.join(args.out_dir, 'severity_lines.png'))
	print('# saved figures to {}'.format(args.out_dir))


if __name__ == '__main__':
	main()
