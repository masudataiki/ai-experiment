"""
Mac で eval_cifar10c.py を確かめるための、偽の CIFAR-10-C を作る（PLAN.md 12 節 ②）

CIFAR-10 テスト画像 10,000 枚を、壊さずにそのまま 5 回（強さ 1〜5 の分）並べ、
本物と同じ形式（50,000 * 32 * 32 * 3、uint8）の .npy と labels.npy を保存する。
画像は壊れていないので、正しく読めていれば、どの壊れ方・強さでも誤り率は元画像と同じになる。

実行例（free_task/ で実行）:
	.venv/bin/python make_fake_cifar10c.py   -> data/CIFAR-10-C-fake/
"""
import argparse
import os

import numpy as np
from torchvision import datasets

from eval_cifar10c import CORRUPTIONS


def main():
	"""CIFAR-10 テスト画像を 5 回並べた偽の CIFAR-10-C（5 種類 + labels.npy）を保存する"""
	parser = argparse.ArgumentParser(description='Make fake CIFAR-10-C files from the CIFAR-10 test set')
	parser.add_argument('--dataset', '-d', default='data',
						help='Root directory of CIFAR-10 (torchvision format)')
	parser.add_argument('--out', '-o', default='data/CIFAR-10-C-fake',
						help='Output directory')
	args = parser.parse_args()

	testset = datasets.CIFAR10(root=args.dataset, train=False, download=True)
	images = np.concatenate([testset.data] * 5)	# 50,000 * 32 * 32 * 3, uint8
	labels = np.concatenate([np.array(testset.targets, dtype=np.uint8)] * 5)

	os.makedirs(args.out, exist_ok=True)
	for name in CORRUPTIONS:
		np.save(os.path.join(args.out, name + '.npy'), images)
	np.save(os.path.join(args.out, 'labels.npy'), labels)
	print('# saved to {}: images {} {}, labels {} {}'.format(
		args.out, images.shape, images.dtype, labels.shape, labels.dtype))


if __name__ == '__main__':
	main()
