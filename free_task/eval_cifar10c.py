"""
自由課題: 学習済みモデル 1 つを、元画像と CIFAR-10-C で評価し、誤り率(%)を CSV に出す（PLAN.md 5 節、12 節 ②）

実行例（free_task/ で実行）:
	VM : python -u eval_cifar10c.py -g 0 -m result/noise_s000
	     -> result/noise_s000/best_epoch.txt のエポックのモデルを評価し、result/eval_noise_s000.csv に保存
	Mac: .venv/bin/python eval_cifar10c.py -m result/debug_s000 --c-dir data/CIFAR-10-C-fake
"""
import argparse
import csv
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

from network import CifarCNN

# 評価する壊れ方（PLAN.md 5.1 節）
CORRUPTIONS = ['gaussian_noise', 'gaussian_blur', 'fog', 'contrast', 'brightness']
N_TEST = 10000	# 強さ 1 段階あたりの枚数
SEVERITIES = [1, 2, 3, 4, 5]


def to_tensor(images):
	"""
	uint8 の画像（N * 32 * 32 * 3、0〜255）を、学習時の ToTensor() と同じ形にする
	-> float の 0〜1（N * 3 * 32 * 32）。正規化はしない（PLAN.md 2 節）
	"""
	x = torch.from_numpy(np.array(images, dtype=np.uint8))	# mmap（読み取り専用）からメモリにコピー
	return x.permute(0, 3, 1, 2).float() / 255


def error_rate(net, images, labels, device, batchsize):
	"""images（uint8, N*32*32*3）と labels に対する誤り率(%)"""
	loader = DataLoader(TensorDataset(to_tensor(images), torch.as_tensor(labels, dtype=torch.long)),
						batch_size=batchsize, shuffle=False)
	correct, total = 0, 0
	with torch.no_grad():
		for x, y in loader:
			x, y = x.to(device), y.to(device)
			correct += (net(x).argmax(1) == y).sum().item()
			total += y.size(0)
	return 100 * (1 - correct / total)


def resolve_model(path):
	"""フォルダが渡されたら、その中の best_epoch.txt のエポックのモデルを使う"""
	if os.path.isdir(path):
		with open(os.path.join(path, 'best_epoch.txt')) as f:
			ep = int(f.read().strip())
		return os.path.join(path, 'model_{:03d}.pth'.format(ep))
	return path


def main():
	"""引数で指定したモデル 1 つを、元画像と CIFAR-10-C 5 種類 × 強さ 1〜5 で評価し、誤り率を CSV に保存する"""
	parser = argparse.ArgumentParser(description='Evaluate a model on CIFAR-10 and CIFAR-10-C')
	parser.add_argument('--model', '-m', required=True,
						help='Model file, or a result folder (uses best_epoch.txt)')
	parser.add_argument('--out', '-o', default=None,
						help='Output CSV (default: result/eval_<folder name>.csv)')
	parser.add_argument('--dataset', '-d', default='data',
						help='Root directory of CIFAR-10 (torchvision format)')
	parser.add_argument('--c-dir', default='data/CIFAR-10-C',
						help='Directory of CIFAR-10-C .npy files')
	parser.add_argument('--gpu', '-g', type=int, default=-1,
						help='GPU ID (negative value indicates CPU)')
	parser.add_argument('--batchsize', '-b', type=int, default=1000,
						help='Number of images in each mini-batch')
	args = parser.parse_args()

	model_path = resolve_model(args.model)
	if args.out is None:
		name = os.path.basename(os.path.normpath(args.model if os.path.isdir(args.model)
												 else os.path.dirname(args.model)))
		args.out = os.path.join('result', 'eval_{}.csv'.format(name))
	print('# model: {}'.format(model_path))
	print('# CIFAR-10-C: {}'.format(args.c_dir))
	print('# output: {}'.format(args.out))
	print('')

	device = 'cuda:' + str(args.gpu) if args.gpu >= 0 else 'cpu'
	net = CifarCNN(10, ch1=32, ch2=64, dropout=0.0)	# 学習時と同じ（PLAN.md 3 節）
	net.load_state_dict(torch.load(model_path, map_location='cpu'))
	net = net.to(device)
	net.eval()

	rows = []

	# ---------- 元画像（CIFAR-10 テスト 10,000 枚） ----------
	# torchvision の .data は CIFAR-10-C と同じ形式（uint8, N*32*32*3）なので、同じ to_tensor を通す
	testset = datasets.CIFAR10(root=args.dataset, train=False, download=True)
	test_labels = np.array(testset.targets)
	err = error_rate(net, testset.data, test_labels, device, args.batchsize)
	rows.append({'corruption': 'clean', 'severity': 0, 'error': err})
	print('clean: {:.2f}%'.format(err))

	# ---------- CIFAR-10-C ----------
	c_labels = np.load(os.path.join(args.c_dir, 'labels.npy'))
	# ラベルの対応の確認: 強さごとの 10,000 個が、CIFAR-10 テストのラベルと同じ順番か
	same = [np.array_equal(c_labels[(k - 1) * N_TEST:k * N_TEST], test_labels) for k in SEVERITIES]
	print('# labels.npy matches CIFAR-10 test labels (severity 1-5): {}'.format(same))

	for name in CORRUPTIONS:
		data = np.load(os.path.join(args.c_dir, name + '.npy'), mmap_mode='r')	# 全部をメモリに載せない
		errs = []
		for k in SEVERITIES:
			# 強さ k は (k-1)*10,000 枚目から k*10,000 枚目（PLAN.md 5.1 節）
			sl = slice((k - 1) * N_TEST, k * N_TEST)
			err = error_rate(net, data[sl], c_labels[sl], device, args.batchsize)
			rows.append({'corruption': name, 'severity': k, 'error': err})
			errs.append(err)
		rows.append({'corruption': name, 'severity': 'mean', 'error': float(np.mean(errs))})
		print('{}: {}  mean {:.2f}%'.format(name, ' '.join('{:.2f}'.format(e) for e in errs), np.mean(errs)))

	os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
	with open(args.out, 'w', newline='') as f:
		writer = csv.DictWriter(f, fieldnames=['corruption', 'severity', 'error'])
		writer.writeheader()
		for r in rows:
			writer.writerow({**r, 'error': '{:.2f}'.format(r['error'])})
	print('')
	print('# saved: {}'.format(args.out))


if __name__ == '__main__':
	main()
