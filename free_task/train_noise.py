"""
自由課題: ガウスノイズを足すデータ拡張で CifarCNN を学習する（PLAN.md 2〜4 節、12 節 ①）
day3_cv/train_cifar10.py をもとに、本物の CIFAR-10 の読み込みとノイズ拡張を追加した

実行例（free_task/ で実行）:
	VM : python -u train_noise.py -g 0 --noise-sigma 0.1 -e 30 -o result/noise_s010
	Mac: .venv/bin/python train_noise.py -g -1 --noise-sigma 0.1 -e 2 --n-train 300 -o result/debug_s010
"""
import argparse
import csv
import os
import time

import matplotlib
matplotlib.use('Agg')	# 画面のない VM でも図を保存できるようにする
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from network import CifarCNN
from noise import add_random_noise

# 確認用画像で並べるノイズの強さ（PLAN.md 4.3 節の σ）
EXAMPLE_SIGMAS = [0, 0.05, 0.1, 0.2, 0.3]


def save_noise_examples(testset, path, n_images=4):
	"""
	同じテスト画像 n_images 枚に、強さ s = 0, 0.05, 0.1, 0.2, 0.3 のノイズを足して並べた PNG を保存する
	（学習では s は 0〜σ から選ぶが、ここでは各 σ の最大の強さ s = σ を見せる）
	"""
	g = torch.Generator().manual_seed(0)	# 毎回同じノイズの絵になるようにする
	images = torch.stack([testset[i][0] for i in range(n_images)])
	fig, axes = plt.subplots(n_images, len(EXAMPLE_SIGMAS),
							 figsize=(1.6 * len(EXAMPLE_SIGMAS), 1.6 * n_images))
	for j, s in enumerate(EXAMPLE_SIGMAS):
		noisy = (images + s * torch.randn(images.shape, generator=g)).clamp(0, 1)
		for i in range(n_images):
			ax = axes[i, j]
			ax.imshow(noisy[i].permute(1, 2, 0).numpy())	# 3 * 32 * 32 -> 32 * 32 * 3
			ax.set_xticks([])
			ax.set_yticks([])
			if i == 0:
				ax.set_title('s = {}'.format(s))
	fig.tight_layout()
	fig.savefig(path)
	plt.close(fig)


def save_curve(log, path):
	"""学習曲線（横軸エポック、縦軸 学習精度と検証精度）を保存する"""
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.plot([r['epoch'] for r in log], [r['train_acc'] for r in log], label='Training')
	ax.plot([r['epoch'] for r in log], [r['val_acc'] for r in log], label='Validation')
	ax.legend()
	ax.set_xlabel('Epoch')
	ax.set_ylabel('Accuracy [%]')
	ax.set_ylim(0, 100)
	fig.savefig(path)
	plt.close(fig)


def main():
	"""引数を読み、CIFAR-10 を 45,000 / 5,000 に分けて CifarCNN を学習する。毎エポック、ログ・モデル・学習曲線を保存する"""
	parser = argparse.ArgumentParser(description='CIFAR-10 with Gaussian noise augmentation')
	parser.add_argument('--noise-sigma', type=float, default=0.0,
						help='Max std of the Gaussian noise (0: no noise). s ~ U(0, sigma) per batch')
	parser.add_argument('--epoch', '-e', type=int, default=30,
						help='Number of sweeps over the training data')
	parser.add_argument('--seed', type=int, default=0,
						help='Random seed for training (the train/val split always uses seed 0)')
	parser.add_argument('--out', '-o', default='result/debug',
						help='Directory to output the result')
	parser.add_argument('--gpu', '-g', type=int, default=-1,
						help='GPU ID (negative value indicates CPU)')
	parser.add_argument('--dataset', '-d', default='data',
						help='Root directory of CIFAR-10 (torchvision format)')
	parser.add_argument('--batchsize', '-b', type=int, default=100,
						help='Number of images in each mini-batch')
	parser.add_argument('--n-train', type=int, default=None,
						help='Use only the first N training images (for a quick check on Mac)')
	args = parser.parse_args()

	os.makedirs(args.out, exist_ok=True)
	for k, v in vars(args).items():
		print('# {}: {}'.format(k, v))
	print('')

	# 学習の乱数（重みの初期値、シャッフル、反転・切り抜き）の種を固定する
	torch.manual_seed(args.seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	# ノイズ（強さ s とノイズの値）は専用の乱数生成器から取る。
	# 共通の乱数を消費しないので、sigma が違っても画像の順番と反転・切り抜きは全モデルで同じになる
	noise_generator = torch.Generator().manual_seed(args.seed)

	device = 'cuda:' + str(args.gpu) if args.gpu >= 0 else 'cpu'

	# ---------- データ（PLAN.md 2 節、4.1 節） ----------
	# 正規化はしない（0〜1 のまま）
	plain = transforms.ToTensor()
	augment = transforms.Compose([
		transforms.RandomHorizontalFlip(),
		transforms.RandomCrop(32, padding=4),
		transforms.ToTensor()])
	# 同じ 50,000 枚を、拡張ありと拡張なしの 2 通りの transform で用意する
	trainset_aug = datasets.CIFAR10(root=args.dataset, train=True, download=True, transform=augment)
	trainset_plain = datasets.CIFAR10(root=args.dataset, train=True, download=True, transform=plain)
	testset = datasets.CIFAR10(root=args.dataset, train=False, download=True, transform=plain)

	# 45,000 / 5,000 の分け方は --seed に関係なく seed 0 で固定する（全モデルで同じ検証データ）
	split = torch.utils.data.random_split(range(len(trainset_plain)), [45000, 5000],
										  generator=torch.Generator().manual_seed(0))
	train_idx, val_idx = list(split[0]), list(split[1])
	if args.n_train is not None:
		train_idx = train_idx[:args.n_train]
	trainset = Subset(trainset_aug, train_idx)	# 学習: 反転・切り抜きあり（ノイズはループの中で足す）
	valset = Subset(trainset_plain, val_idx)	# 検証: 拡張もノイズもなし
	print('# train: {} images, validation: {} images'.format(len(trainset), len(valset)))

	trainloader = DataLoader(trainset, batch_size=args.batchsize, shuffle=True, num_workers=2)
	valloader = DataLoader(valset, batch_size=args.batchsize, shuffle=False, num_workers=2)

	# ノイズの強さの確認用画像
	save_noise_examples(testset, os.path.join(args.out, 'noise_examples.png'))

	# ---------- ネットワーク（PLAN.md 3 節） ----------
	net = CifarCNN(10, ch1=32, ch2=64, dropout=0.0).to(device)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(), lr=0.001)

	# ---------- 学習 ----------
	log = []
	best_epoch, best_acc = 0, -1.0
	for ep in range(1, args.epoch + 1):
		start = time.time()

		net.train()
		loss_sum, correct_train, total_train, n_batch = 0.0, 0, 0, 0
		for images, labels in trainloader:
			# PLAN.md 4.2 節: バッチごとに強さ s を選んでノイズを足す（sigma = 0 なら何もしない）
			images, _ = add_random_noise(images, args.noise_sigma, noise_generator)
			images, labels = images.to(device), labels.to(device)

			optimizer.zero_grad()
			outputs = net(images)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			loss_sum += loss.item()
			n_batch += 1
			# 学習精度は、ノイズを足した画像（モデルが実際に見た画像）に対する値
			correct_train += (outputs.argmax(1) == labels).sum().item()
			total_train += labels.size(0)

		net.eval()
		correct_val, total_val = 0, 0
		with torch.no_grad():
			for images, labels in valloader:
				images, labels = images.to(device), labels.to(device)
				correct_val += (net(images).argmax(1) == labels).sum().item()
				total_val += labels.size(0)

		row = {
			'epoch': ep,
			'train_loss': loss_sum / n_batch,	# バッチごとの損失の平均
			'train_acc': 100 * correct_train / total_train,
			'val_acc': 100 * correct_val / total_val,
			'seconds': time.time() - start,
		}
		log.append(row)
		print('[epoch {epoch}] loss {train_loss:.4f}, train {train_acc:.2f}%, '
			  'validation {val_acc:.2f}%, {seconds:.1f} s'.format(**row))

		# 毎エポックのモデルを残す
		torch.save(net.state_dict(), os.path.join(args.out, 'model_{:03d}.pth'.format(ep)))
		if row['val_acc'] > best_acc:
			best_epoch, best_acc = ep, row['val_acc']

		# 途中で止まっても結果が残るように、CSV と図は毎エポック書き直す
		with open(os.path.join(args.out, 'train_log.csv'), 'w', newline='') as f:
			writer = csv.DictWriter(f, fieldnames=list(row.keys()))
			writer.writeheader()
			writer.writerows(log)
		save_curve(log, os.path.join(args.out, 'accuracy.png'))

	# 検証精度が最大だったエポック（同じ値なら早いほう）
	with open(os.path.join(args.out, 'best_epoch.txt'), 'w') as f:
		f.write('{}\n'.format(best_epoch))
	print('Finished Training')
	print('# best epoch: {} (validation {:.2f}%) -> model_{:03d}.pth'.format(best_epoch, best_acc, best_epoch))


if __name__ == '__main__':
	main()
