#!/usr/bin/env bash
# 自由課題で使うデータを free_task/data/ に取得する
#   - CIFAR-10（torchvision）     -> free_task/data/cifar-10-batches-py/
#   - CIFAR-10-C（Zenodo の tar）  -> free_task/data/CIFAR-10-C/*.npy
#
# 使い方（どのディレクトリから実行してもよい）:
#   Mac: PYTHON=.venv/bin/python bash free_task/download_data.sh
#   VM : source ~/venvs/aiexp-cv/bin/activate && bash free_task/download_data.sh
#
# すでに取得済みのデータはスキップする（何度実行してもよい）。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
PYTHON="${PYTHON:-python}"

# CIFAR-10-C から取り出すファイル（PLAN.md 5.1 の表）。
# defocus_blur を評価に加える場合は、ここに追加して再実行する。
CORRUPTIONS=(gaussian_noise gaussian_blur fog contrast brightness labels)

mkdir -p "$DATA_DIR"
echo "# データの保存先: $DATA_DIR"
df -h "$DATA_DIR"

# ---------------------------------------------------------------
# 1. CIFAR-10（学習用 50,000 枚 + テスト用 10,000 枚、約 170 MB）
# ---------------------------------------------------------------
echo "# CIFAR-10 を取得"
"$PYTHON" -c '
import sys
from torchvision import datasets
root = sys.argv[1]
for train in (True, False):
    d = datasets.CIFAR10(root=root, train=train, download=True)
    print("  train=%s: %d 枚" % (train, len(d)))
' "$DATA_DIR"

# ---------------------------------------------------------------
# 2. CIFAR-10-C（tar は約 2.9 GB。必要なファイルだけ展開して tar は削除）
# ---------------------------------------------------------------
missing=()
for name in "${CORRUPTIONS[@]}"; do
	if [ ! -f "$DATA_DIR/CIFAR-10-C/$name.npy" ]; then
		missing+=("CIFAR-10-C/$name.npy")
	fi
done

if [ ${#missing[@]} -eq 0 ]; then
	echo "# CIFAR-10-C は取得済み"
else
	echo "# CIFAR-10-C を取得: ${missing[*]}"
	cd "$DATA_DIR"
	# -L: zenodo.org/record/... は /records/... にリダイレクトされるので、それに従う
	# --fail: HTTP エラーのときにエラーページを tar として保存しない
	curl -L --fail -O https://zenodo.org/record/2535967/files/CIFAR-10-C.tar
	tar -xf CIFAR-10-C.tar "${missing[@]}"
	rm CIFAR-10-C.tar
fi

echo "# 取得したファイル"
ls -lh "$DATA_DIR/CIFAR-10-C/"
