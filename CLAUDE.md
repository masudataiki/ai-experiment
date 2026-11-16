# このリポジトリについて
- 東大EEIC「人工知能演習」の基本課題（day2 NLP/RL, day3 CV, day4 speech）用リポジトリ。
- 学習目的の課題。穴埋め部分は答えを先に書かず、ヒントと理論の説明を優先すること。

# 実行環境
- Python は必ず `.venv/bin/python` を使う（pip も `.venv/bin/python -m pip`）。
- Mac (arm64) では CUDA が無いので、配布コードは `-g -1`（CPU）で動作確認する。
- GPU を使う本番の学習は Google Colab で行う。

# 実験書（docs/2026_人工知能演習.pdf）
演習について聞かれたら、まず該当ページを読んでから答えること。
| フォルダ | 章 | ページ |
|---|---|---|
| day2_nlp | 第2章 Natural Language Processing（理論 2.1-2.6、演習 2.7） | p.11-19 |
| day2_rl | 第3章 Reinforcement Learning（理論 3.1-3.3、演習 3.4） | p.21-31 |
| day3_cv | 第4章 Image Recognition（理論 4.1、演習 4.2-4.3） | p.33-47 |
| day4_speech | 第5章 Audio and Speech Processing（理論 5.1-5.3、演習 5.4） | p.49-59 |

# 説明のしかた
- 演習のコードが実験書のどの式（例: 式(2.1)）に対応するかを、式番号つきで示す。
- 実験書の記法（W_h, U_h, h_t など）とコードの変数名（Wxhi, Whrhr など）の対応表を示す。
- 穴埋めは、答えのコードを先に書かない。場所・方針・ヒントを示し、私が書いたものをレビューする。

# GPU が必要な実行（Colab）
- コードは Mac で編集して GitHub に push し、Colab のノート（Drive の Colab Notebooks）で
  セル1〜3（Drive 接続 → git clone/pull → data/ と result/ を Drive にリンク）を実行してから使う。
- フォルダを Colab にアップロードする方法は使わない。
- 実行コマンドは `| tee result/<名前>/train.log` を付けてログを Drive に残す。
- 結果は Drive の AI_Experiment/results/<day>/ に保存される。README 用には Mac の <day>/result/ にダウンロードして使う。

- Git のブランチは作らず、すべて main で作業・コミット・push する（Colab は main を取得するため）。

# GPU 実行環境（2026-11-10 以降）
- Colab は GPU 利用上限に達したため、GPU が必要な実行は Azure VM（ssh aiexp）で行う。
- VM 上の場所: ~/ai-experiment（main を git pull）、venv は ~/venvs/aiexp-cv
- データは VM の day3_cv/data にコピー済み（Drive は使わない）
- 実行は tmux の中で、`( time python -u ... ) 2>&1 | tee result/<NAME>/train.log` の形で行い、gpu.txt と packages.txt も残す
- 結果は Mac から `scp -r aiexp:~/ai-experiment/<day>/result/<NAME> <day>/result/` で取得する
- 実験を比較するときは、すべて同じ環境で実行する

# 自由課題（free_task/）
- 計画は free_task/PLAN.md に書いてある。作業の前に必ず読むこと。PLAN.md にないことを勝手に決めず、必要なら私に確認する。
- day3_cv のコードはコピーして使い、day3_cv 自体は変更しない。
- データ（CIFAR-10、CIFAR-10-C）は free_task/data/ に置き、Git には入れない。取得手順は free_task/download_data.sh にまとめる。
- 学習は Azure VM で行う（実行方法は上の「GPU 実行環境」の節のとおり）。Mac では小さい設定で動作確認だけする。