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