# 初心者botter会_先物乖離03
視聴してくれた方がノートブックだけで再現できるよう、ipynb/ と csv/ を同梱しました。

## 使い方

1. `ipynb/` のノートを開く  
2. 上から順に実行する  
   - `20260326_初心者botter会_先物乖離02-4_上場タイミングとワースト分析.ipynb`
3. ノートと同じフォルダに `homework_report_common.py` を置く  
   - ノート内で `homework_report_common.py` を import して集計・図作成・最終まとめを実行します

## フォルダ構成

- `ipynb/`: イベント用ノートブック（Part 2-4）
- `csv/`: ノートブックが参照する入力データ、および実行時に生成される集計データ
- `homework_report_common.py`: 集計・判定・可視化の共通ロジック

## CSV一覧

### 入力（最低限これだけで実行可）

- `homework_input_summary.csv`  
  1銘柄1行の集計データ（`symbol`, `sharpe_net`, `corr_zscore_next`, `turnover_pct`, `total_cost` など）。
- `homework_input_daily.csv`  
  日次時系列データ（`date`, `symbol`, `event_day`, `pnl_net` など）。

### 実行時に出力される集計CSV

- `altcoin_basis_hw_post_listing_curve_14_59_eventday_summary.csv`  
  event_day別の中央値・分位（宿題1の全体カーブ要約）。
- `altcoin_basis_hw_post_listing_curve_14_59_symbol_daily.csv`  
  銘柄別のevent_day区間日次データ。
- `altcoin_basis_hw_post_listing_curve_14_59_symbol_month.csv`  
  1ヶ月目/2ヶ月目リターンなどの銘柄別スナップショット。
- `altcoin_basis_hw_post_listing_window_symbol_metrics.csv`  
  窓別メトリクス（early/late比較の元データ）。
- `altcoin_basis_hw_post_listing_window_summary.csv`  
  窓別サマリー（中央値・銘柄数など）。
- `altcoin_basis_hw_post_listing_paired_diff.csv`  
  early-late差分のペアデータ。
- `altcoin_basis_hw_worst_diagnostics.csv`  
  ワースト銘柄の診断テーブル（主因タイプ・補助タグの判定に使用）。
- `manifest.json`  
  出力ファイル一覧と設定値。

## 免責事項
* 本資料の実行・利用により生成または保存されるデータの管理は利用者の責任で行ってください。
* お客様によるコンテンツの利用等に関して生じうるいかなる損害について責任を負いません。
* 執筆者によって提供されたいかなる見解または意見は当該執筆者自身のその時点における見解や分析であって、当社の見解、分析ではありません。
* 暗号資産（仮想通貨）は法定通貨ではありません。
* また、法定通貨とは異なり、日本円やドルなどのように国又は特定の者によりその価値を保証されているものではありません。
* 暗号資産の価格の変動等により損失が発生する可能性があります。
* 暗号資産は代価の弁済を受ける者の同意がある場合に限り、代価の弁済のために使用することができます。
* 暗号資産信用取引は、価格の変動等により当初差入れた保証金を上回る損失が発生する可能性があります。十分なご理解の上で、自己責任にてお取引ください。
* お取引を行う際には、弊社のWebサイトに記載の「契約締結前交付書面兼説明書」「各種規約」「取引ルール」をご確認のうえ、取引内容を十分に理解し、お客様ご自身の責任と判断を持って行ってください。
