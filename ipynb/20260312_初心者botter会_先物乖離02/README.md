# 先物乖離 Part 2
* イベント時の分析について、視聴してくれた方が再現できるようipynbデータとcsvデータを同梱しました。
* 初心者の方が分析をたどれるよう、分析テーマごとにnotebookを分けています。

## 使い方

1. `ipynb/` のノートを開く  
2. 上から順に実行する  
   - `20260312_初心者botter会_先物乖離02-1_アルトコイン.ipynb`
   - `20260312_初心者botter会_先物乖離02-2_コスト分析.ipynb`
   - `20260312_初心者botter会_先物乖離02-3_時間足.ipynb`

## フォルダ構成

- `ipynb/`: イベント用ノートブック（02-1 / 02-2 / 02-3）
- `csv/`: ノートブックが参照する事前計算済みデータ

## CSV一覧

- `altcoin_basis_backtest.csv`  
  1銘柄1行の集計データ（`symbol`, `sharpe`, `total_return`, `win_rate`, `max_drawdown`, 年別リターン など）。Part 2-1 の全体統計・分布・ランキングで使用。
- `altcoin_basis_pnl_top10.csv`  
  TOP銘柄の日次時系列データ（`date`, `symbol`, `signal`, `zscore`, `perp_ror`, `pnl`, `cum_hodl`, `cum_pnl`）。Part 2-1 のPnL曲線で使用。
- `altcoin_basis_cost_backtest.csv`  
  コスト込み集計データ（`sharpe_gross/net`, `return_gross/net`, `turnover_pct`, `total_cost` など）。Part 2-2 のコスト前後比較で使用。
- `altcoin_basis_cost_pnl_top10.csv`  
  TOP銘柄のコスト前後PnL時系列（`pnl_gross`, `pnl_net`, `cum_gross`, `cum_net`）。Part 2-2 のPnL曲線で使用。
- `altcoin_basis_tf_backtest.csv`  
  銘柄×時間足（`1h/4h/1d`）の集計データ（`tf_label`, `sharpe_net`, `return_net`, `win_net`, `turnover_pct` など）。Part 2-3 の時間足比較で使用。
- `altcoin_basis_tf_pnl_top10.csv`  
  銘柄×時間足のPnL時系列（`date`, `symbol`, `tf_label`, `pnl_gross`, `pnl_net`, `cum_gross`, `cum_net`）。Part 2-3 の時間足別PnL曲線で使用。

## 免責事項
* 本資料の実行・利用により生成または保存されるデータの管理は利用者の責任で行ってください。
* お客様によるコンテンツの利用等に関して生じうるいかなる損害について責任を負いません。
* 執筆者によって提供されたいかなる見解または意見は当該執筆者自身のその時点における見解や分析であって、当社の見解、分析で
はありません。
* 暗号資産（仮想通貨）は法定通貨ではありません。
* また、法定通貨とは異なり、日本円やドルなどのように国又は特定の者によりその価値を保証されているものではありません。
* 暗号資産の価格の変動等により損失が発生する可能性があります。
* 暗号資産は代価の弁済を受ける者の同意がある場合に限り、代価の弁済のために使用することができます。
* 暗号資産信用取引は、価格の変動等により当初差入れた保証金を上回る損失が発生する可能性があります。十分なご理解の上で、自
己責任にてお取引ください。
* お取引を行う際には、弊社のWebサイトに記載の「契約締結前交付書面兼説明書」「各種規約」「取引ルール」をご確認のうえ、取
引内容を十分に理解し、お客様ご自身の責任と判断を持って行ってください。
