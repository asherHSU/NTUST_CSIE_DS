# NTUST_CSIE_DS

## 專案結構
```
repo/
├─ DataSet/ # 原始資料集
├─ XGBoost/ # XGBoost 模型
└─ pyproject.toml # 包管理
```

## 安裝依賴
```bash
pip install -e .
```

## conda 環境
```bash
conda create -n ntust_ds python=3.11 -y
conda activate ntust_ds
```

## 資料表說明
### T1
- 是否高頻轉出、轉入
- 是否有外幣 
- 接收多少不同來源帳戶 
- 轉出多少不同收款帳戶 

### T2
- 匯出匯入差額 
- 是否夜間交易 
- 匯款, 收款的交易比例(次數、金額)
- 是否短時間大量接收帳戶

### T3
- 轉出、轉入金額資訊：max, min, average, var, std
- 互相轉帳的交易占比 
- 是否單一交易通路 
- 首次交易是否高轉帳金額

### basic
- 幣值轉換
- 是否跨行轉帳
- 日期時間轉換成timestamp
- 交易次數