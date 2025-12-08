# NTUST_CSIE_DS

## 專案結構
```
repo/
├─ DataSet/                  # 原始與處理後資料集 (csv)
├─ XGBoost/                  # XGBoost 模型與訓練筆記本
│  ├─ Trainning.ipynb
│  ├─ xgb_model.json
│  └─ output.csv
├─ Util/                     # 工具模組與視覺化
│  ├─ Evaluater.py           # 評估模型工具                       
│  ├─ Visualization.ipynb    # 視覺化筆記本
│  └─ MergeData.ipynb        # 資料合併筆記本
├─ PreProcessing_T1/         # T1 前處理
│  └─ PreProcessing_T1.ipynb
├─ PreProcessing_T2/         # T2 前處理
│  └─ PreProcessing_T2.ipynb
├─ Preprocessing_T3/         # T3 前處理
│  └─ preprocessing_T3.ipynb
├─ Preprocessing_basic/      # 基礎前處理
│  └─ basic_preprocessing.py
├─ Example/                  # 範例程式
│  └─ example.py
└─ pyproject.toml            # 套件管理與可安裝設定
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

<<<<<<< HEAD
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
=======
## 使用 Evaluater
### def evaluate_model(model, test_data)
- model: 傳入分類器模型（如 XGBClassifier）  
- test_data: 傳入 train_test_split 取得的4個參數，Tuple (X_train, X_test, y_train, y_test)

#### 範例：
```python
import sys
import os
sys.path.append(os.path.dirname(os.getcwd()))
from Util import Evaluater

Evaluater.evaluate_model(model, (X_train, X_test, y_train, y_test))
```
>>>>>>> origin/main
