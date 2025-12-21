# NTUST_CSIE_DS

## 專案結構
```
repo/
├─ DataSet/                  # 原始與處理後資料集 (csv)
├─ XGBoost/                  # XGBoost 模型與訓練筆記本/腳本
│  ├─ Trainning.ipynb
│  ├─ Training.py            # 訓練腳本（含搜尋/訓練/推論輸出）
│  └─ Result/
│     ├─ Model/              # 儲存 .json 模型
│     └─ CSV/                # 儲存預測輸出
├─ Util/                     # 工具模組與視覺化
│  ├─ Evaluater.py           # 評估模型工具
│  ├─ PrepareData.py         # 數據前處理/切分/SMOTE
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

## 使用 Evaluater
- ModelEvaluator 用於：
  - evaluate_model：輸出混淆矩陣、分類報告、ACC、AUC（若模型支援 predict_proba）。
  - plot_pr_threshold：繪製 PR 曲線並回傳 AP 與最佳 F1 的閥值。
  - plot_feature_importance：繪製特徵重要度。

### 範例
```python
from Util.Evaluater import ModelEvaluator
evaluator = ModelEvaluator(result_dir="XGBoost/Result/") 

#輸出分類報告與混淆矩陣
metrics = evaluator.evaluate_model(model, (X_train, X_test, y_train, y_test))

# 繪製 PR 曲線（需分類器支援 predict_proba）
ap, best_thr = evaluator.plot_pr_threshold(model, (X_train, X_test, y_train, y_test), title=file_name)

# 繪製特徵重要度
evaluator.plot_feature_importance(
    model,
    title="Feature Importance",
    importance_type="gain",
    max_num=20
)
```

> 注意：
> - result_dir 會在 repo 根目錄下建立 Logs、PR_Curve、Feature_Importance 子資料夾，用於儲存評估結果與圖表。
> - 若模型不支援 predict_proba，AUC 與 PR 曲線不會計算/繪製。

## 使用 PrepareData
- DataPreparer 提供三種前處理/切分方法，進行數值欄位過濾、標準化（StandardScaler），並於訓練集內使用 SMOTE（視方法而定）。
- DataFrame 需含欄位 "label"（0/1）。

### 1) prepare_cutting
- 依 neg_ratio 抽樣負類，並以 pos_scale 放大正類（SMOTE，僅訓練集）。
```python
from Util.PrepareData import DataPreparer
preparer = DataPreparer(random_state=42)
dataset = preparer.prepare_cutting(
    df,
    neg_ratio=0.015, 
    pos_scale=5, 
    test_size=0.20
    )
```

### 2) prepare_data_pure
- 僅清理/切分/標準化，不做 SMOTE。
```python
from Util.PrepareData import DataPreparer
preparer = DataPreparer(random_state=42)
X_train, X_test, y_train, y_test = preparer.prepare_data_pure(df, test_size=0.2)
```

### 3) prepare_data_smote
- 使訓練集正類比例約為 target_pos_ratio（0~1 之間）。
```python
from Util.PrepareData import DataPreparer
preparer = DataPreparer(random_state=42)
X_train, X_test, y_train, y_test = preparer.prepare_data_smote(
    df,
    target_pos_ratio=0.5,  # 期望訓練集中正類約佔 50%
    test_size=0.2
)
```
