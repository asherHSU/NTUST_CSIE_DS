import os
import random
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from pathlib import Path

print("XGBoost version:", xgboost.__version__)
print(xgboost.build_info())

BASE_DIR = Path(__file__).resolve().parent

PROJECT_ROOT = BASE_DIR.parent
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# 從 Util 套件匯入
from Util import Evaluater, PrepareData
#=============================================================================

# Global variables
dir_path = "C:\\school\\SchoolProgram\\NTUST_CSIE_DS\\DataSet"
dataSetNames = ['preprocessing_T1_basic.csv', 'preprocessing_T1_2.csv', 'preprocessing_T1_2_3.csv']
outputPath = ''
random_state = 42
training_random_state = [42]

def load_data(fileNames: list) -> list[pd.DataFrame]:
    df_results = []
    for file in fileNames:
        df = pd.read_csv(os.path.join(dir_path, file))
        df_results.append(df)
        print(f"Loaded {file}, shape: {df.shape}")
    return df_results

def training_hyperParameter(random_state: list = [42], training_dataSet: tuple = None) -> list[XGBClassifier]:
    """
    訓練 XGBoost 模型，並根據輸入的隨機種子和訓練資料集進行多次訓練。

    Args:
        random_state (list): 隨機種子列表，用於多次訓練。
        training_dataSet (tuple): 包含 (X_train, X_test, y_train, y_test) 的訓練資料集。

    Returns:
        list[XGBClassifier]: 訓練完成的 XGBoost 分類器列表。
    """
    if training_dataSet is None:
        raise ValueError("training_dataSet 參數不能為 None，請提供 (X_train, X_test, y_train, y_test) 資料。")

    X_train, X_test, y_train, y_test = training_dataSet
    trained_models = []

    for state in random_state:
        # 定義參數網格
        param_grid = {
            'n_estimators': [550, 600, 650],
            'max_depth': [6, 7, 8],
            'learning_rate': [0.2, 0.22, 0.25],
            'subsample': [0.95, 1.0],
            'colsample_bytree': [0.82,0.85, 0.87]
        }

        # 初始化 XGBoost 分類器
        xgb_clf = XGBClassifier(
            objective='binary:logistic',
            random_state=state,
            tree_method='hist',
            eval_metric=['aucpr'],
            scale_pos_weight= sum(y_train == 0) / sum(y_train == 1),
            device='gpu'
        )

        print(f"開始訓練 XGBoost 模型，隨機種子: {state}")
        # 使用 GridSearchCV 搜索最佳參數
        grid_search = RandomizedSearchCV(
            estimator=xgb_clf,
            param_distributions=param_grid,
            cv=3,
            scoring='average_precision',
            n_jobs=1,
            verbose=3,
            random_state=state
        )
        grid_search.fit(X_train, y_train)

        # 使用最佳參數初始化分類器
        best_params, best_score = grid_search.best_params_, grid_search.best_score_
        print(f"隨機種子: {state}，最佳參數: {best_params}，最佳分數: {best_score:.4f}")

        xgb_clf = XGBClassifier(
            **best_params,
            objective='binary:logistic',
            random_state=state,
            tree_method='hist',
            device='cpu',            
            scale_pos_weight= sum(y_train == 0) / sum(y_train == 1),
            eval_metric=['aucpr']
        )
        xgb_clf.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)], verbose=False
        )

        trained_models.append(xgb_clf)

        # 儲存模型
        ts = datetime.now().strftime("%m%d_%H%M%S")
        model_filename = f"xgb_model_{ts}.json"
        xgb_clf.save_model(model_filename)
        print(f"隨機種子: {state}，模型已儲存為 {model_filename}。")

    print("所有模型訓練完成！")
    return trained_models

def training(random_state: list = [42], training_dataSet: tuple = None) -> list[XGBClassifier]:
    """
    訓練 XGBoost 模型，並根據輸入的隨機種子和訓練資料集進行多次訓練。

    Args:
        random_state (list): 隨機種子列表，用於多次訓練。
        training_dataSet (tuple): 包含 (X_train, X_test, y_train, y_test) 的訓練資料集。

    Returns:
        list[XGBClassifier]: 訓練完成的 XGBoost 分類器列表。
    """
    if training_dataSet is None:
        raise ValueError("training_dataSet 參數不能為 None，請提供 (X_train, X_test, y_train, y_test) 資料。")

    X_train, X_test, y_train, y_test = training_dataSet
    trained_models = []

    for state in random_state:
        xgb_clf = XGBClassifier(
            objective='binary:logistic',
            random_state=state,
            tree_method='hist',
            device='cpu',            
            scale_pos_weight= sum(y_train == 0) / sum(y_train == 1),
            eval_metric=['aucpr'],
            n_estimators= 650,
            max_depth= 7,
            learning_rate= 0.22,
            subsample= 0.95,
            colsample_bytree= 0.85
        )
        xgb_clf.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)], verbose=False
        )

        trained_models.append(xgb_clf)

        # 儲存模型
        ts = datetime.now().strftime("%m%d_%H%M%S")
        model_filename = f"xgb_model_{ts}.json"
        xgb_clf.save_model(model_filename)
        print(f"隨機種子: {state}，模型已儲存為 {model_filename}。")

    print("所有模型訓練完成！")
    return trained_models

def evaluate_ensemble(trained_models: list[XGBClassifier], test_data: tuple) -> None:
    """
    使用 Evaluater.evaluate_model 評每個模型並計算平均，
    另計算平均預測機率的集成表現。
    """
    X_train, X_test, y_train, y_test = test_data

    print("=== 個別模型評估（Evaluater）===")
    for i, model in enumerate(trained_models, start=1):
        print(f"模型 {i} 評估結果:")
        Evaluater.evaluate_model(model, test_data)

def predictions_to_csv(
    df_origin: pd.DataFrame,
    trained_models: list[XGBClassifier],
    seeds: list[int] | None = None,
    threshold: float = 0.5
) -> None:
    """
    針對每個模型分別輸出預測結果一份 CSV。
    若提供 seeds，檔名會包含對應的隨機種子；否則用序號標識。

    檔案命名：xgboost_seed_{seed}_{YYYYMMDD_HHMM}.csv
    欄位：acct, label, proba（若模型不支援 predict_proba，則省略 proba）
    """
    df_test_acct = pd.read_csv(os.path.join(dir_path, 'acct_alert.csv'))
    df_test = (
        df_origin[df_origin['acct'].isin(df_test_acct['acct'])]
        .copy()
        .select_dtypes(include=[np.number])
        .dropna(axis=1)
    )
    scaler = StandardScaler()
    X_test = scaler.fit_transform(df_test.drop(columns=['label']))

    ts = datetime.now().strftime("%m%d_%H%M%S")
    for idx, model in enumerate(trained_models):

        # 使用機率進行預測
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_test)[:, 1]
                y_pred = (proba >= threshold).astype(int)  # 使用指定的預測閾值
            except Exception:
                proba = None
                y_pred = model.predict(X_test)
        else:
            proba = None
            y_pred = model.predict(X_test)

        df_out = pd.DataFrame({'acct': df_test_acct['acct'], 'label': y_pred})
        if proba is not None:
            df_out['proba'] = proba

        fname = f"xgboost_seed_{ts}.csv"
        df_out.to_csv(os.path.join(outputPath, fname), index=False)
        print(f"(Finish) Individual predictions saved: {os.path.join(outputPath, fname)}")

if __name__ == "__main__":
    # 載入資料
    df = load_data(dataSetNames)
    
    for i, data in enumerate(df):
        print(f"{'='*10}DataSet {i+1} shape: {data.shape}, Positive ratio: {(data['label'] == 1).mean():.2%}{'='*10}")

        # 資料預處理
        training_dataSet = PrepareData.prepare_data_cutting(data.copy(), random_state=random_state, neg_ratio=0.01, pos_scale=3, test_size=0.30)

        # 訓練模型
        trained_models = training_hyperParameter(random_state=training_random_state, training_dataSet=training_dataSet)
        
        #load trained models
        # trained_models = []
        # for state in training_random_state:
        #     model_filename = f"xgb_model_seed_{state}.json"
        #     xgb_clf = XGBClassifier()
        #     xgb_clf.load_model("XGBoost/" + model_filename)
        #     trained_models.append(xgb_clf)
        #     print(f"Loaded model from {model_filename}")

        # 個別模型與平均指標評估
        evaluate_ensemble(trained_models, training_dataSet)

        # 針對第一個模型繪製 PR 曲線並取得最佳閾值
        try:
            best_thr, best_f1, ap = Evaluater.plot_pr_and_best_threshold(trained_models[0], training_dataSet, title=f"DataSet {i+1}")
        except Exception as e:
            print(f"繪製 PR 曲線失敗: {e}")
            best_thr = 0.5  # 預設閾值
        # 輸出預測結果至 CSV
        predictions_to_csv(data.copy(), trained_models, seeds=training_random_state, threshold=0.25)