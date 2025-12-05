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
dataSetNames = ['preprocessing_T1_2_3.csv']
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
        model_filename = f"xgb_model_seed_{state}.json"
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

    # 集成：平均預測機率
    print("=== 集成結果（平均預測）===")
    all_proba = [m.predict_proba(X_test)[:, 1] for m in trained_models]
    avg_proba = np.mean(np.vstack(all_proba), axis=0)
    avg_pred = (avg_proba >= 0.5).astype(int)

    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
    acc_avg = accuracy_score(y_test, avg_pred)
    auc_avg = roc_auc_score(y_test, avg_proba)
    print(f"Ensemble Accuracy: {acc_avg:.4f} | Ensemble ROC AUC: {auc_avg:.4f}")
    print("Confusion Matrix (Ensemble):")
    print(confusion_matrix(y_test, avg_pred))
    print("Classification Report (Ensemble):")
    print(classification_report(y_test, avg_pred))


def average_predictions_to_csv(df_origin: pd.DataFrame, trained_models: list[XGBClassifier], threshold: float) -> None:
    df_test_acct = pd.read_csv(os.path.join(dir_path, 'acct_alert.csv'))
    df_test = (df_origin[df_origin['acct'].isin(df_test_acct['acct'])]
               .copy()
               .select_dtypes(include=[np.number])
               .dropna(axis=1))
    scaler = StandardScaler()
    X_test = scaler.fit_transform(df_test.drop(columns=['label']))

    all_proba = [m.predict_proba(X_test)[:, 1] for m in trained_models]
    avg_proba = np.mean(np.vstack(all_proba), axis=0)
    avg_label = (avg_proba >= threshold).astype(int)

    df_pred = pd.DataFrame({'acct': df_test_acct['acct'], 'avg_label': avg_label, 'avg_proba': avg_proba})
    current_time = datetime.now().strftime("%m%d_%H%M")
    output_file = f"xgboost_avg_{current_time}.csv"
    df_pred.to_csv(os.path.join(outputPath, output_file), index=False)
    print(f"(Finish) Averaged predictions saved to {os.path.join(outputPath, output_file)}")

def individual_predictions_to_csv(
    df_origin: pd.DataFrame,
    trained_models: list[XGBClassifier],
    seeds: list[int] | None = None
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

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    for idx, model in enumerate(trained_models):
        seed = seeds[idx] if seeds and idx < len(seeds) else idx + 1

        y_pred = model.predict(X_test)
        # 嘗試機率輸出
        proba = None
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_test)[:, 1]
            except Exception:
                proba = None

        df_out = pd.DataFrame({'acct': df_test_acct['acct'], 'label': y_pred})
        if proba is not None:
            df_out['proba'] = proba

        fname = f"xgboost_seed_{seed}_{ts}.csv"
        df_out.to_csv(os.path.join(outputPath, fname), index=False)
        print(f"(Finish) Individual predictions saved: {os.path.join(outputPath, fname)}")

if __name__ == "__main__":
    # 載入資料
    df = load_data(dataSetNames)

    # 資料預處理
    training_dataSet = PrepareData.prepare_data(df[0].copy(), random_state=random_state, target_pos_ratio=0.5, test_size=0.30)

    # 訓練模型（多隨機種子）
    trained_models = training(random_state=training_random_state, training_dataSet=training_dataSet)
    
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

    # 輸出平均結果
    average_predictions_to_csv(df[0].copy(), trained_models, threshold=0.4)
    individual_predictions_to_csv(df[0].copy(), trained_models, training_random_state)