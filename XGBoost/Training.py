import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # 放在檔頭 imports（若尚未匯入）
from Util.PrepareData import DataPreparer
from Util.Evaluater import ModelEvaluator

IS_SAVE_RESULT = True

print("XGBoost version:", xgboost.__version__)
print(xgboost.build_info())

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

class XGBoostTrainer:
    def __init__(self, random_state: int):
        self.random_state = random_state

    def train_with_search(self, dataset, param_grid):
        X_train, _, y_train, _ = dataset
        
        #split validation set from training set
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.3, stratify=y_train, random_state=self.random_state
        )
        
        base_model = XGBClassifier(
            objective="binary:logistic",
            tree_method="hist",
            eval_metric="aucpr",
            device="gpu",
            random_state=self.random_state,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()
        )

        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_grid,
            scoring="f1",
            cv=3,
            n_jobs=1,
            verbose=2,
            random_state=self.random_state
        )

        search.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        print(f"Best params: {search.best_params_}, Best score: {search.best_score_:.4f}")
        print("finish hyperparameter search\n")
        return search.best_estimator_
    
    def train(self, dataset, params: dict):
        X_train, _, y_train, _ = dataset
        
        #split validation set from training set
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.3, stratify=y_train, random_state=self.random_state
        )

        model = XGBClassifier(
            **params,
            objective="binary:logistic",
            tree_method="hist",
            eval_metric="aucpr",
            device="gpu",
            random_state=self.random_state,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum()
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        print("finish training\n")
        return model

    @staticmethod
    def save_model(model):
        save_dir = Path(__file__).resolve().parent / "Result" / "Model"
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"xgb_model_{datetime.now():%m%d_%H%M%S}.json"
        model.save_model(path)
        print(f"model saved: {path}")

def predictions_result(
    df_origin: pd.DataFrame,
    model: XGBClassifier,
    data_dir: str,
    threshold: float = 0.5,
    alert_file: str = "acct_alert.csv"
) -> str:
    """
    針對 acct_alert.csv 指定的帳戶輸出預測結果 CSV。
    欄位：acct, label, proba（若模型不支援 predict_proba，則省略 proba）
    """
    # 讀取測試帳戶清單
    alert_path = Path(data_dir) / alert_file
    df_test_acct = pd.read_csv(alert_path)

    # 過濾原始資料，僅保留指定帳戶與數值欄位；移除空欄
    df_test = (
        df_origin[df_origin['acct'].isin(df_test_acct['acct'])]
        .copy()
        .select_dtypes(include=[np.number])
        .dropna(axis=1, how='all')
    )

    # 建立特徵矩陣（若有 label 欄位則移除）
    feature_cols = [c for c in df_test.columns if c != 'label']
    scaler = StandardScaler()
    X_test = scaler.fit_transform(df_test[feature_cols])

    # 進行預測
    proba = None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1]
            y_pred = (proba >= threshold).astype(int)
        else:
            y_pred = model.predict(X_test)
    except Exception:
        # 發生例外時退回直接預測
        y_pred = model.predict(X_test)

    # 組裝輸出
    out = pd.DataFrame({
        'acct': df_test_acct['acct'],
        'label': y_pred
    })
    if proba is not None:
        out['proba'] = proba

    # 儲存檔案
    save_dir = Path(__file__).resolve().parent / "Result" / "CSV"
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / f"predictions_{datetime.now():%m%d_%H%M%S}.csv"
    out.to_csv(file_path, index=False)
    print(f"Finish Individual predictions\n")
    return str(file_path)

if __name__ == "__main__":
    random_state = 42
    data_dir = "C:/school/SchoolProgram/NTUST_CSIE_DS/DataSet"
    dataset_names = [
    "preprocessing_T1_2_3.csv",
    "preprocessing_T1_2.csv",
    "preprocessing_T1_basic.csv"
    ]

    param_grid = {
    "n_estimators": [550, 600, 650],
    "max_depth": [6, 7, 8],
    "learning_rate": [0.2, 0.22, 0.25],
    "subsample": [0.95, 1.0],
    "colsample_bytree": [0.82, 0.85, 0.87]
    }

    preparer = DataPreparer(random_state)
    evaluator = ModelEvaluator()
    trainer = XGBoostTrainer(random_state)

    for file_name in dataset_names:
        print(f"===== start training {file_name} =====")
        df = pd.read_csv(Path(data_dir) / file_name)
        df.drop(columns=['acct_type_x', 'acct_type_y'], inplace=True, errors='ignore')
        print(f"Loaded {file_name}, shape: {df.shape}, Positive ratio: {(df['label'] == 1).mean():.2%}\n")
        
        # dataset = preparer.prepare_data_pure(df,test_size=0.20)
        dataset = preparer.prepare_cutting(df,neg_ratio=0.015, pos_scale=5, test_size=0.20)
        # dataset = preparer.prepare_data_smote(df,target_pos_ratio=0.33, test_size=0.20)
        
        model = trainer.train_with_search(dataset, param_grid)
        # model = trainer.train(dataset, params={
        #     "n_estimators": 650,
        #     "max_depth": 7,
        #     "learning_rate": 0.25,
        #     "subsample": 0.95,
        #     "colsample_bytree": 0.85
        # })
        
        if IS_SAVE_RESULT:
            trainer.save_model(model)
            metrics = evaluator.evaluate_model(model, dataset)
            ap, thresholds = evaluator.plot_pr_threshold(model, dataset, title=file_name)
            evaluator.plot_feature_importance(
                model,
                title="Feature Importance",
                importance_type="gain",
                max_num=20
            )
            
            predictions_result(df_origin=df,model=model,data_dir=data_dir,threshold=0.2,alert_file="acct_alert.csv")
        
        print(f"===== finish training {file_name} =====")