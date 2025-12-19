import os
import random
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in map(str, sys.path):
    sys.path.insert(0, str(PROJECT_ROOT))
# Custom project imports (unchanged)

from Util import Evaluater, PrepareData

#===========================================================
# Global Settings
#===========================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in map(str, sys.path):
    sys.path.insert(0, str(PROJECT_ROOT))

dir_path = "D:\\讀書==\\NTUST\\大三上\\資料科學\\NTUST_CSIE_DS\\DataSet"
dataSetNames = ['preprocessing_T1_basic.csv', 'preprocessing_T1_2.csv', 'preprocessing_T1_2_3.csv']
outputPath = ''
random_state = 42
training_random_state = [42]

#===========================================================
# Data Loader
#===========================================================
def load_data(fileNames: list) -> list[pd.DataFrame]:
    results = []
    for file in fileNames:
        df = pd.read_csv(os.path.join(dir_path, file))
        results.append(df)
        print(f"Loaded {file}, shape: {df.shape}")
    return results

#===========================================================
# Train Isolation Forest (XGB → IF replacement)
#===========================================================
def training_IF(random_state_list=[42], training_dataSet=None) -> list[IsolationForest]:

    if training_dataSet is None:
        raise ValueError("training_dataSet 必須提供 (X_train, X_test, y_train, y_test)")

    X_train, X_test, y_train, y_test = training_dataSet

    trained_models = []

    # IMPORTANT:
    # Isolation Forest is unsupervised → train only on NORMAL samples.
    X_train_normals = X_train[y_train == 0]

    print(f"Training IsolationForest on {len(X_train_normals)} normal samples")

    for seed in random_state_list:
        iso = IsolationForest(
            n_estimators=650,          # Kept high for robustness
            max_features=0.8,          # <--- CHANGED: Increased from 0.1 for better feature coverage
            contamination=0.01,        # <--- CHANGED: Increased from 0.005 to test a wider threshold
            max_samples='auto',        # Kept optimal default
            random_state=seed,
            n_jobs=-1
        )
        iso.fit(X_train_normals)

        # ... (rest of the function remains the same) ...
        # Save "model"
        ts = datetime.now().strftime("%m%d_%H%M%S")
        model_filename = f"isoForest_{ts}.npy"
        np.save(model_filename, iso)
        print(f"Isolation Forest saved as: {model_filename}")

        trained_models.append(iso)

    return trained_models

#===========================================================
# Evaluate Models
#===========================================================
def evaluate_ensemble(trained_models: list[IsolationForest], test_data: tuple) -> None:
    X_train, X_test, y_train, y_test = test_data

    print("===  (Model Evaluater) ===")
    for i, model in enumerate(trained_models, start=1):
        print(f"Model {i} reuslts:")
        Evaluater.evaluate_model(model, test_data)

#===========================================================
# Output Predictions to CSV
#===========================================================
def predictions_to_csv(
    df_origin: pd.DataFrame,
    trained_models: list,
    seeds=None,
    threshold: float = 0.5
):

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

    for i, model in enumerate(trained_models):
        if hasattr(model, "predict_proba"):
            try:
                scores = model.decision_function(X_test)
                threshold = np.percentile(scores, 1)  # or contamination * 100
                y_pred = (scores < threshold).astype(int)
            except Exception:
                scores = model.decision_function(X_test)
                threshold = np.percentile(scores, 1)  # or contamination * 100
                y_pred = (scores < threshold).astype(int)
        else:
            scores = model.decision_function(X_test)
            threshold = np.percentile(scores, 1)  # or contamination * 100
            y_pred = (scores < threshold).astype(int)

        df_out = pd.DataFrame({'acct': df_test_acct['acct'], 'label': y_pred})

        fname = f"isolationForest_seed_{ts}.csv"
        df_out.to_csv(os.path.join(outputPath, fname), index=False)
        print(f"(Finish) Individual predictions saved: {os.path.join(outputPath, fname)}")


#===========================================================
# Main Execution
#===========================================================
if __name__ == "__main__":
    df_list = load_data(dataSetNames)

    for i, data in enumerate(df_list):
        print("="*10 + f" DataSet {i+1} " + "="*10)

        # CHECK THIS STEP: 
        # Ensure 'prepare_data_cutting' does not aggressively undersample
        # the normal class (y_train == 0), as this set is used to train IF.
        training_dataSet = PrepareData.prepare_data_cutting(
            data.copy(),
            random_state=random_state,
            neg_ratio=0.01, 
            pos_scale=3, 
            test_size=0.30
        )

        trained_models = training_IF(
            random_state_list=training_random_state,
            training_dataSet=training_dataSet
        )

        evaluate_ensemble(trained_models, training_dataSet)

        predictions_to_csv(
            data.copy(), 
            trained_models,
            seeds=training_random_state,
            threshold=0.0
        )
