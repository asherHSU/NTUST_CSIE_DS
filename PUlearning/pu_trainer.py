import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score
)
import matplotlib
matplotlib.use('Agg') # Set non-interactive backend for matplotlib
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
from datetime import datetime

# Add Util directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Util.Evaluater import ModelEvaluator

# Define paths
DATA_DIR = "DataSet/"
# Ensure results go to PUlearning/PU_Results/
RESULTS_DIR = "PUlearning/PU_Results/"
ACCT_ALERT_PATH = DATA_DIR + "acct_alert.csv"
ACCT_PREDICT_PATH = DATA_DIR + "acct_predict.csv"
FEATURE_SETS = {
    "T1_basic": DATA_DIR + "preprocessingV2_T1_basic.csv",
    "T1_2": DATA_DIR + "preprocessingV2_T1_2.csv",
    "T1_2_3": DATA_DIR + "preprocessingV2_T1_2_3.csv",
}

def load_and_prepare_data(feature_set_path: str):
    """
    Loads feature data, removes leaky columns, handles non-numeric values,
    and identifies positive/unlabeled samples.
    Returns: train_features_df, train_y_pu, predict_df, train_y_true
    """
    print(f"\n--- Loading and Preparing Data for {feature_set_path.split('/')[-1]} ---")
    
    if not os.path.exists(feature_set_path):
        print(f"!!! ERROR: Data file not found at {feature_set_path}. Skipping.")
        return None, None, None, None

    features_df = pd.read_csv(feature_set_path)

    # --- Extract True Labels for evaluation (NOT for training) ---
    if 'label' in features_df.columns:
        y_true_all = features_df['label']
    else:
        print("!!! WARNING: 'label' column not found for true evaluation. Using dummy 0s.")
        y_true_all = pd.Series(0, index=features_df.index)

    # --- Remove the data leakage sources ---
    leaky_columns = ['label', 'event_date']
    for col in leaky_columns:
        if col in features_df.columns:
            features_df = features_df.drop(columns=[col])
    
    # Handle non-numeric feature columns
    object_cols = features_df.select_dtypes(include='object').columns.tolist()
    if 'acct' in object_cols:
        object_cols.remove('acct')
    
    if len(object_cols) > 0:
        print(f"Found non-numeric feature columns: {object_cols}. Applying one-hot encoding.")
        features_df = pd.get_dummies(features_df, columns=object_cols, dummy_na=False)

    features_df = features_df.fillna(0)

    # Load alert and prediction account lists
    acct_alert_df = pd.read_csv(ACCT_ALERT_PATH)
    positive_account_ids = set(acct_alert_df['acct'].unique())
    acct_predict_df = pd.read_csv(ACCT_PREDICT_PATH)
    predict_account_ids = set(acct_predict_df['acct'].unique())

    if 'acct' in features_df.columns:
        features_df = features_df.set_index('acct')
        y_true_all.index = features_df.index
    else:
        raise ValueError(f"'acct' column not found in {feature_set_path}")

    # Define Positive (P) and Unlabeled (U) sets
    y_pu = pd.Series(-1, index=features_df.index, dtype=int)
    y_pu[y_pu.index.isin(positive_account_ids)] = 1
    
    # Filter for final prediction set
    predict_df = features_df.loc[features_df.index.isin(predict_account_ids)].copy()
    
    # Filter for training/evaluation pool
    train_features_df = features_df.loc[~features_df.index.isin(predict_account_ids)]
    train_y_pu = y_pu.loc[~y_pu.index.isin(predict_account_ids)]
    train_y_true = y_true_all.loc[~y_pu.index.isin(predict_account_ids)]

    print(f"Total samples in source file: {len(features_df)}")
    print(f"Known Positive Samples in training pool: {sum(train_y_pu == 1)}")
    print(f"Unlabeled Samples in training pool: {sum(train_y_pu == -1)}")
    print(f"Samples for final prediction: {len(predict_df)}")
    print("-" * 60)

    return train_features_df, train_y_pu, predict_df, train_y_true

def save_model(model, model_name: str, result_dir: str):
    """
    Saves the trained XGBoost model to a JSON file.
    """
    save_dir = Path(result_dir) / "Model"
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"xgb_model_{model_name}_{datetime.now():%m%d_%H%M%S}.json"
    model.save_model(path)
    print(f"Model saved: {path}")

def run_pu_learning_workflow(X_train_pu, y_train_pu, X_original_predict, y_true_eval, model_name):
    print(f"Starting PU Learning workflow for {model_name}...")
    
    # Initialize Evaluator
    evaluator = ModelEvaluator(RESULTS_DIR)

    # Create a cleaned version of X_original_predict for column alignment
    X_predict_cleaned = X_original_predict.copy()
    
    # Align columns between training and prediction sets
    X_train_pu.columns = [col.replace('[', '').replace(']', '').replace('<', '') for col in X_train_pu.columns]
    X_predict_cleaned.columns = [col.replace('[', '').replace(']', '').replace('<', '') for col in X_predict_cleaned.columns]
    
    train_cols = X_train_pu.columns.tolist()
    predict_cols = X_predict_cleaned.columns.tolist()
    
    for c in (set(train_cols) - set(predict_cols)): X_predict_cleaned[c] = 0
    for c in (set(predict_cols) - set(train_cols)): X_train_pu[c] = 0
    X_predict_cleaned = X_predict_cleaned[train_cols]

    # Step 1: Identify Reliable Negatives (RN) using RandomForest
    print("Step 1: Identifying Reliable Negatives with RandomForestClassifier...")
    y_pu_step1 = (y_train_pu == 1).astype(int)
    num_positives = sum(y_pu_step1)

    if num_positives == 0:
        print("No positive samples in training pool. Aborting.")
        return None

    initial_classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    initial_classifier.fit(X_train_pu, y_pu_step1)
    
    unlabeled_mask = (y_train_pu == -1)
    X_unlabeled = X_train_pu[unlabeled_mask]
    
    if not X_unlabeled.empty:
        unlabeled_proba = initial_classifier.predict_proba(X_unlabeled)[:, 1]
        k = num_positives
        if k > len(unlabeled_proba): k = len(unlabeled_proba)
        sorted_indices = unlabeled_proba.argsort()
        reliable_negative_indices = X_unlabeled.iloc[sorted_indices[:k]].index
        print(f"Identified {len(reliable_negative_indices)} Reliable Negatives.")
    else:
        reliable_negative_indices = []
    
    if len(reliable_negative_indices) == 0:
        print("Could not identify any reliable negatives. Aborting workflow.")
        return None

    # Step 2: Train and Evaluate final classifier
    # We compare the model's performance against the REAL labels (y_true_eval)
    print("\nStep 2: Training and Evaluating with XGBClassifier...")
    positive_indices = X_train_pu[y_train_pu == 1].index
    X_final_train = X_train_pu.loc[positive_indices.union(reliable_negative_indices)]
    y_final_train = pd.Series(0, index=X_final_train.index)
    y_final_train.loc[positive_indices] = 1

    # For a fair "real-world" evaluation, we test on a set that includes the TRUE labels
    # We split the entire pool (excluding predict set) to see how it performs on unseen true data
    X_eval_train, X_eval_test, _, _ = train_test_split(
        X_train_pu, y_true_eval, test_size=0.3, random_state=42, stratify=y_true_eval
    )
    y_eval_train_true = y_true_eval.loc[X_eval_train.index]
    y_eval_test_true = y_true_eval.loc[X_eval_test.index]

    # The actual training still only uses P + RN
    # But we align X_eval_train to only use samples from (P + RN)
    X_train_limited = X_eval_train.loc[X_eval_train.index.isin(X_final_train.index)]
    y_train_limited = y_final_train.loc[X_train_limited.index]

    scale_pos_weight = (y_train_limited == 0).sum() / (y_train_limited == 1).sum()
    
    eval_classifier = XGBClassifier(
        n_estimators=150, max_depth=6, learning_rate=0.1, 
        random_state=42, use_label_encoder=False, 
        eval_metric='aucpr', scale_pos_weight=scale_pos_weight
    )
    eval_classifier.fit(X_train_limited, y_train_limited)
    
    # EVALUATION against TRUE labels
    print("\n--- Final Evaluation Report (Compared to True Labels) ---")
    evaluator.evaluate_model(eval_classifier, (X_eval_train, X_eval_test, y_eval_train_true, y_eval_test_true))
    
    # Plot PR Curve and Feature Importance
    evaluator.plot_pr_threshold(eval_classifier, (X_eval_train, X_eval_test, y_eval_train_true, y_eval_test_true), title=f"PU Learning - {model_name}")
    evaluator.plot_feature_importance(eval_classifier, title=f"Feature Importance (Gain) - {model_name}", importance_type="gain")
    
    # Save the model
    save_model(eval_classifier, model_name, RESULTS_DIR)

    # FINAL PREDICTION on target set
    if not X_original_predict.empty:
        proba = eval_classifier.predict_proba(X_predict_cleaned)[:, 1]
        threshold = 0.5 # Can be adjusted
        out = pd.DataFrame({
            'acct': X_original_predict.index,
            'label': (proba >= threshold).astype(int),
            'proba': proba
        })
        return out
    return None


if __name__ == "__main__":
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    for name, path in FEATURE_SETS.items():
        try:
            X_train, y_pu_train, X_predict_final, y_true_eval = load_and_prepare_data(path)
            
            if X_train is None:
                continue

            results_df = run_pu_learning_workflow(X_train, y_pu_train, X_predict_final, y_true_eval, name)
            
            if results_df is not None:
                output_path = RESULTS_DIR + f"pu_predictions_{name}_XGB.csv"
                results_df.to_csv(output_path, index=False)
                print(f"Predictions for {name} (XGB) saved to {output_path}")

        except Exception as e:
            print(f"Error processing {name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n--- All PU Learning workflows completed ---")


