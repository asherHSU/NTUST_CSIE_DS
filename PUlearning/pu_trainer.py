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

# Add Util directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Util.Evaluater import evaluate_model

# Define paths
DATA_DIR = "DataSet/"
RESULTS_DIR = DATA_DIR + "PU_Results/"
ACCT_ALERT_PATH = DATA_DIR + "acct_alert.csv"
ACCT_PREDICT_PATH = DATA_DIR + "acct_predict.csv"
FEATURE_SETS = {
    "T1_basic": DATA_DIR + "preprocessing_T1_basic.csv",
    "T1_2": DATA_DIR + "preprocessing_T1_2.csv",
    "T1_2_3": DATA_DIR + "preprocessing_T1_2_3.csv",
}

def load_and_prepare_data(feature_set_path: str):
    """
    Loads feature data, removes leaky columns, handles non-numeric values,
    and identifies positive/unlabeled samples.
    """
    print(f"\n--- Loading and Preparing Data for {feature_set_path.split('/')[-1]} ---")
    
    if not os.path.exists(feature_set_path):
        print(f"!!! ERROR: Data file not found at {feature_set_path}. Skipping.")
        return None, None, None

    features_df = pd.read_csv(feature_set_path)

    print(f"Columns in loaded features_df BEFORE dropping leaky columns: {features_df.columns.tolist()}")

    # --- Remove the data leakage sources ---
    leaky_columns = ['label', 'event_date']
    for col in leaky_columns:
        if col in features_df.columns:
            print(f"Found and removed '{col}' column to prevent data leakage.")
            features_df = features_df.drop(columns=[col])
        else:
            print(f"'{col}' column not found in features_df (or already removed).")
    
    print(f"Columns in loaded features_df AFTER dropping leaky columns: {features_df.columns.tolist()}")


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
    else:
        raise ValueError(f"'acct' column not found in {feature_set_path}")

    # Check for overlap
    overlap = features_df.index.intersection(predict_account_ids)
    if len(overlap) == 0:
        print("!!! WARNING: Found 0 overlapping accounts between feature set and prediction list. !!!")

    # Define Positive (P) and Unlabeled (U) sets
    y_pu = pd.Series(-1, index=features_df.index, dtype=int)
    y_pu[y_pu.index.isin(positive_account_ids)] = 1
    
    predict_df = features_df.loc[features_df.index.isin(predict_account_ids)].copy()
    train_features_df = features_df.loc[~features_df.index.isin(predict_account_ids)]
    train_y_pu = y_pu.loc[~y_pu.index.isin(predict_account_ids)]

    print(f"Total samples in source file: {len(features_df)}")
    print(f"Known Positive Samples in training pool: {sum(train_y_pu == 1)}")
    print(f"Unlabeled Samples in training pool: {sum(train_y_pu == -1)}")
    print(f"Samples for final prediction: {len(predict_df)}")
    print("-" * 60)

    return train_features_df, train_y_pu, predict_df


def run_pu_learning_workflow(X_train_pu, y_train_pu, X_original_predict, model_name):
    print(f"Starting PU Learning workflow for {model_name}...")

    # Create a cleaned version of X_original_predict for column alignment
    X_predict_cleaned = X_original_predict.copy()
    
    # Align columns between training and prediction sets
    # Sanitize column names for XGBoost
    X_train_pu.columns = [col.replace('[', '').replace(']', '').replace('<', '') for col in X_train_pu.columns]
    X_predict_cleaned.columns = [col.replace('[', '').replace(']', '').replace('<', '') for col in X_predict_cleaned.columns]
    
    train_cols = X_train_pu.columns.tolist()
    predict_cols = X_predict_cleaned.columns.tolist()
    
    missing_in_predict = set(train_cols) - set(predict_cols)
    for c in missing_in_predict: X_predict_cleaned[c] = 0
    missing_in_train = set(predict_cols) - set(train_cols)
    for c in missing_in_train: X_train_pu[c] = 0
    
    X_predict_cleaned = X_predict_cleaned[train_cols]


    # Step 1: Identify Reliable Negatives (RN) using RandomForest
    print("Step 1: Identifying Reliable Negatives with RandomForestClassifier...")
    y_pu_step1 = (y_train_pu == 1).astype(int)
    num_positives = sum(y_pu_step1)

    if num_positives == 0:
        print("No positive samples in training pool. Aborting.")
        return pd.Series([], dtype=float)

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
        return pd.Series(0, index=X_original_predict.index) if not X_original_predict.empty else pd.Series([], dtype=float)

    # Step 2: Train and Evaluate final classifier using XGBoost with K-Fold Cross-Validation
    print("\nStep 2: Training and Evaluating with XGBClassifier...")
    positive_indices = X_train_pu[y_train_pu == 1].index
    
    X_final_data = X_train_pu.loc[positive_indices.union(reliable_negative_indices)]
    y_final_data = pd.Series(0, index=X_final_data.index, name='target')
    y_final_data.loc[positive_indices] = 1

    if X_final_data.empty or len(np.unique(y_final_data)) < 2:
        print("Not enough data or classes to train and evaluate.")
        return pd.Series(0, index=X_original_predict.index) if not X_original_predict.empty else pd.Series([], dtype=float)
        
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics = {'acc': [], 'roc_auc': []}
    
    print(f"\n--- Starting 5-Fold Cross-Validation for Final Classifier ({model_name}) ---")
    for fold, (train_index, test_index) in enumerate(kf.split(X_final_data, y_final_data)):
        print(f"  Fold {fold+1}/5:")
        X_train_fold, X_test_fold = X_final_data.iloc[train_index], X_final_data.iloc[test_index]
        y_train_fold, y_test_fold = y_final_data.iloc[train_index], y_final_data.iloc[test_index]
        
        scale_pos_weight = sum(y_train_fold == 0) / sum(y_train_fold == 1)
        
        fold_classifier = XGBClassifier(
            n_estimators=100, random_state=42, use_label_encoder=False, 
            eval_metric='logloss', scale_pos_weight=scale_pos_weight
        )
        fold_classifier.fit(X_train_fold, y_train_fold)

        y_pred_fold = fold_classifier.predict(X_test_fold)
        y_proba_fold = fold_classifier.predict_proba(X_test_fold)[:, 1]

        acc = accuracy_score(y_test_fold, y_pred_fold)
        roc_auc = roc_auc_score(y_test_fold, y_proba_fold)
        
        print(f"    Accuracy: {acc:.4f}, ROC AUC: {roc_auc:.4f}")
        fold_metrics['acc'].append(acc)
        fold_metrics['roc_auc'].append(roc_auc)
    
    print("\n--- Cross-Validation Results Summary (Per Fold) ---")
    print(f"Average Accuracy: {np.mean(fold_metrics['acc']):.4f} (Std: {np.std(fold_metrics['acc']):.4f})")
    print(f"Average ROC AUC: {np.mean(fold_metrics['roc_auc']):.4f} (Std: {np.std(fold_metrics['roc_auc']):.4f})")
    print("---------------------------------------")

    # --- Final Evaluation Report using evaluate_model (70/30 Split) ---
    print("\n--- Final Evaluation Report (70/30 Split for detailed report/charts) ---")
    X_eval_train, X_eval_test, y_eval_train, y_eval_test = train_test_split(
        X_final_data, y_final_data, test_size=0.3, random_state=42, stratify=y_final_data
    )
    scale_pos_weight_eval = sum(y_eval_train == 0) / sum(y_eval_train == 1)
    eval_classifier = XGBClassifier(
        n_estimators=100, random_state=42, use_label_encoder=False, 
        eval_metric='logloss', scale_pos_weight=scale_pos_weight_eval
    )
    eval_classifier.fit(X_eval_train, y_eval_train)
    evaluate_model(eval_classifier, (X_eval_train, X_eval_test, y_eval_train, y_eval_test))
    
    # Calculate and print Average Precision (AP)
    y_pred_proba_eval = eval_classifier.predict_proba(X_eval_test)[:, 1]
    ap_score = average_precision_score(y_eval_test, y_pred_proba_eval)
    print(f"Average Precision (AP): {ap_score:.4f}")
    
    print("---------------------------------------")

    # --- Plotting PR Curve with Best Threshold ---
    precision, recall, thresholds = precision_recall_curve(y_eval_test, y_pred_proba_eval)
    # calculate f1 score for each threshold, ignoring the last element
    f1_scores = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1])
    f1_scores = np.nan_to_num(f1_scores) # handle division by zero
    # locate the index of the largest f1 score
    best_idx = np.argmax(f1_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label='PR Curve')
    plt.scatter(recall[best_idx], precision[best_idx], marker='o', color='red', s=100, zorder=5, label=f'Best Threshold (F1={f1_scores[best_idx]:.2f})')
    plt.axvline(x=recall[best_idx], color='red', linestyle='--', label=f'Best Recall = {recall[best_idx]:.2f}')
    
    plt.title(f'Precision-Recall Curve for {model_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.grid()
    pr_curve_path = os.path.join(RESULTS_DIR, f"pr_curve_{model_name}.png")
    plt.savefig(pr_curve_path)
    plt.close()
    print(f"PR Curve for {model_name} with best threshold saved to {pr_curve_path}")
    print("---------------------------------------")


    # Retrain on the entire P+RN dataset for final predictions
    print(f"Retraining final classifier on full P+RN data for final predictions...")
    scale_pos_weight_full = sum(y_final_data == 0) / sum(y_final_data == 1) 
    final_classifier_for_submission = XGBClassifier( # Renamed to avoid confusion
        n_estimators=100, random_state=42, use_label_encoder=False,
        eval_metric='logloss', scale_pos_weight=scale_pos_weight_full
    )
    final_classifier_for_submission.fit(X_final_data, y_final_data)

    if not X_original_predict.empty:
        predictions_proba = final_classifier_for_submission.predict_proba(X_predict_cleaned)[:, 1]
        final_predictions = pd.Series(predictions_proba, index=X_original_predict.index)
    else:
        final_predictions = pd.Series([], dtype=float)

    print(f"Finished PU Learning workflow for {model_name}.")
    return final_predictions


if __name__ == "__main__":
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    for name, path in FEATURE_SETS.items():
        try:
            X_train, y_pu_train, X_predict_final = load_and_prepare_data(path)
            
            if X_train is None:
                continue

            predictions = run_pu_learning_workflow(X_train, y_pu_train, X_predict_final, name)
            
            if not predictions.empty:
                output_path = RESULTS_DIR + f"pu_predictions_{name}_XGB.csv"
                predictions_df = pd.DataFrame(predictions, columns=['score'])
                predictions_df.index.name = 'acct'
                predictions_df.to_csv(output_path)
                print(f"Predictions for {name} (XGB) saved to {output_path}")

        except Exception as e:
            print(f"Error processing {name}: {e}")

    print("\n--- All PU Learning workflows completed ---")
