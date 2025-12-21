from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np

# 對輸入模型進行評分
def evaluate_model(model, test_data):
    """
    評估分類模型於訓練與測試資料的表現。
    data: (X_train, X_test, y_train, y_test)

    Returns:
        tuple: (train_acc, train_roc_auc, test_acc, test_roc_auc)
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        classification_report,
        roc_auc_score,
    )
    from sklearn.ensemble import IsolationForest

    X_train, X_test, y_train, y_test = test_data
    
    # 基本預測
    if isinstance(model, IsolationForest):
        train_scores = model.decision_function(X_train)
        train_threshold = np.percentile(train_scores, 1)
        y_train_pred = (train_scores < train_threshold).astype(int)
        
        test_scores = model.decision_function(X_test)
        test_threshold = np.percentile(test_scores, 1)
        y_test_pred = (test_scores < test_threshold).astype(int)
    else:
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

    # 若模型支援 predict_proba，則計算 ROC AUC
    has_proba = hasattr(model, "predict_proba")
    y_train_proba = None
    y_test_proba = None
    if has_proba:
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]

    def report(title, y_true, y_pred, y_proba=None):
        print(f"{title}:")
        if y_proba is not None:
            # 使用概率值進行分類
            y_pred_proba = (y_proba >= 0.5).astype(int)
            print("Confusion Matrix (using probabilities):")
            print(confusion_matrix(y_true, y_pred_proba))
            print("\nClassification Report (using probabilities):")
            print(classification_report(y_true, y_pred_proba, zero_division=0))
        else:
            print("Confusion Matrix:")
            print(confusion_matrix(y_true, y_pred))
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, zero_division=0))

        acc = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {acc:.4f}")

        roc = None
        if y_proba is not None:
            roc = roc_auc_score(y_true, y_proba)
            
            print(f"ROC AUC Score: {roc:.4f}")
        print("-" * 60)
        return acc, roc

    # 訓練集
    train_acc, train_roc = report(
        "Training dataset",
        y_train,
        y_train_pred,
        y_train_proba if has_proba else None
    )

    # 測試集
    test_acc, test_roc = report(
        "Testing dataset",
        y_test,
        y_test_pred,
        y_test_proba if has_proba else None
    )

    return train_acc, train_roc, test_acc, test_roc

def plot_pr_curve_if(model, test_data, title="PR Curve"):
    X_train, X_test, y_train, y_test = test_data

    # 1. Get anomaly scores
    scores = model.decision_function(X_test)

    # 2. Convert to "anomaly probability-like" score
    #    Higher = more anomalous
    anomaly_scores = -scores

    # 3. Compute PR curve
    precision, recall, thresholds = precision_recall_curve(
        y_test,
        anomaly_scores
    )

    # 4. Average Precision (AP)
    ap = average_precision_score(y_test, anomaly_scores)

    # 5. Plot
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

    return precision, recall, thresholds, ap

def plot_feature_importance(self, model, title: str = "Feature Importance",
                                importance_type: str = "gain", max_num: int = 20) -> str:
        if not hasattr(model, "feature_importances_") and not hasattr(model, "get_booster"):
            raise ValueError("The model does not support feature importance plotting.")

        fig, ax = plt.subplots(figsize=(8, 6))

        if hasattr(model, "get_booster"):  # For XGBoost models
            import xgboost
            xgboost.plot_importance(
                model,
                ax=ax,
                importance_type=importance_type,
                max_num_features=max_num,
                show_values=False
            )
        elif hasattr(model, "feature_importances_"):  # For models with feature_importances_ attribute
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1][:max_num]
            ax.barh(range(len(indices)), importances[indices], align="center")
            ax.set_yticks(range(len(indices)))
            ax.set_yticklabels([f"Feature {i}" for i in indices])
            ax.invert_yaxis()
            ax.set_title(title)

        plt.tight_layout()

        save_dir = self.result_dir / "Feature_Importance"
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.close(fig)

        print(f"finish plot_feature_importance\n")
        return

def evaluate_alert():
    pass