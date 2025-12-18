from typing import Tuple
from pathlib import Path
import numpy as np
from datetime import datetime  # 新增

# 建立/指定結果與日誌檔路徑
RESULT_DIR = Path(__file__).resolve().parent.parent / "XGBoost" / "Result"/ "Evaluater_Logs"
RESULT_DIR.mkdir(parents=True, exist_ok=True)
_LOG_PATH = RESULT_DIR / f"evaluater_log_{datetime.now().strftime('%m%d_%H%M%S')}.txt"

def _log(msg: str = "") -> None:
    with _LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")

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
        roc_auc_score
    )

    X_train, X_test, y_train, y_test = test_data
    
    # 基本預測
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
        lines = [f"{title}:"]
        if y_proba is not None:
            # 使用概率值進行分類
            y_pred_proba = (y_proba >= 0.5).astype(int)
            cm = confusion_matrix(y_true, y_pred_proba)
            cr = classification_report(y_true, y_pred_proba)
            lines.append("Confusion Matrix (using probabilities):")
            lines.append(np.array2string(cm))
            lines.append("")
            lines.append("Classification Report (using probabilities):")
            lines.append(cr)
        else:
            cm = confusion_matrix(y_true, y_pred)
            cr = classification_report(y_true, y_pred)
            lines.append("Confusion Matrix:")
            lines.append(np.array2string(cm))
            lines.append("")
            lines.append("Classification Report:")
            lines.append(cr)

        acc = accuracy_score(y_true, y_pred)
        lines.append(f"Accuracy: {acc:.4f}")

        roc = None
        if y_proba is not None:
            roc = roc_auc_score(y_true, y_proba)
            lines.append(f"ROC AUC Score: {roc:.4f}")
        lines.append("-" * 60)

        _log("\n".join(lines))
        return acc, roc

    # 訓練集
    train_acc, train_roc = report(
        "訓練資料集",
        y_train,
        y_train_pred,
        y_train_proba if has_proba else None
    )

    # 測試集
    test_acc, test_roc = report(
        "測試資料集",
        y_test,
        y_test_pred,
        y_test_proba if has_proba else None
    )

    print("finish evaluate_model")
    return train_acc, train_roc, test_acc, test_roc

def plot_pr_and_best_threshold(model, test_data, title) -> Tuple[float, float, float]:
    """
    繪製 PR 曲線，並找出最佳閾值（以 F1 最大為準）。
    回傳 (best_threshold, best_f1, ap)：
      - best_threshold: 使 F1 最大的機率閾值
      - best_f1: 對應的 F1 分數
      - ap: Average Precision (AP)
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, average_precision_score
    from datetime import datetime

    X_train, X_test, y_train, y_test = test_data

    if not hasattr(model, "predict_proba"):
        raise ValueError("模型不支援 predict_proba，無法繪製 PR 曲線。")

    y_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)

    # 計算 F1 = 2PR / (P+R)，忽略 P+R=0 的情況
    denom = (precision + recall)
    f1 = (2 * precision * recall) / np.where(denom == 0, np.nan, denom)

    # thresholds 的長度比 precision/recall 少 1，對齊索引
    valid_idx = ~np.isnan(f1[:-1])
    f1_valid = f1[:-1][valid_idx]
    thresholds_valid = thresholds[valid_idx]
    precision_valid = precision[:-1][valid_idx]
    recall_valid = recall[:-1][valid_idx]

    # 找最大 F1
    best_idx = int(np.nanargmax(f1_valid))
    best_threshold = float(thresholds_valid[best_idx])
    best_f1 = float(f1_valid[best_idx])
    best_p = float(precision_valid[best_idx])
    best_r = float(recall_valid[best_idx])

    _log(f"PR 曲線 AP: {ap:.4f}")
    _log(f"最佳閾值: {best_threshold:.4f} | F1: {best_f1:.4f} | P: {best_p:.4f} | R: {best_r:.4f}")

    # 繪圖
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f'PR curve (AP={ap:.3f})')
    plt.scatter(best_r, best_p, color='red', label=f'Best F1={best_f1:.3f} @ thr={best_threshold:.3f}', zorder=3)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve: {title}')  # Add title to the graph
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    timestamp = datetime.now().strftime("%m%d_%H%M%S")  # Generate timestamp
    save_dir = Path(__file__).resolve().parent.parent / "XGBoost" / "Result" / "PR_Curve"
    save_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    file_name = save_dir / f'PR_curve_{timestamp}.png'  # Generate file name with timestamp
    plt.savefig(file_name)  # Save the graph as a file
    plt.close()  # Close the plot to avoid displaying it
    print("finish plot_pr_and_best_threshold")

    return best_threshold, best_f1, ap

def plot_feature_importance(model,
                            title: str = "Feature Importance",
                            importance_type: str = "gain",
                            max_num: int = 20,
                            save_dir: str | None = None) -> str:
    """
    繪製並儲存 XGBoost 特徵重要性圖。
    importance_type: 'weight' | 'gain' | 'cover' | 'total_gain' | 'total_cover'
    """
    import matplotlib.pyplot as plt
    import xgboost
    from datetime import datetime

    fig, ax = plt.subplots(figsize=(8, 6))
    xgboost.plot_importance(
        model, ax=ax, importance_type=importance_type, max_num_features=max_num, show_values=False
    )
    ax.set_title(title)
    plt.tight_layout()

    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    # 與原先 Training.py 行為一致：輸出到 XGBoost/Result/Feature_Importance
    save_dir_path = Path(__file__).resolve().parent.parent / "XGBoost" / "Result" / "Feature_Importance"
    save_dir_path.mkdir(parents=True, exist_ok=True)
    file_name = save_dir_path / f'feature_importance_{importance_type}_{timestamp}.png'
    plt.savefig(file_name, dpi=150)
    plt.close(fig)
    print("finish plot_feature_importance")
    return str(file_name)
