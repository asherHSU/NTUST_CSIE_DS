import numpy as np
from sklearn.metrics import *
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

class ModelEvaluator:
    def __init__(self):
        self.result_dir = Path(__file__).resolve().parent.parent / "XGBoost" / "Result"/ "Logs"
        self.result_dir.mkdir(parents=True, exist_ok=True)
        # 單次執行共用一個 log 檔
        self.log_path = self.result_dir / f"evaluater_log_{datetime.now():%m%d_%H%M%S}.txt"

    def _log(self, msg: str = "") -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def evaluate_model(self, model, dataset):
        from sklearn.metrics import (
            accuracy_score, confusion_matrix, classification_report, roc_auc_score
        )

        X_train, X_test, y_train, y_test = dataset
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        has_proba = hasattr(model, "predict_proba")
        y_train_proba = model.predict_proba(X_train)[:, 1] if has_proba else None
        y_test_proba = model.predict_proba(X_test)[:, 1] if has_proba else None

        def report(title, y_true, y_pred, y_proba=None):
            lines = [f"{title}:"]
            if y_proba is not None:
                y_pred_thr = (y_proba >= 0.5).astype(int)
                cm = confusion_matrix(y_true, y_pred_thr)
                cr = classification_report(y_true, y_pred_thr)
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

            if y_proba is not None:
                roc = roc_auc_score(y_true, y_proba)
                lines.append(f"ROC AUC Score: {roc:.4f}")
            lines.append("------------------------------------------------------------")
            self._log("\n".join(lines))

        report("訓練資料集", y_train, y_train_pred, y_train_proba)
        report("測試資料集", y_test, y_test_pred, y_test_proba)

        # 保留回傳的 metrics 結構（供程式其他地方使用）
        metrics = {
            "train_acc": accuracy_score(y_train, y_train_pred),
            "test_acc": accuracy_score(y_test, y_test_pred)
        }
        if has_proba:
            metrics["train_auc"] = roc_auc_score(y_train, y_train_proba)
            metrics["test_auc"] = roc_auc_score(y_test, y_test_proba)

        print(f"finish evaluate_model\n")
        return metrics

    def plot_pr_threshold(self, model, dataset, title: str):
        from sklearn.metrics import precision_recall_curve, average_precision_score

        _, X_test, _, y_test = dataset
        if not hasattr(model, "predict_proba"):
            raise ValueError("模型不支援 predict_proba，無法繪製 PR 曲線。")

        y_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)

        # F1 = 2PR / (P+R)，忽略分母為 0
        denom = precision + recall
        f1 = (2 * precision * recall) / np.where(denom == 0, np.nan, denom)

        # 對齊 thresholds 長度（少一個）
        best_idx = int(np.nanargmax(f1[:-1]))
        best_threshold = float(thresholds[best_idx])
        best_p = float(precision[:-1][best_idx])
        best_r = float(recall[:-1][best_idx])
        best_f1 = float(f1[:-1][best_idx])

        # 日誌輸出（僅測試集，與提供範例一致）
        self._log(f"PR 曲線 AP: {ap:.4f}")
        self._log(f"最佳閾值: {best_threshold:.4f} | F1: {best_f1:.4f} | P: {best_p:.4f} | R: {best_r:.4f}")

        # 圖檔輸出
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, label=f'PR curve (AP={ap:.3f})')
        plt.scatter(best_r, best_p, color='red', label=f'Best F1={best_f1:.3f} @ thr={best_threshold:.3f}', zorder=3)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve: {title}')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        save_dir = Path(__file__).resolve().parent.parent / "XGBoost" / "Result" / "PR_Curve"
        save_dir.mkdir(parents=True, exist_ok=True)
        file = save_dir / f'PR_curve_{datetime.now():%m%d_%H%M%S}.png'
        plt.savefig(file)
        plt.close()
        print(f"finish plot_pr_threshold\n")
        return ap, best_threshold

    def plot_feature_importance(self, model, title: str = "Feature Importance",
                                importance_type: str = "gain", max_num: int = 20) -> str:
        import xgboost

        fig, ax = plt.subplots(figsize=(8, 6))
        xgboost.plot_importance(
            model,
            ax=ax,
            importance_type=importance_type,
            max_num_features=max_num,
            show_values=False
        )
        ax.set_title(title)
        plt.tight_layout()

        save_dir = Path(__file__).resolve().parent.parent / "XGBoost" / "Result" / "Feature_Importance"
        save_dir.mkdir(parents=True, exist_ok=True)
        file = save_dir / f"feature_importance_{importance_type}_{datetime.now():%m%d_%H%M%S}.png"
        plt.savefig(file, dpi=500)
        plt.close(fig)

        print(f"finish plot_feature_importance\n")
        return str(file)
