#對輸入模型進行評分
def evaluate_model(model, test_data):
    """
    評估分類模型於訓練與測試資料的表現。
    data: (X_train, X_test, y_train, y_test)
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
    if has_proba:
        y_train_proba = model.predict_proba(X_train)[:, 1]
        y_test_proba = model.predict_proba(X_test)[:, 1]

    def report(title, y_true, y_pred, y_proba=None):
        print(f"{title}:")
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

        if y_proba is not None:
            print(f"ROC AUC Score: {roc_auc_score(y_true, y_proba):.4f}")
        print("-" * 60)

    # 訓練集
    report(
        "訓練資料集",
        y_train,
        y_train_pred,
        y_train_proba if has_proba else None
    )

    # 測試集
    report(
        "測試資料集",
        y_test,
        y_test_pred,
        y_test_proba if has_proba else None
    )
    return