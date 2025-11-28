#對輸入模型進行評分
def evaluate_model(model, test_data):
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
    X_test, y_test = test_data
    y_pred = model.predict(X_test)

    # Evaluate on test dataset
    conf_matrix_test = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix (Test Data):")
    print(conf_matrix_test)    
    
    class_report_test = classification_report(y_test, y_pred)
    print("Classification Report (Test Data):")
    print(class_report_test)
    
    accuracy_test = accuracy_score(y_test, y_pred)
    print(f"Accuracy (Test Data): {accuracy_test:.4f}")

    # Evaluate on training dataset
    X_train, y_train = model.X_train, model.y_train  # Assuming the model stores training data
    y_train_pred = model.predict(X_train)

    conf_matrix_train = confusion_matrix(y_train, y_train_pred)
    print("Confusion Matrix (Training Data):")
    print(conf_matrix_train)    
    
    class_report_train = classification_report(y_train, y_train_pred)
    print("Classification Report (Training Data):")
    print(class_report_train)
    
    accuracy_train = accuracy_score(y_train, y_train_pred)
    print(f"Accuracy (Training Data): {accuracy_train:.4f}")
    
    return