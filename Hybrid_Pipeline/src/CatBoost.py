from catboost import CatBoostClassifier   
import numpy as np
from sklearn.model_selection import train_test_split,RandomizedSearchCV

class catboost_model:
    def __init__(self):
        pass

    # 定義 CatBoost 模型
    # auto_class_weights='Balanced' 對於異常偵測極為重要，自動處理樣本不平衡
    model = CatBoostClassifier(
        auto_class_weights='Balanced', # 覆蓋預設值 (None -> Balanced)
        eval_metric='AUC',             # 覆蓋預設值 (Logloss -> AUC)
    )

    # 定義參數網格 (Parameter Grid)
    param_dist = {
        'iterations': [200, 500],           # 樹的數量
        'learning_rate': [0.01, 0.05, 0.1], # 學習率
        'depth': [4, 6, 8],                 # 樹的深度 (太深容易 overfitting)
        'l2_leaf_reg': [1, 3, 5, 7],        # L2 正則化係數
        'random_strength': [1, 5, 10],      # 防止過擬合的隨機性強度
        'bagging_temperature': [0, 1]       # 貝葉斯自助抽樣的強度
    }

    @staticmethod
    def catBoost_cut(X_reliable_negatives,X_known_anomalies):
        # 建立&切分訓練集
        # 建構訓練集：可靠正常 (Label 0) + 已知異常 (Label 1)
        X_train_final = np.vstack([X_reliable_negatives, X_known_anomalies])
        y_train_final = np.hstack([np.zeros(len(X_reliable_negatives)), np.ones(len(X_known_anomalies))])

        # 切分驗證集 (為了調參使用)
        X_train, X_test, y_train, y_test = train_test_split(X_train_final, y_train_final, test_size=0.2, stratify=y_train_final, random_state=42)
        return X_train, X_test, y_train, y_test
    
    @staticmethod
    def tuneParameters(train_turn,X_train, X_test, y_train, y_test):
        # 使用 RandomizedSearchCV 進行參數調整
        random_search = RandomizedSearchCV(
            estimator=catboost_model.model,
            param_distributions=catboost_model.param_dist,
            n_iter = train_turn,  # 隨機嘗試 10 組參數
            scoring='roc_auc',
           cv=3,       # 3-Fold Cross Validation
           verbose=1,
           n_jobs=-1   # 使用所有 CPU 核心
        )

        # 開始訓練與搜索
        random_search.fit(X_train, y_train, eval_set=(X_test, y_test))

        # 取得最佳模型
        best_model = random_search.best_estimator_
        print(f"\n最佳參數組合: {random_search.best_params_}")
        print(f"最佳 CV AUC 分數: {random_search.best_score_:.4f}")
        return best_model
    
    def train(best_model,X_train, y_train):
        # 使用最佳參數訓練最終模型
        best_model.fit(X_train, y_train)
        return best_model