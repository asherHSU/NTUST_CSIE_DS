from catboost import CatBoostClassifier   
import numpy as np
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.metrics import make_scorer, fbeta_score

class catboost_model:
    def __init__(self):
        pass

    # 定義 CatBoost 模型
    # auto_class_weights='Balanced' 對於異常偵測極為重要，自動處理樣本不平衡
    model = CatBoostClassifier(
    iterations=1000,
    scale_pos_weight=141,  # 推薦的起始點
    eval_metric='AUC',     # 這種比例下，Accuracy 完全無效，請看 AUC
    early_stopping_rounds=50
    )

    # 定義參數網格 (Parameter Grid)
    param_dist = {
        # 原本是 [50, 100, 141, 200...], 現在改成更保守的範圍
        # 甚至包含 1 (不加權)，看是否光靠特徵就能分出來
        'scale_pos_weight': [1,2,3,4,5, 10, 20, 30, 42, 60, 100, 200, 300], 

        'depth': [4, 5, 6, 7, 8, 9, 10, 11, 12], 
        'l2_leaf_reg': [5, 10, 15, 20, 30, 40 , 50], # 加強正則化，減少誤判
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'random_strength': [5, 10, 20]     # 增加隨機擾動，避免過度擬合雜訊
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

        # # Define custom scorer for F-beta score with beta=0.5
        f05_scorer = make_scorer(fbeta_score, beta=0.5)
        # 使用 RandomizedSearchCV 進行參數調整
        random_search = RandomizedSearchCV(
            estimator=catboost_model.model,
            param_distributions=catboost_model.param_dist,
            n_iter = train_turn,  # 隨機嘗試 10 組參數
            scoring = f05_scorer, #'precision'->可能會太嚴格(test)、f05_scorer、f1
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
        print(best_model.get_params())
        return best_model
    
    def train(best_model,X_train, y_train):
        # 使用最佳參數訓練最終模型
        best_model.fit(X_train, y_train)
        return best_model