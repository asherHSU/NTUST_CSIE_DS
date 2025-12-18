import numpy as np

from src.AE import AE
from src.CatBoost import catboost_model
from src.data_processing import DataProcessor


# 結合 AE 與 CatBoost 的混合模型 (用於異常偵測)
class AE_CatBoost_Model:
    """
    整合 AutoEncoder 與 CatBoost 的混合模型
    - AutoEncoder: 用於特徵壓縮與異常偵測
    - CatBoost: 用於監督學習的分類
    """
    
    def __init__(self,threshold=0.5):
        """
        初始化模型
        - ae_model: AutoEncoder 模型物件
        - ae_encoder: 用於提取壓縮特徵的編碼器
        - cat_model: CatBoost 分類器
        - input_dim: 輸入特徵維度
        """
        self.ae_model = AE()                   # AE 模型 (訓練後會被設定) 
        self.cat_model = catboost_model.model  # CatBoost 分類器
        self.input_dim = None                   # 輸入特徵維度
        self.uncertain_samples = None          # 儲存 AE 篩選出的不確定樣本
        self.threshold = threshold 

    def fit(self, X, y, tune_params=False, train_turn=10):
        """
        訓練完整的混合模型
        
        輸入參數：  *可套用進(X_train, y_train)的模式*
        - X: 輸入特徵
        - y: 標籤
        - tune_params: 是否進行超參數調整 (預設 False)
        - train_turn: 超參數調整的迭代次數 (預設 10)
        
        訓練參數：
        - X_unlabeled_scaled: 標準化的無標籤資料 (用於訓練 AE)
        - X_unlabeled: 未標準化的無標籤資料 (用於篩選可靠正常樣本)
        - X_known_anomalies: 已知異常樣本特徵

        流程：
        1. 訓練 AutoEncoder 並篩選可靠的正常樣本
        2. 建立混合資料集 (原始 + 壓縮特徵)
        3. 訓練 CatBoost 分類器
        """
        # 建立訓練參數
        X_unlabeled = X
        # X_known_anomalies = DataProcessor.split_diff_label(X, y, positive_label=True).values
        X_known_anomalies = DataProcessor.split_diff_label(X, y, positive_label=True)
        X_unlabeled_scaled = DataProcessor.scaler(X_unlabeled)

        # 步驟 1: 訓練 AutoEncoder 並篩選可靠正常樣本
        print("=" * 50)
        print("步驟 1: 訓練 AutoEncoder 並篩選樣本...")
        print("=" * 50)
        
        self.input_dim = X_unlabeled_scaled.shape[1]
        
        # 訓練 AE 並取得可靠正常樣本
        X_reliable_negatives = self.ae_model.train(X_unlabeled_scaled, X_unlabeled)
        
        # 步驟 2: 建立混合特徵 (原始特徵 + 壓縮特徵)
        print("\n" + "=" * 50)
        print("步驟 2: 建立混合資料集並訓練 CatBoost...")
        print("=" * 50)
        
        # 切分訓練集與驗證集
        X_train, X_test, y_train, y_test = catboost_model.catBoost_cut(
            X_reliable_negatives, 
            X_known_anomalies
        )
        
        # 步驟 4: 訓練 CatBoost
        if tune_params:
            # 進行超參數調整
            print("\n執行超參數調整 (RandomizedSearchCV)...")
            best_model = catboost_model.tuneParameters(
                train_turn, 
                X_train, X_test, y_train, y_test
            )
            self.cat_model = best_model
        
        # 使用最終資料訓練 CatBoost
        self.cat_model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=0)
        print("\n" + "=" * 50)
        print(f"CatBoost 訓練完成")
        print("=" * 50)

    def predict(self, X):
        """
        預測樣本的標籤
        
        參數：
        - X: 輸入特徵 *可套用進(X_test)的模式*
        
        回傳：
        - 預測標籤 (0: 正常, 1: 異常)
        """
         # 1. 先取得預測機率 (Shape: [n_samples, 2])
        # 注意：我們只需要 "異常 (Class 1)" 的機率
        probs = self.cat_model.predict_proba(X)[:, 1]
        
        # 2. 根據設定的閾值進行切分
        # 大於等於閾值回傳 1，否則回傳 0
        predictions = (probs >= self.threshold).astype(int)
        
        return predictions

    def predict_proba(self, X):
        """
        預測樣本的概率
        
        參數：
        - X: 輸入特徵
        
        回傳：
        - 各類別的預測概率
        """
        return self.cat_model.predict_proba(X)
    
    def set_threshold(self, new_threshold):
        """
        ★ 新增功能：動態調整閾值
        讓您在不重新訓練的情況下改變靈敏度
        """
        print(f"模型判定閾值已從 {self.threshold} 修改為 {new_threshold}")
        self.threshold = new_threshold
    
    