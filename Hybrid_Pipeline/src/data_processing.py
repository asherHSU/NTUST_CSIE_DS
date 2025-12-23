import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self):
        pass

    @staticmethod
    # === 取得特定標籤的資料 ===
    def split_diff_label(X, y, positive_label=False):
        target = 1 if positive_label else 0
    
        # 確保 y 的索引跟 X 對齊 (選用，避免因索引錯亂導致的空交集)
        if hasattr(X, 'index') and not X.index.equals(y.index):
            y = y.values 
        
        # 直接使用布林遮罩篩選，不要用 for 迴圈
        mask = (y == target)
        return X[mask]
    
    @staticmethod
    # === 標準化 MinMaxScaler ===
    def scaler(data):
        scaler = MinMaxScaler()
        temp = scaler.fit_transform(data)
        return temp
    
    @staticmethod
    # === 去掉 label 並回傳 label ===
    # 輸入 pd 去掉 label 
    # return pd label , pd on label (no change input data)
    def cut_label(data):
        temp = data.copy()
        temp = temp.drop(columns=['label'])
        return data['label'],temp
    
    @staticmethod
    # === 切分訓練與測試資料集 ===
    # x -> 無標籤資料集
    # y -> x 的標籤
    # test -> 結果產出用的 ; X_test -> 丟AE -> 拿"高風險"當結果的 X_test
    # X_train -> 丟進 AE -> 跑訓練
    # y_train -> 提取出 label 為 1 
    def train_test_seed_split(df_preprocessing,rand_seed):
        """
        切分訓練與測試資料集
        - df_preprocessing: 預處理後的完整資料集
        - rand_seed: 隨機種子 (用於 reproducibility)
        回傳：
        - train_data: 訓練資料集特徵
        - test_data: 測試資料集特徵
        - train_label: 訓練資料集標籤
        - test_label: 測試資料集標籤
        """ 
        y,X = DataProcessor.cut_label(df_preprocessing)
        train_data, test_data, train_label, test_label = train_test_split(X, y, test_size=0.3, random_state=rand_seed, stratify=y)
        return train_data, test_data, train_label, test_label