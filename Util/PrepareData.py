import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from typing import Tuple

class DataPreparer:
    def __init__(self, random_state: int):
        self.random_state = random_state
        self.scaler = StandardScaler()

    def prepare_cutting(self,df: pd.DataFrame,neg_ratio: float,pos_scale: float,test_size: float)-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        if "label" not in df.columns:
            raise ValueError("DataFrame must contain label")

        df = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')

        pos_df = df[df["label"] == 1]
        neg_df = df[df["label"] == 0].sample(frac=neg_ratio,random_state=self.random_state)

        sampled_df = pd.concat([pos_df, neg_df]).sample(frac=1.0,random_state=self.random_state)

        X = sampled_df.drop(columns=["label"])
        y = sampled_df["label"]

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,stratify=y,random_state=self.random_state)

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        if pos_scale > 1:
            n_pos = (y_train == 1).sum()
            n_neg = (y_train == 0).sum()
            smote = SMOTE(
            sampling_strategy={0: n_neg, 1: int(n_pos * pos_scale)},
            random_state=self.random_state
            )
            X_train, y_train = smote.fit_resample(X_train, y_train)

        print(f"Training data:\tPositive samples: {(y_train == 1).sum()},\tNegative samples: {(y_train == 0).sum()},Ratio: {(y_train == 1).mean():.2f}\n"
              f"Test data:\tPositive samples: {(y_test == 1).sum()},\tNegative samples: {(y_test == 0).sum()},\tRatio: {(y_test == 1).mean():.2f}")
        print("finish prepare_data_cutting\n")
        return X_train, X_test, y_train, y_test
    
    def prepare_data_pure(self,df: pd.DataFrame,test_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        僅將資料清理、切分並標準化，不進行 SMOTE。
        返回 (X_train, X_test, y_train, y_test) 為 numpy 陣列。
        """
        if "label" not in df.columns:
            raise ValueError("DataFrame must contain label")

        df_numeric = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')

        y = df_numeric["label"].copy()
        X = df_numeric.drop(columns=["label"]).copy()

        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size,stratify=y,random_state=self.random_state)

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training data:\tPositive samples: {(y_train == 1).sum()},\tNegative samples: {(y_train == 0).sum()},\tRatio: {(y_train == 1).mean():.2f}\n"
              f"Test data:\tPositive samples: {(y_test == 1).sum()},\tNegative samples: {(y_test == 0).sum()},\tRatio: {(y_test == 1).mean():.2f}")
        print("finish prepare_data_cutting\n")
        return X_train_scaled, X_test_scaled, y_train, y_test

    def prepare_data_smote(self,df: pd.DataFrame,target_pos_ratio: float,test_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        清理、切分、標準化後，於訓練集使用 SMOTE 使正類比例達到 target_pos_ratio。
        target_pos_ratio 介於 (0, 1)，例如 0.5 表示希望正類約佔一半。
        返回 (X_train, X_test, y_train, y_test) 為 numpy 陣列。
        """
        if "label" not in df.columns:
            raise ValueError("DataFrame must contain label")
        if not (0 < target_pos_ratio < 1):
            raise ValueError("target_pos_ratio 必須介於 (0, 1) 之間")

        df_numeric = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')

        y = df_numeric["label"].copy()
        X = df_numeric.drop(columns=["label"]).copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y,
            random_state=self.random_state
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 將目標正類比例轉為 SMOTE 的 minority/majority 比例
        sampling_ratio = target_pos_ratio / (1 - target_pos_ratio)

        smote = SMOTE(sampling_strategy=sampling_ratio, random_state=self.random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

        print(f"Training data:\tPositive samples: {(y_train_resampled == 1).sum()},\tNegative samples: {(y_train_resampled == 0).sum()},\tRatio: {(y_train_resampled == 1).mean():.2f}\n"
              f"Test data:\tPositive samples: {(y_test == 1).sum()},\tNegative samples: {(y_test == 0).sum()},\tRatio: {(y_test == 1).mean():.2f}")
        print("finish prepare_data_smote\n")
        return X_train_resampled, X_test_scaled, y_train_resampled, y_test
