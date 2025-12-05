from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def prepare_data(
    df: pd.DataFrame,
    random_state: int = 42,
    target_pos_ratio: float = 0.50,
    test_size: float = 0.30
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    將原始資料清理、切分、標準化並進行 SMOTE 過採樣後，回傳 (X_train, X_test, y_train, y_test)。

    Args:
        df (pd.DataFrame): 原始資料，需包含 'label' 欄位。
        random_state (int): 隨機種子。
        target_pos_ratio (float): 希望訓練集過採樣後的正類比例，例如 0.50。
        test_size (float): 測試集比例。

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            (X_train_resampled, X_test_scaled, y_train_resampled, y_test)
    """
    if 'label' not in df.columns:
        raise ValueError("輸入的 DataFrame 必須包含 'label' 欄位。")

    # 僅保留數值欄位並移除全為 NaN 欄位
    df_numeric = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')

    # 拆分特徵與標籤
    y = df_numeric['label'].copy()
    X = df_numeric.drop(columns=['label']).copy()

    # 先切分資料（避免資料洩漏）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # 標準化：用訓練集擬合，再套用到訓練/測試
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 計算 SMOTE 的 sampling_strategy
    # sampling_strategy = 目標正類比例 / (1 - 目標正類比例)
    if not (0 < target_pos_ratio < 1):
        raise ValueError("target_pos_ratio 必須介於 (0, 1) 之間。")
    sampling_strategy = target_pos_ratio / (1 - target_pos_ratio)

    # 訓練集 SMOTE
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    # 訊息輸出
    print("SMOTE 後訓練集大小:", X_train_resampled.shape, "正類比例:", y_train_resampled.mean())
    print(f"訓練集大小: {X_train_resampled.shape}")
    print(f"測試集大小: {X_test_scaled.shape}")
    print(f"訓練集中警示帳戶比例: {y_train_resampled.mean():.2%}")
    print(f"測試集中警示帳戶比例: {y_test.mean():.2%}")

    return X_train_resampled, X_test_scaled, y_train_resampled, y_test