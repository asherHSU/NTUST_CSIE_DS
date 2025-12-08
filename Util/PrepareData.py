from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def prepare_data_pure(
    df: pd.DataFrame,
    random_state: int = 42,
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
    
    print(f"訓練集大小: {X_train.shape}")
    print(f"測試集大小: {X_test.shape}")
    print(f"訓練集中警示帳戶比例: {y_train.mean():.2%}")
    print(f"測試集中警示帳戶比例: {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test

def prepare_data_cutting(
    df: pd.DataFrame,
    neg_ratio: float = 0.50,
    pos_scale: float = 1.0,
    random_state: int = 42,
    test_size: float = 0.30
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    保留全部正類（label=1），依比例抽樣負類（label=0），切分與標準化。
    若 pos_scale > 1.0，使用 SMOTE 於訓練集將正類數量放大為原本的 pos_scale 倍。
    """
    if 'label' not in df.columns:
        raise ValueError("輸入的 DataFrame 必須包含 'label' 欄位。")
    if not (0 < neg_ratio <= 1):
        raise ValueError("neg_ratio 必須介於 (0, 1] 之間。")
    if pos_scale <= 0:
        raise ValueError("pos_scale 必須 > 0。")

    df_numeric = df.select_dtypes(include=[np.number]).dropna(axis=1, how='all')

    pos_df = df_numeric[df_numeric['label'] == 1]
    neg_df = df_numeric[df_numeric['label'] == 0]

    neg_sample = neg_df.sample(frac=neg_ratio, random_state=random_state, replace=False)
    sampled_df = pd.concat([pos_df, neg_sample], axis=0).sample(frac=1.0, random_state=random_state)

    y = sampled_df['label'].copy()
    X = sampled_df.drop(columns=['label']).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 當 pos_scale > 1.0 時，於訓練集使用 SMOTE 放大正類數量（僅增正類，負類不變）
    if pos_scale > 1.0:
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        if n_pos == 0:
            raise ValueError("訓練集沒有正類，無法執行 SMOTE。")
        # 目標正類數量
        n_pos_target = int(n_pos * pos_scale)
        
        # 只提高正類到 n_pos_target，負類維持 n_neg（不增不減）
        sampling_strategy = {0: n_neg, 1: n_pos_target}

        # 若 k_neighbors 大於正類樣本數-1，需調小以避免錯誤
        k_neighbors = max(1, min(5, n_pos - 1))

        smote = SMOTE(sampling_strategy=sampling_strategy,
                      random_state=random_state,
                      k_neighbors=k_neighbors)
        X_train_out, y_train_out = smote.fit_resample(X_train_scaled, y_train)
    else:
        X_train_out, y_train_out = X_train_scaled, y_train

    print(f"訓練集大小: {X_train_out.shape}, 測試集大小: {X_test_scaled.shape}")
    print(f"訓練集正類比例: {y_train_out.mean():.2%}, 測試集正類比例: {y_test.mean():.2%}")

    return X_train_out, X_test_scaled, y_train_out, y_test

def prepare_data_smote(
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

def performPCA(data, n_components=2):
    from sklearn.decomposition import PCA
    df_numeric = data.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
    pca_model = PCA(n_components=n_components)
    reduced_data = pca_model.fit_transform(df_numeric.drop(columns=['label'], errors='ignore'))
    result_df = pd.DataFrame(reduced_data, columns=[f'PC{i+1}' for i in range(n_components)])
    if 'label' in df_numeric.columns:
        result_df = pd.concat([result_df, df_numeric['label'].reset_index(drop=True)], axis=1)
    return result_df