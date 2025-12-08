import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from catboost import CatBoostClassifier

# ==========================================
# 0. 資料準備 (模擬情境)
# ==========================================
# 假設 X_unlabeled 是所有交易紀錄 (包含混雜的正常與未知異常)
# 假設 X_known_anomalies 是已知的異常帳戶特徵
# ------------------------------------------

def prepare_demo_data():
    # 僅為產生範例資料，實際應用請替換為您的 DataFrame
    rng = np.random.RandomState(42)
    # 模擬 10,000 筆未標記資料 (假設大部分正常)
    X_unlabeled = rng.normal(loc=0, scale=1, size=(10000, 20)) 
    # 模擬 200 筆已知異常 (分佈不同)
    X_known_anomalies = rng.normal(loc=2, scale=1.5, size=(200, 20))
    return X_unlabeled, X_known_anomalies

X_unlabeled, X_known_anomalies = prepare_demo_data()

# 資料標準化 (Autoencoder 對數值尺度非常敏感，務必縮放)
scaler = MinMaxScaler()
X_unlabeled_scaled = scaler.fit_transform(X_unlabeled)
X_known_anomalies_scaled = scaler.transform(X_known_anomalies)

# ==========================================
# 1. 第一階段：Autoencoder (篩選可靠正常樣本)
# ==========================================

def build_autoencoder(input_dim):
    """建立簡單的 AE 模型"""
    input_layer = Input(shape=(input_dim,))
    
    # Encoder: 壓縮特徵
    encoded = Dense(16, activation='relu')(input_layer)
    encoded = Dense(8, activation='relu')(encoded)
    
    # Decoder: 還原特徵
    decoded = Dense(16, activation='relu')(encoded)
    output_layer = Dense(input_dim, activation='sigmoid')(decoded) # 若資料正規化到 0-1 用 sigmoid
    
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return autoencoder

print("--- Stage 1: Training Autoencoder for Noise Filtering ---")

# 建立並訓練 AE
# 注意：這裡使用全部未標記資料訓練，模型會傾向學習"大眾(正常)"的模式
input_dim = X_unlabeled_scaled.shape[1]
ae_model = build_autoencoder(input_dim)
ae_model.fit(
    X_unlabeled_scaled, X_unlabeled_scaled,
    epochs=20, 
    batch_size=256, 
    shuffle=True, 
    verbose=0
)

# 計算重建誤差 (Reconstruction Error / MSE)
reconstructions = ae_model.predict(X_unlabeled_scaled)
mse = np.mean(np.power(X_unlabeled_scaled - reconstructions, 2), axis=1)

# 設定閾值：選取誤差最小的前 80% 作為「可靠的正常樣本 (Reliable Negatives)」
# 剩下的 20% 被視為模糊地帶，暫時不參與訓練，避免誤導模型
threshold = np.percentile(mse, 80) 
mask_reliable = mse < threshold

X_reliable_negatives = X_unlabeled[mask_reliable] # 取得原始數值(非 scaled)
print(f"篩選出 {len(X_reliable_negatives)} 筆可靠正常樣本，剔除 {len(X_unlabeled) - len(X_reliable_negatives)} 筆潛在雜訊。")


# ==========================================
# 2. 第二階段：CatBoost (監督式分類與調參)
# ==========================================

print("\n--- Stage 2: Training CatBoost with Hyperparameter Tuning ---")

# 建構訓練集：可靠正常 (Label 0) + 已知異常 (Label 1)
X_train_final = np.vstack([X_reliable_negatives, X_known_anomalies])
y_train_final = np.hstack([np.zeros(len(X_reliable_negatives)), np.ones(len(X_known_anomalies))])

# 切分驗證集 (為了調參使用)
X_train, X_val, y_train, y_val = train_test_split(X_train_final, y_train_final, test_size=0.2, stratify=y_train_final, random_state=42)

# 定義 CatBoost 模型
# auto_class_weights='Balanced' 對於異常偵測極為重要，自動處理樣本不平衡
cb_model = CatBoostClassifier(
    loss_function='Logloss',
    eval_metric='AUC',
    auto_class_weights='Balanced', 
    verbose=0,
    early_stopping_rounds=50
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

# 使用 RandomizedSearchCV 進行參數調整
random_search = RandomizedSearchCV(
    estimator=cb_model,
    param_distributions=param_dist,
    n_iter=10,  # 隨機嘗試 10 組參數
    scoring='roc_auc',
    cv=3,       # 3-Fold Cross Validation
    verbose=1,
    n_jobs=-1   # 使用所有 CPU 核心
)

# 開始訓練與搜索
random_search.fit(X_train, y_train, eval_set=(X_val, y_val))

# 取得最佳模型
best_model = random_search.best_estimator_
print(f"\n最佳參數組合: {random_search.best_params_}")
print(f"最佳 CV AUC 分數: {random_search.best_score_:.4f}")

# ==========================================
# 3. 最終應用 (Inference)
# ==========================================
# 拿最好的模型去預測那些「當初被 AE 剔除的模糊資料」
# 或是新的未知資料

X_uncertain = X_unlabeled[~mask_reliable] # 第一階段被剔除的高誤差群體
probs = best_model.predict_proba(X_uncertain)[:, 1]

# 找出高風險帳戶 (例如預測機率 > 0.9)
high_risk_indices = np.where(probs > 0.9)[0]
print(f"\n在模糊地帶資料中，模型額外發現了 {len(high_risk_indices)} 個高風險異常。")