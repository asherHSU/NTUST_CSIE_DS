from tensorflow.keras.models import Model           #AE
from tensorflow.keras.layers import Input, Dense    #AE
from tensorflow.keras.optimizers import Adam        #AE
import numpy as np


class AE:
    @staticmethod
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
    
    @staticmethod
    # X_unlabeled_scaled    標準化無標記綜合資料(含特徵)
    # X_unlabeled           無標記綜合資料(含特徵)
    def train(X_unlabeled_scaled,X_unlabeled):
        # 建立並訓練 AE
        # 注意：這裡使用全部未標記資料訓練，模型會傾向學習"大眾(正常)"的模式
        input_dim = X_unlabeled_scaled.shape[1]
        ae_model = AE.build_autoencoder(input_dim)
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
        X_uncertain = X_unlabeled[~mask_reliable] # 第一階段被剔除的高誤差群體
        print(f"篩選出 {len(X_reliable_negatives)} 筆可靠正常樣本，剔除 {len(X_unlabeled) - len(X_reliable_negatives)} 筆潛在雜訊。")
        
        # --- 修改處：加上重置索引 ---
        # 如果 X_unlabeled 是 DataFrame，這一步至關重要
        if hasattr(X_reliable_negatives, 'reset_index'):
            X_reliable_negatives = X_reliable_negatives.reset_index(drop=True)
            X_uncertain = X_uncertain.reset_index(drop=True)
            
        return X_reliable_negatives,X_uncertain