import pandas as pd
import numpy as np
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier

trainDataFile = 'DataSet\\acct_transaction.csv'
alertDataFile = 'DataSet\\acct_alert.csv'
outputSourceFile = 'DataSet\\acct_predict.csv'
outputPath = 'DataSet\\'

def load_data():
    """讀取 CSV 檔案並合併標籤"""
    try:
        df = pd.read_csv(trainDataFile)
        alert_df = pd.read_csv(alertDataFile)
        print("主資料維度:", df.shape)
        print("警示標籤維度:", alert_df.shape)
        # 根據 acct 是否在警示名單中，標記 alert_label
        alert_set = set(alert_df['acct'])
        df['alert_label'] = df['to_acct'].apply(lambda x: 1 if x in alert_set else 0)
        print("合併後資料維度:", df.shape)
        print("欄位列表:", df.columns.tolist())
    except FileNotFoundError:
        print("錯誤：找不到指定的資料檔案。請確認檔案路徑和名稱是否正確。")
        exit()
    return df

def preprocess_data(df):
    """資料前處理"""
    # 假設起始日期為 2020-01-01
    base_date = pd.to_datetime("2020-01-01")
    df['txn_date'] = df['txn_date'].astype(int)
    df['txn_time'] = df['txn_time'].astype(str)
    df['txn_datetime'] = df['txn_date'].apply(lambda x: base_date + pd.Timedelta(days=x)).astype(str) + ' ' + df['txn_time']
    df['txn_datetime'] = pd.to_datetime(df['txn_datetime'])
    df['txn_hour'] = df['txn_datetime'].dt.hour
    df['txn_day_of_week'] = df['txn_datetime'].dt.dayofweek
    df['is_night_txn'] = ((df['txn_hour'] < 6) | (df['txn_hour'] > 22)).astype(int)

    # 只對低基數類別做獨熱編碼
    categorical_features = ['from_acct_type', 'is_self_txn', 'currency_type', 'channel_type']
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    # 移除高基數與原始欄位
    df = df.drop(['from_acct', 'to_acct', 'txn_date', 'txn_time', 'txn_datetime'], axis=1)
    print("處理後資料維度:", df.shape)
    return df

def create_features(df):
    """創造新特徵"""
    # 範例1：資產與交易的比例
    # 避免除以零的錯誤
    #df['txn_amt_to_asset_ratio'] = df['total_txn_amt_L3M'] / (df['total_asset'] + 1)

    # 範例2：交易頻率
    #df['avg_txn_per_day_L3M'] = df['total_txn_count_L3M'] / 90

    # 範例3：夜間交易比例
    # 假設有 'night_txn_count_L3M' 欄位
    #df['night_txn_ratio'] = df['night_txn_count_L3M'] / (df['total_txn_count_L3M'] + 1)

    print("特徵工程完成！")
    return df

def split_features_labels(df, label_column='alert_label'):
    """分割特徵和標籤"""
    X = df.drop(columns=[label_column], axis=1)
    y = df[label_column]
    return X, y

if __name__ == "__main__":
    print("XGBoost version:", xgboost.__version__)
    print(xgboost.build_info())

    df = load_data() # 讀取資料
    # print(df.head(60)) # 顯示前50筆資料
    #print(df[df['alert_label'] == 1].head(10))#get first 10 data which alert_label is 1
    df_processed = preprocess_data(df) # 資料前處理
    df_processed = create_features(df_processed) # 特徵工程
    X, y = split_features_labels(df_processed) # 定義特徵 X 和目標 y

    # 切分訓練集和測試集 (80% 訓練, 20% 測試)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"訓練集大小: {X_train.shape}")
    print(f"測試集大小: {X_test.shape}")
    print(f"訓練集中警示帳戶比例: {y_train.mean():.2%}")
    print(f"測試集中警示帳戶比例: {y_test.mean():.2%}")
    
    # 計算 scale_pos_weight 以處理類別不平衡
    # 它的值是 (負類別數量 / 正類別數量)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Scale Pos Weight: {scale_pos_weight:.2f}")

    # 初始化 XGBoost 分類器
    # objective='binary:logistic': 用於二元分類
    # eval_metric='logloss': 評估指標
    # use_label_encoder=False: 避免未來版本中的警告
    xgb_clf = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight, # 處理不平衡資料的關鍵參數
        n_estimators=200,                 # 樹的數量
        max_depth=5,                      # 樹的最大深度
        learning_rate=0.1,                # 學習率
        subsample=0.8,                    # 訓練每棵樹時使用的樣本比例
        colsample_bytree=0.8,             # 訓練每棵樹時使用的特徵比例
        random_state=42,
        device='gpu',
        tree_method='gpu_hist'
    )

    # 訓練模型
    print("開始訓練 XGBoost 模型...")
    xgb_clf.fit(X_train, y_train)
    print("模型訓練完成！")
    
    # save model
    xgb_clf.save_model(outputPath + 'xgb_model.json')
    print(f"模型已儲存至 {outputPath + 'xgb_model.json'}")
    
    # 在測試集上進行預測
    y_pred = xgb_clf.predict(X_test)
    y_pred_proba = xgb_clf.predict_proba(X_test)[:, 1] # 預測為警示帳戶的機率

    # 1. 混淆矩陣 (Confusion Matrix)
    cm = confusion_matrix(y_test, y_pred)
    print("混淆矩陣:\n", cm)

    # 2. 分類報告 (Classification Report)
    print("\n分類報告:\n", classification_report(y_test, y_pred))

    # 3. ROC-AUC 分數
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC-AUC 分數: {auc:.4f}")

    # read outputSourceFile's acct,mapping to df and feed them into model and add result to it
    # output_df = pd.read_csv(outputSourceFile)
    # output_df = output_df.merge(df[['to_acct', 'alert_label']], on='acct', how='left')
    # output_df['alert_prediction'] = xgb_clf.predict(output_df)
    # output_df.to_csv(outputPath + 'acct_predict_with_alert.csv', index=False)
    # print(f"預測結果已儲存至 {outputPath + 'acct_predict_with_alert.csv'}")
