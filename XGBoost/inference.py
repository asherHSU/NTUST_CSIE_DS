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
outputPath = 'XGBoost\\'

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
    categorical_features = ['from_acct_type','to_acct_type', 'is_self_txn', 'txn_amt', 'currency_type', 'channel_type']
    df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    # 移除高基數與原始欄位
    df = df.drop(['from_acct', 'to_acct', 'txn_time', 'txn_datetime'], axis=1)
    print("處理後資料維度:", df.shape)
    return df

if __name__ == "__main__":
    print("XGBoost version:", xgboost.__version__)
    print(xgboost.build_info())

    df = load_data() # 讀取資料
    print(df.head(50)) # 顯示前50筆資料
    
    output_df = pd.read_csv(outputSourceFile)
    for index, row in output_df.iterrows():
        temp_df = df[(df['to_acct'] == row['acct']) | (df['from_acct'] == row['acct'])]
        if not temp_df.empty:
            print(f"to_acct: {row['acct']}")
            print(temp_df)
            print("\n")
    
            #predict temp_df
            temp_df_processed = preprocess_data(temp_df) # 資料前處理
            
    #save to csv
    output_df.to_csv(outputPath + 'output.csv', index=False)
    print(f"預測結果已儲存至 {outputPath + 'output.csv'}")