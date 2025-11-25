import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def LoadCSV(dir_path):
    
    try:
        df_txn = pd.read_csv('D:\\讀書==\\NTUST\\大三上\\資料科學\\NTUST_CSIE_DS\\DataSet\\acct_transaction.csv')
        df_alert = pd.read_csv('D:\\讀書==\\NTUST\\大三上\\資料科學\\NTUST_CSIE_DS\\DataSet\\acct_alert.csv')
        print("交易資料維度:", df_txn.shape)
        print("警示標籤維度:", df_alert.shape)
        
        
    except FileNotFoundError:
        print("錯誤：找不到指定的資料檔案。請確認檔案路徑和名稱是否正確。")
        exit()
    
    print("(Finish) Load Dataset.")
    return df_txn, df_alert

def PreProcessing(df_txn):
    """
    資料前處理
    """    
    print("(Start) PreProcessing...")
    # 將時間拆分並轉換為秒數
    time_parts = df_txn['txn_time'].str.split(':')
    hour = time_parts.str[0].astype(int)
    minute = time_parts.str[1].astype(int)
    second = time_parts.str[2].astype(int)
    
    # 將時間轉換為當日的總秒數
    time_in_seconds = hour * 3600 + minute * 60 + second
    
    # 創建時間戳：(日期-1) * 86400秒 + 當日秒數
    df_txn['timestamp'] = (df_txn['txn_date'] - 1) * 86400 + time_in_seconds
    df_txn.drop(columns=['txn_date', 'txn_time'], inplace=True)

    # 幣值對照
    # AUD *= 20
    # CAD *= 22
    # CHF *= 39
    # CNY *= 4.4
    # EUR *= 36
    # GBP *= 41
    # HKD *= 4 
    # JPY *= 0.2
    # MXN *= 1.7
    # NZD *= 17
    # SEK *= 3
    # SGD *= 24
    # THB *= 1
    # TWD *= 1
    # USD *= 30
    # ZAR *= 1.8
    
    currency_map = {
        'AUD': 20,
        'CAD': 22,
        'CHF': 39,
        'CNY': 4.4,
        'EUR': 36,
        'GBP': 41,
        'HKD': 4,
        'JPY': 0.2,
        'MXN': 1.7,
        'NZD': 17,
        'SEK': 3,
        'SGD': 24,
        'THB': 1,
        'TWD': 1,
        'USD': 30,
        'ZAR': 1.8
    }
    
    # 交易次數
    send_times = df_txn.groupby('from_acct')['timestamp'].apply(list)
    recv_times = df_txn.groupby('to_acct')['timestamp'].apply(list)
    all_times = pd.concat([send_times, recv_times]).groupby(level=0).apply(lambda x: sum(x.tolist(), []))

    # 交易次數 (Basic實作)
    txn_count = all_times.apply(lambda lst: len(lst)).rename('txn_count')
    
    # 每個帳號的timestamp給他寫出來
    # 第一筆
    first_txn_ts = all_times.apply(lambda lst: min(lst) if len(lst) > 0 else 0).rename('first_txn_ts')
    # 最後一筆
    last_txn_ts = all_times.apply(lambda lst: max(lst) if len(lst) > 0 else 0).rename('last_txn_ts')
    # 多筆交易時間的標準差
    std_txn_ts = all_times.apply(lambda lst: np.std(lst, ddof=0) if len(lst) > 1 else 0).rename('std_txn_ts')

    # quick check (first few accounts)
    print(all_times.head())

    df_result = pd.concat([
        txn_count, first_txn_ts, last_txn_ts, std_txn_ts
    ], axis=1).fillna(0).reset_index()
    df_result.rename(columns={'index': 'acct'}, inplace=True)
    
    # 2. 'is_esun': is esun account or not
    df_from = df_txn[['from_acct', 'from_acct_type']].rename(columns={'from_acct': 'acct', 'from_acct_type': 'is_esun'})
    df_to = df_txn[['to_acct', 'to_acct_type']].rename(columns={'to_acct': 'acct', 'to_acct_type': 'is_esun'})
    df_acc = pd.concat([df_from, df_to], ignore_index=True).drop_duplicates().reset_index(drop=True)

    # 是否跨行轉帳
    send_types = df_txn.groupby('from_acct')['from_acct_type'].apply(lambda s: set(s.dropna().astype(int))).rename('from_types')
    recv_types = df_txn.groupby('to_acct')['to_acct_type'].apply(lambda s: set(s.dropna().astype(int))).rename('to_types')
    types = pd.concat([send_types, recv_types], axis=1)
    
    types['from_types'] = types['from_types'].apply(lambda x: x if isinstance(x, set) else set())
    types['to_types'] = types['to_types'].apply(lambda x: x if isinstance(x, set) else set())
    # 0代表沒有 1代表有
    types['cross_type'] = types.apply(lambda r: 0 if len((r['from_types'] | r['to_types'])) <= 1 else 1, axis=1)
    cross_type_series = types['cross_type']
    
    # 3. 把帳號ID跟功能對照起來
    df_result = pd.merge(df_result, df_acc, on='acct', how='left')    
    # merge cross_type by acct index (accounts missing cross_type -> assume 0)
    df_result = pd.merge(df_result, cross_type_series.rename('cross_type'), left_on='acct', right_index=True, how='left')
    df_result['cross_type'] = df_result['cross_type'].fillna(0).astype(int)
    
    # 幣值轉換 unified_amount
    multipliers = df_txn['currency_type'].map(currency_map).fillna(1)
    df_txn['uni_amt'] = df_txn['txn_amt'] * multipliers

    # 4. merge (1), (2), and (3)
    df_result = pd.merge(df_result, df_acc, on='acct', how='left')    
    print("(Finish) PreProcessing.")
    return df_result

if __name__ == "__main__":
    dir_path = "DataSet\\"
    df_txn, df_alert = LoadCSV(dir_path)
    df_X = PreProcessing(df_txn)
    df_X.to_csv('D:\\讀書==\\NTUST\\大三上\\資料科學\\NTUST_CSIE_DS\\XGBoost\\result.csv', index=False)
    print("(Finish) Save preprocessed data to 'XGBoost\\result.csv'.")