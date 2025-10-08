import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def LoadCSV(dir_path):
    """
    讀取提供的3個資料集：交易資料、警示帳戶註記、待預測帳戶清單
    Args:
        dir_path (str): 資料夾，請把上述3個檔案放在同一個資料夾
    
    Returns:
        df_txn: 交易資料 DataFrame
        df_alert: 警示帳戶註記 DataFrame
    """
    
    try:
        df_txn = pd.read_csv(os.path.join(dir_path, 'acct_transaction.csv'))
        df_alert = pd.read_csv(os.path.join(dir_path, 'acct_alert.csv'))
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

    # 1. 'total_send/recv_amt': total amount sent/received by each acct
    send = df_txn.groupby('from_acct')['txn_amt'].sum().rename('total_send_amt')
    recv = df_txn.groupby('to_acct')['txn_amt'].sum().rename('total_recv_amt')

    # 2. max, min, avg txn_amt for each account
    max_send = df_txn.groupby('from_acct')['txn_amt'].max().rename('max_send_amt')
    min_send = df_txn.groupby('from_acct')['txn_amt'].min().rename('min_send_amt')
    avg_send = df_txn.groupby('from_acct')['txn_amt'].mean().rename('avg_send_amt')
    var_send = df_txn.groupby('from_acct')['txn_amt'].var().rename('var_send_amt')
    std_send = df_txn.groupby('from_acct')['txn_amt'].std().rename('std_send_amt')

    max_recv = df_txn.groupby('to_acct')['txn_amt'].max().rename('max_recv_amt')
    min_recv = df_txn.groupby('to_acct')['txn_amt'].min().rename('min_recv_amt')
    avg_recv = df_txn.groupby('to_acct')['txn_amt'].mean().rename('avg_recv_amt')
    var_recv = df_txn.groupby('to_acct')['txn_amt'].var().rename('var_recv_amt')
    std_recv = df_txn.groupby('to_acct')['txn_amt'].std().rename('std_recv_amt')
    
    # high frequency send by timestamp
    #collect acct timestamp
    send_times = df_txn.groupby('from_acct')['timestamp'].apply(list)
    recv_times = df_txn.groupby('to_acct')['timestamp'].apply(list)
    #combine send and recv times list
    all_times = pd.concat([send_times, recv_times], axis=0).sort_index()
    print(all_times.head())

    df_result = pd.concat([max_send, min_send, avg_send, var_send, std_send, max_recv, min_recv, avg_recv, var_recv, std_recv, send, recv], axis=1).fillna(0).reset_index()
    df_result.rename(columns={'index': 'acct'}, inplace=True)
    
    # 2. 'is_esun': is esun account or not
    df_from = df_txn[['from_acct', 'from_acct_type']].rename(columns={'from_acct': 'acct', 'from_acct_type': 'is_esun'})
    df_to = df_txn[['to_acct', 'to_acct_type']].rename(columns={'to_acct': 'acct', 'to_acct_type': 'is_esun'})
    df_acc = pd.concat([df_from, df_to], ignore_index=True).drop_duplicates().reset_index(drop=True)
    
    # 4. merge (1), (2), and (3)
    df_result = pd.merge(df_result, df_acc, on='acct', how='left')    
    print("(Finish) PreProcessing.")
    return df_result

if __name__ == "__main__":
    dir_path = "DataSet\\"
    df_txn, df_alert = LoadCSV(dir_path)
    df_X = PreProcessing(df_txn)
    df_X.to_csv('XGBoost\\preprocessed_data.csv', index=False)
    print("(Finish) Save preprocessed data to 'XGBoost\\preprocessed_data.csv'.")