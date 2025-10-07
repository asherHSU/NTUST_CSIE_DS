import os
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
    # 1. 'total_send/recv_amt': total amount sent/received by each acct
    send = df_txn.groupby('from_acct')['txn_amt'].sum().rename('total_send_amt')
    recv = df_txn.groupby('to_acct')['txn_amt'].sum().rename('total_recv_amt')

    # 2. max, min, avg txn_amt for each account
    max_send = df_txn.groupby('from_acct')['txn_amt'].max().rename('max_send_amt')
    min_send = df_txn.groupby('from_acct')['txn_amt'].min().rename('min_send_amt')
    avg_send = df_txn.groupby('from_acct')['txn_amt'].mean().rename('avg_send_amt')

    max_recv = df_txn.groupby('to_acct')['txn_amt'].max().rename('max_recv_amt')
    min_recv = df_txn.groupby('to_acct')['txn_amt'].min().rename('min_recv_amt')
    avg_recv = df_txn.groupby('to_acct')['txn_amt'].mean().rename('avg_recv_amt')

    df_result = pd.concat([max_send, min_send, avg_send, max_recv, min_recv, avg_recv, send, recv], axis=1).fillna(0).reset_index()
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