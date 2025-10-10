import pandas as pd
import numpy as np
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
import os

dir_path = "DataSet\\"
outputPath = 'XGBoost\\'

def TrainTestSplit(df, df_alert, df_test):
    """
    切分訓練集及測試集，並為訓練集的帳戶標上警示label (0為非警示、1為警示)
    
    備註:
        1. 測試集為待預測帳戶清單，你需要預測它們
        2. 此切分僅為範例，較標準的做法是基於訓練集再且分成train和validation，請有興趣的參賽者自行切分
        3. 由於待預測帳戶清單僅為玉山戶，所以我們在此範例僅使用玉山帳戶做訓練
    """  
    X_train = df[(~df['acct'].isin(df_test['acct'])) & (df['is_esun']==1)].drop(columns=['is_esun']).copy()
    y_train = X_train['acct'].isin(df_alert['acct']).astype(int)
    X_test = df[df['acct'].isin(df_test['acct'])].drop(columns=['is_esun']).copy()
    
    print(f"(Finish) Train-Test-Split")
    return X_train, X_test, y_train

# def Modeling(X_train, y_train, X_test):
#     """
#     Decision Tree的範例程式，參賽者可以在這裡實作自己需要的方法
#     """
#     model = DecisionTreeClassifier(random_state=42)
#     model.fit(X_train.drop(columns=['acct']), y_train)
#     y_pred = model.predict(X_test.drop(columns=['acct']))   
    
#     print(f"(Finish) Modeling")
#     return y_pred

def OutputCSV(path, df_test, y_pred):
    """
    根據測試資料集及預測結果，產出預測結果之CSV，該CSV可直接上傳於TBrain    
    """
    df_pred = pd.DataFrame({
        'acct': df_test['acct'],
        'label': y_pred
    })
    
    df_out = df_test[['acct']].merge(df_pred, on='acct', how='left')
    df_out.to_csv(os.path.join(path, 'submission.csv'), index=False)
    
    print(f"(Finish) Output saved to {path}")
    
if __name__ == "__main__":
    print("XGBoost version:", xgboost.__version__)
    print(xgboost.build_info())

    df_processed = pd.read_csv('XGBoost\\preprocessed_data.csv')
    df_alert = pd.read_csv(os.path.join(dir_path, 'acct_alert.csv'))
    df_test = pd.read_csv(os.path.join(dir_path, 'acct_predict.csv'))
    print(df_processed.head(50)) # 顯示前50筆資料
    
    # 切分訓練集和測試集 (80% 訓練, 20% 測試)
    #X_train, X_test, y_train = TrainTestSplit(df_processed, df_alert, df_test)
    
    X = df_processed[(~df_processed['acct'].isin(df_test['acct'])) & (df_processed['is_esun']==1)]
    X = X.drop(columns=['is_esun']).copy()
    y = X['acct'].isin(df_alert['acct']).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    print(f"X_train:\n{X_train.head()}")
    print(f"訓練集大小: {X_train.shape}")
    print(f"測試集大小: {X_test.shape}")
    print(f"訓練集中警示帳戶比例: {y_train.mean():.2%}")
    print(f"測試集中警示帳戶比例: {y_test.mean():.2%}")
    
    # 計算 scale_pos_weight 以處理類別不平衡
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum() # (負類別數量 / 正類別數量)
    print(f"Scale Pos Weight: {scale_pos_weight:.2f}")

    # 初始化 XGBoost 分類器
    xgb_clf = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        scale_pos_weight=scale_pos_weight, # 處理不平衡資料的參數
        n_estimators=200,                 # 樹的數量
        max_depth=6,                      # 樹的最大深度
        learning_rate=0.1,                # 學習率
        subsample=0.8,                    # 訓練每棵樹時使用的樣本比例
        colsample_bytree=0.8,             # 訓練每棵樹時使用的特徵比例
        random_state=42,
        device='gpu',
        tree_method='gpu_hist'
    )

    # 訓練模型
    print("開始訓練 XGBoost 模型...")
    xgb_clf.fit(X_train.drop(columns=['acct']), y_train)
    print("模型訓練完成！")
    
    # save model
    xgb_clf.save_model(outputPath + 'xgb_model.json')
    print(f"模型已儲存至 {outputPath + 'xgb_model.json'}")
    
    # 在測試集上進行預測
    y_pred = xgb_clf.predict(X_test.drop(columns=['acct']))
    y_pred_proba = xgb_clf.predict_proba(X_test.drop(columns=['acct']))[:, 1] # 預測為警示帳戶的機率

    # 1. 混淆矩陣 (Confusion Matrix)
    cm = confusion_matrix(y_test, y_pred)
    print("混淆矩陣:\n", cm)

    # 2. 分類報告 (Classification Report)
    print("\n分類報告:\n", classification_report(y_test, y_pred))

    # 3. ROC-AUC 分數
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC-AUC 分數: {auc:.4f}")
    
    df_test = df_processed[df_processed['acct'].isin(df_test['acct'])].drop(columns=['is_esun']).copy()
    y_test_pred = xgb_clf.predict(df_test.drop(columns=['acct']))
    OutputCSV(outputPath, df_test, y_test_pred)   
    