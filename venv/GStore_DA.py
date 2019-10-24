import os  #載入系統
import time  #時間套件
import json  #JSON套件
import numpy as np  #數學計算套件
import pandas as pd  #python版的excel
from pandas.io.json import json_normalize  #用python版的excel處理JSON
import matplotlib.pyplot as plt  #繪圖套件
from sklearn import preprocessing  #sklearn的前處理套件
from sklearn.metrics import mean_squared_error  #sklearn中評估指標: mse
from sklearn.model_selection import train_test_split  #sklearn模型選擇套件中的訓練測試切割器
from datetime import datetime  #日期時間套件
import lightgbm as lgb  #LightGBM套件
import xgboost as xgb  #XGBoost套件
from sklearn.tree import DecisionTreeRegressor  #決策樹套件
from sklearn.ensemble import RandomForestRegressor  #隨機森林套件

pd.options.mode.chained_assignment = None #鏈式賦值警告關閉
pd.options.display.max_columns = 999      #最大顯示欄位數量

#處理JSON欄位的方法: 分解巢狀欄位
def load_df(csv_path="D:/EJ wang/google_data/train_v3.csv", nrows=None):  #csv_path: data的絕對路徑, nrows: 載入的列樹
    JSON_COLUMNS = ["device", "geoNetwork", "totals", "trafficSource"]  #自行設定JSON欄位名稱
    df = pd.read_csv(csv_path,
                     converters={column: json.loads for column in JSON_COLUMNS},
                     dtype={"fullVisitorId": "str"},  #因為fullVisitorId格式的緣故，必須轉成字串格式讓每個id都是唯一的
                     nrows=nrows)
    for column in JSON_COLUMNS:  #分解JSON
        column_as_df = json_normalize(df[column])  #展開JSON欄位內的物件
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]  #強制規定格式, 用'.'來連接兩物件
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)  #展開完畢後刪除原本的JSON欄位
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")  #印出data的資訊, 確認載入狀態
    return df

def rmse(y_true, y_predict):  #計算rmse
    return round(np.sqrt(mean_squared_error(y_true, y_predict)), 3) #計算rmse取小數點後3位

def run_lgb(X_train, y_train, X_valid, y_valid, X_test, y_test):  #LghtGBM模型建立
    #X_train: 訓練集, y_train: 目標值(訓練集), X_valid: 驗證集, y_valid: 目標值(驗證集), X_test: 測試集, y_test: 目標值(測試集)
    params = {  #參數設定
        "objective" : "regression",     #模型目標: 回歸
        "metric" : "rmse",              #用RMSE評估模型
        "max_depth": 6,                 #最大深度
        "num_leaves" : 50,              #樹葉數量
        "learning_rate" : 0.005,        #學習率
        "bagging_fraction" : 0.75,      #採用樣本比例
        "feature_fraction" : 0.85,      #採用特徵比例
        "bagging_frequency" : 9,        #每9代的裝袋取樣頻率
        "verbosity" : -1
    }
    lgb_train = lgb.Dataset(X_train, label=y_train)  #訓練集設定
    lgb_valid = lgb.Dataset(X_valid, label=y_valid, reference=lgb_train)  #驗證集設定
    evals_result={}  #放模型訓練後資訊的空間

    # 模型建立
    model = lgb.train(params, lgb_train, 1000,  #放模型參數、訓練集、迭代數
                      valid_sets=[lgb_train, lgb_valid],  #觀察訓練集、驗證集的績效數值變化
                      # early_stopping_rounds=50,
                      verbose_eval=5, evals_result=evals_result)  #每5代顯示一筆、將訓練後資訊存到eval_result

    y_predict_train = model.predict(X_train, num_iteration=model.best_iteration)  #訓練的y 績效
    y_predict_valid = model.predict(X_valid, num_iteration=model.best_iteration)  #驗證的y 績效
    y_predict_test = model.predict(X_test, num_iteration=model.best_iteration)    #測試的y 績效

    #印出訓練、驗證、測試的結果
    print(f"LGBM:  RMSE train: {rmse(y_train, y_predict_train)}   RMSE val: {rmse(y_valid, y_predict_valid)}   RMSE test: {rmse(y_test, y_predict_test)}")
    return y_predict_test, model, evals_result  #回傳測試的y 績效、模型資訊、模型訓練後的資訊

def run_xgb(X_train, y_train, X_val, y_val, X_test, y_test):  #XGBoost模型建立
    #X_train: 訓練集, y_train: 目標值(訓練集), X_val: 驗證集, y_val: 目標值(驗證集), X_test: 測試集, y_test: 目標值(測試集)
    params = {'objective': 'reg:linear',    #模型目標: 回歸
              'eval_metric': 'rmse',        #用RMSE評估模型
              'eta': 0.3,                   #學習率
              'max_depth': 8,               #最大深度
              'subsample': 0.7,             #採用樣本比例
              'colsample_bytree': 0.7,      #採用特徵比例
              'alpha':0.001,                #正規化系數
              'silent': True}               #印出模型的執行資訊
    f_names = X_train.columns  #訓練集的欄位名稱
    xgb_train_data = xgb.DMatrix(np.asmatrix(X_train), y_train, feature_names=f_names)  #訓練集設定
    xgb_val_data = xgb.DMatrix(np.asmatrix(X_val), y_val, feature_names=f_names)  #驗證集設定
    xgb_test_data = xgb.DMatrix(np.asmatrix(X_test), feature_names=f_names)  #測試集設定
    evals_result = {}  #放模型訓練後資訊的空間

    #模型建立
    model = xgb.train(params, xgb_train_data,  #放模型參數、訓練集
                      num_boost_round=1000,  #迭代數
                      evals= [(xgb_train_data, 'train'), (xgb_val_data, 'valid')],  #觀察訓練集、驗證集的績效數值變化
                      # early_stopping_rounds=5,  #早停機制，k代內沒有改善時停止
                      verbose_eval=5, evals_result=evals_result,)  #每5代顯示一筆、將訓練後資訊存到eval_result
    y_predict_train = model.predict(xgb_train_data, ntree_limit=model.best_ntree_limit)  #訓練的y 績效
    y_predict_valid = model.predict(xgb_val_data, ntree_limit=model.best_ntree_limit)    #驗證的y 績效
    y_predict_test = model.predict(xgb_test_data, ntree_limit=model.best_ntree_limit)    #測試的y 績效

    # 印出訓練、驗證、測試的結果
    print(f"XGBoost:  RMSE train: {rmse(y_train, y_predict_train)}   RMSE val: {rmse(y_val, y_predict_valid)}   RMSE test: {rmse(y_test, y_predict_test)}")
    return y_predict_test, model, evals_result  #回傳測試的y 績效、模型資訊、模型訓練後的資訊

def run_tree(X_train, y_train, X_val, y_val, X_test, y_test, depth):  #決策樹模型建立
    #X_train: 訓練集, y_train: 目標值(訓練集), X_val: 驗證集, y_val: 目標值(驗證集), X_test: 測試集, y_test: 目標值(測試集), depth: 樹木深度
    my_tree=DecisionTreeRegressor(max_depth=depth, max_features='auto')  #建立決策樹模型(參數: 最大深度, 最大特徵數(自動))
    my_tree.fit(X_train, y_train)  #決策樹模型訓練
    y_predict_train = my_tree.predict(X_train)  #訓練的y 績效
    y_predict_valid = my_tree.predict(X_val)    #驗證的y 績效
    y_predict_test = my_tree.predict(X_test)    #測試的y 績效
    print(f"Decision tree: RMSE train: {rmse(y_train, y_predict_train)}  RMSE val: {rmse(y_val, y_predict_valid)}  RMSE test: {rmse(y_test, y_predict_test)}")
    return y_predict_test, my_tree

def run_forest(X_train, y_train, X_val, y_val, X_test, y_test, depth, n_tree):  #隨機森林模型建立
    #X_train: 訓練集, y_train: 目標值(訓練集), X_val: 驗證集, y_val: 目標值(驗證集), X_test: 測試集, y_test: 目標值(測試集)
    #depth: 樹木深度, n_tree: 決策樹數量
    my_forest=RandomForestRegressor(max_depth=depth, n_estimators=n_tree, max_features='auto') #建立隨機森林模型(參數: 最大深度, 決策樹數量, 最大特徵數(自動))
    my_forest.fit(X_train, y_train)  #隨機森林模型訓練
    y_predict_train = my_forest.predict(X_train)  #訓練的y 績效
    y_predict_valid = my_forest.predict(X_val)    #驗證的y 績效
    y_predict_test = my_forest.predict(X_test)    #測試的y 績效
    print(f"Random Forest: RMSE train: {rmse(y_train, y_predict_train)}  RMSE val: {rmse(y_val, y_predict_valid)}  RMSE test: {rmse(y_test, y_predict_test)}")
    return y_predict_test, my_forest

train_df = load_df()  #讀入訓練集
train_df["date"] = pd.to_datetime(train_df["date"], format="%Y%m%d")  #將訓練集的日期欄位轉成datetime格式
test_df = load_df("D:/EJ wang/google_data/test_v3.csv")  #讀入測試集
test_df["date"] = pd.to_datetime(test_df["date"], format="%Y%m%d")    #將測試集的日期欄位轉成datetime格式

const_cols = [c for c in train_df.columns if train_df[c].nunique(dropna=False)==1]  #找出欄位中只有一種值(無變化)的欄位
train_df = train_df.drop(const_cols + ["trafficSource.campaignCode", 'totals.totalTransactionRevenue'], axis=1)  #移除訓練集不必要之欄位
test_df = test_df.drop(const_cols + ['totals.totalTransactionRevenue'], axis=1)  #移除測試集不必要之欄位

# 遺漏值補0 (補值)
train_df["totals.transactionRevenue"].fillna(0, inplace=True)
train_df["totals.sessionQualityDim"].fillna(0, inplace=True)
train_df["totals.timeOnSite"].fillna(0, inplace=True)
train_df["totals.transactions"].fillna(0, inplace=True)
train_df["totals.bounces"].fillna(0, inplace=True)
train_df["totals.newVisits"].fillna(0, inplace=True)
train_df["totals.pageviews"].fillna(0, inplace=True)

test_df['totals.transactionRevenue'].fillna(0, inplace=True)
test_df['totals.transactions'].fillna(0, inplace=True)
test_df['totals.bounces'].fillna(0, inplace=True)
test_df['totals.timeOnSite'].fillna(0, inplace=True)
test_df['totals.newVisits'].fillna(0, inplace=True)
test_df['totals.pageviews'].fillna(0, inplace=True)

##用label encoding轉換"類別型"特徵
cat_cols = ["channelGrouping", "device.browser", "device.deviceCategory", "device.operatingSystem",
            "geoNetwork.city", "geoNetwork.continent", "geoNetwork.country", "geoNetwork.metro",
            "geoNetwork.networkDomain", "geoNetwork.region", "geoNetwork.subContinent", "trafficSource.adContent",
            "trafficSource.adwordsClickInfo.adNetworkType", "trafficSource.adwordsClickInfo.gclId",
            "trafficSource.adwordsClickInfo.page", "trafficSource.adwordsClickInfo.slot", "trafficSource.campaign",
            "trafficSource.keyword", "trafficSource.medium", "trafficSource.referralPath", "trafficSource.source",
            'trafficSource.adwordsClickInfo.isVideoAd', 'trafficSource.isTrueDirect']
for col in cat_cols:  #將類別型資料一起用lbl編碼器編碼
    lbl = preprocessing.LabelEncoder()  #建立label編碼器物件
    lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))  #類別特徵轉換成1, 2, 3, ...
    train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
    test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))
    print(col)

num_cols = ["totals.hits", "totals.pageviews", "visitNumber", 'totals.bounces',  'totals.newVisits',]  #數值型特徵陣列
            # 'totals.sessionQualityDim', 'totals.timeOnSite',
            # 'totals.transactions']
for col in num_cols:  #將數值型欄位型態轉float
    train_df[col] = train_df[col].astype(float)
    test_df[col] = test_df[col].astype(float)
    print(col)

X=train_df.drop(['date', 'fullVisitorId', 'visitId', 'visitStartTime', 'totals.transactionRevenue'], axis=1)   ## X:資料集
num_cols_2 = ['totals.sessionQualityDim', 'totals.timeOnSite',   ## 數值型態欄位
            'totals.transactions']
for col in num_cols_2:   ## 數值型態轉換float
    X[col] = X[col].astype(float)
    test_df[col]=test_df[col].astype(float)
    print(col)

y = train_df["totals.transactionRevenue"].values.astype(float) / 1000000   ##訓練資料目標值型態轉float以及還原真實金額(除以10^6)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size= 0.3, random_state=11)   ## 訓練7 / 驗證3 資料拆分

testRange = test_df[test_df['date']<=datetime(2018,7,16)]   ##測試資料範圍縮小為約20萬筆 (原始資料約40幾萬)
X_test = testRange.drop(['date', 'fullVisitorId', 'visitId', 'visitStartTime', 'totals.transactionRevenue'], axis=1)  ##測試資料集3版
y_test = testRange['totals.transactionRevenue'].values.astype(float)/1000000   ###測試資料的目標值

lgb_start=time.time()  #lgb開始時間
lgb_preds, lgb_model, lgb_evals_result = run_lgb(X_train, y_train, X_valid, y_valid, X_test, y_test)  #執行lgb模型
lgb_end=time.time()  #lgb結束時間

xgb_start=time.time()  #xgb開始時間
xgb_preds, xgb_model, xgb_evals_result = run_xgb(X_train, y_train, X_valid, y_valid, X_test, y_test)  #執行xgb模型
xgb_end=time.time()  #xgb結束時間

tree_start=time.time()  #tree開始時間
tree_preds, tree_model = run_tree(X_train, y_train, X_valid, y_valid, X_test, y_test, depth=5)  #執行decisionTree模型
tree_end=time.time()  #tree結束時間

forest_start=time.time()  #forest開始時間
forest_preds, forest_model = run_forest(X_train, y_train, X_valid, y_valid, X_test, y_test, depth=3, n_tree=30)  #執行randomforest模型
forest_end=time.time()  #forest結束時間

print(f"Train shape: {X_train.shape}")  #訓練集的維度資訊
print(f"Validation shape: {X_valid.shape}")  #驗證集的維度資訊
print(f"Test (submit) shape: {X_test.shape}")  #測試集的維度資訊

#畫迭代圖
    ##LGBM##
ax = lgb.plot_metric(lgb_evals_result, metric='rmse')
plt.title('LightGBM Metric during training (learning rate: 0.01)')
plt.show()
    ##XGB##
epochs = len(xgb_evals_result['train']['rmse'])  #取出迭代次數
x_axis = range(0, epochs)  #設定x軸的範圍 0~epochs
fig, ax = plt.subplots()   #開始畫圖
ax.plot(x_axis, xgb_evals_result['train']['rmse'], label='Train')  #畫訓練集的迭代曲線
ax.plot(x_axis, xgb_evals_result['valid']['rmse'], label='Valid')  #畫驗證集的迭代曲線
ax.legend()
plt.xlabel('iterations')
plt.ylabel('rmse')
plt.title('XGBoost Metric during training (learning rate: 0.01)')
plt.grid(True)
plt.show()

#畫importance
    ##LGBM##
ax = lgb.plot_importance(lgb_model, max_num_features=32)
plt.title('LightGBM Feature Importance (learning rate: 0.01, best iteration: 200)')
plt.show()
    ##XGB##
ax = xgb.plot_importance(xgb_model, max_num_features=32)
plt.title('XGBoost Feature Importance (learning rate: 0.01, best iteration: 100)')
plt.show()
    ##Decision Tree/RandomForest
n_features = X_train.shape[1]
plt.barh(range(n_features), tree_model.feature_importances_, align='center')
plt.yticks(np.arange(n_features), X_train.columns)
plt.grid(True)
plt.title('Decision Tree Feature Importance (depth: 2)')
plt.xlabel("Feature importance")
plt.ylabel("Features")