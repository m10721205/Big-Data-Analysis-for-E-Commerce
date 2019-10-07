import os  #載入系統
import time  #時間套件
import json  #JSON套件
import numpy as np  #數學計算套件
import pandas as pd  #python版的excel
from pandas.io.json import json_normalize  #用python版的excel處理JSON
import matplotlib.pyplot as plt  #畫圖的
from sklearn import preprocessing  #sklearn的前處理套件
from sklearn.metrics import mean_squared_error  #sklearn中評估指標: mse
from sklearn.model_selection import train_test_split  #sklearn模型選擇套件中的訓練測試切割器
# from sklearn.model_selection import GridSearchCV  ##sklearn的網格搜尋
# from hyperopt import fmin, tpe, hp, partial, STATUS_OK, Trials  ##hyperopt是更好用的調參神器
# from numpy.random import RandomState  ##hyperopt要用到的numpy套件: random套件
from datetime import datetime  #日期時間套件
import lightgbm as lgb  #LightGBM套件
import xgboost as xgb  #XGBoost套件

pd.options.mode.chained_assignment = None #鏈式賦值警告關閉
pd.options.display.max_columns = 999      #最大顯示欄位數量

#處理JSON欄位的方法: 分解巢狀欄位
def load_df(csv_path="D:/EJ wang/google_data/train_v3.csv", nrows=None):
    JSON_COLUMNS = ["device", "geoNetwork", "totals", "trafficSource"]
    df = pd.read_csv(csv_path,
                     converters={column: json.loads for column in JSON_COLUMNS},
                     dtype={"fullVisitorId": "str"},  # Important!!
                     nrows=nrows)
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df

#計算rmse
def rmse(y_true, y_predict):
    return round(np.sqrt(mean_squared_error(y_true, y_predict)), 3) #計算rmse取小數點後3位

#LghtGBM模型建立
def run_lgb(X_train, y_train, X_valid, y_valid, X_test, y_test):
    params = {  #參數設定
        "objective" : "regression",
        "metric" : "rmse",
        "max_depth": 6,
        "num_leaves" : 50,
        "learning_rate" : 0.005,
        "bagging_fraction" : 0.75,
        "feature_fraction" : 0.85,
        "bagging_frequency" : 9,
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

#XGBoost模型建立
def run_xgb(X_train, y_train, X_val, y_val, X_test, y_test):
    params = {'objective': 'reg:linear',  #參數設定
              'eval_metric': 'rmse',
              'eta': 0.01,
              'max_depth': 8,
              'subsample': 0.7,
              'colsample_bytree': 0.7,
              'alpha':0.001,
              'silent': True}
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

# train_df["totals.transactionRevenue"]=train_df["totals.transactionRevenue"].values.astype(float)/1000000
# y = np.log1p(train_df["totals.transactionRevenue"].values.astype(float))

# train_id = train_df["fullVisitorId"].values
# test_id = test_df["fullVisitorId"].values

##用label encoding轉換"類別型"特徵
cat_cols = ["channelGrouping", "device.browser", "device.deviceCategory", "device.operatingSystem",
            "geoNetwork.city", "geoNetwork.continent", "geoNetwork.country", "geoNetwork.metro",
            "geoNetwork.networkDomain", "geoNetwork.region", "geoNetwork.subContinent", "trafficSource.adContent",
            "trafficSource.adwordsClickInfo.adNetworkType", "trafficSource.adwordsClickInfo.gclId",
            "trafficSource.adwordsClickInfo.page", "trafficSource.adwordsClickInfo.slot", "trafficSource.campaign",
            "trafficSource.keyword", "trafficSource.medium", "trafficSource.referralPath", "trafficSource.source",
            'trafficSource.adwordsClickInfo.isVideoAd', 'trafficSource.isTrueDirect']
for col in cat_cols:  #for迴圈
    lbl = preprocessing.LabelEncoder()  #建立label編碼器物件
    lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))  #類別特徵轉換成1, 2, 3, ...
    train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
    test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))
    print(col)

##將數值型資料型態轉換成float
num_cols = ["totals.hits", "totals.pageviews", "visitNumber", 'totals.bounces',  'totals.newVisits',]  #數值型特徵陣列
            # 'totals.sessionQualityDim', 'totals.timeOnSite',
            # 'totals.transactions']
for col in num_cols:  #用for將數值型欄位型態轉float
    train_df[col] = train_df[col].astype(float)
    test_df[col] = test_df[col].astype(float)
    print(col)

##訓練 / 驗證 / 測試 資料拆分
   ##method_1
X=train_df.drop(['date', 'fullVisitorId', 'visitId', 'visitStartTime', 'totals.transactionRevenue'], axis=1)   ## X:資料集
num_cols_2 = ['totals.sessionQualityDim', 'totals.timeOnSite',   ## 數值型態欄位
            'totals.transactions']
for col in num_cols_2:   ## 數值型態轉換float
    X[col] = X[col].astype(float)
    test_df[col]=test_df[col].astype(float)
    print(col)
# y=np.log1p(train_df["totals.transactionRevenue"].values.astype(float))
y = train_df["totals.transactionRevenue"].values.astype(float) / 1000000   ##訓練資料目標值型態轉float以及還原真實金額(除以10^6)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size= 0.3, random_state=11)   ## 訓練7 / 驗證3 資料拆分
# test_X2=test_df.drop(['date', 'fullVisitorId', 'visitId', 'visitStartTime', 'totals.transactionRevenue'], axis=1)  ##測試資料集2版
# y_test2 = test_df['totals.transactionRevenue'].values.astype(float) / 1000000  #測試資料的目標值

test3 = test_df[test_df['date']<=datetime(2018,7,16)]   ##測試資料範圍縮小為約16萬 (原始資料約40幾萬)
test_X3 = test3.drop(['date', 'fullVisitorId', 'visitId', 'visitStartTime', 'totals.transactionRevenue'], axis=1)  ##測試資料集3版
y_test3=test3['totals.transactionRevenue'].values.astype(float)/1000000   ###測試資料的目標值

#    ##method_2
# dev_df = train_df[train_df['date']<=datetime(2017,12,24)]   #根據最早的日期 80%為訓練資料
# val_df = train_df[train_df['date']>datetime(2017,12,24)]    #剩下20%為驗證資料
# dev_X = dev_df[cat_cols+num_cols]    # 訓練資料
# val_X = val_df[cat_cols+num_cols]    # 驗證資料
#
# dev_y = np.log1p(dev_df["totals.transactionRevenue"].values.astype(float))   #訓練資料的目標值
# val_y = np.log1p(val_df["totals.transactionRevenue"].values.astype(float))   #驗證資料的目標值
#
# test_X = test_df[cat_cols+num_cols]   #測試資料
# y_test = np.log1p(test_df['totals.transactionRevenue'].values.astype(float))  #測試資料的目標值

##建立模型
# xgb_preds, xgb_model, xgb_evals_result = run_xgb(dev_X, dev_y, val_X, val_y, test_X, y_test)
# lgb_preds, lgb_model, lgb_evals_result = run_lgb(dev_X, dev_y, val_X, val_y, test_X, y_test)

lgb_start=time.time()  #lgb開始時間
lgb_preds, lgb_model, lgb_evals_result = run_lgb(X_train, y_train, X_valid, y_valid, test_X3, y_test3)  #執行lgb模型
lgb_end=time.time()  #lgb結束時間

xgb_start=time.time()  #xgb開始時間
xgb_preds, xgb_model, xgb_evals_result = run_xgb(X_train, y_train, X_valid, y_valid, test_X3, y_test3)  #執行xgb模型
xgb_end=time.time()  #xgb結束時間

print(f"Train shape: {X_train.shape}")  #訓練集的維度資訊
print(f"Validation shape: {X_valid.shape}")  #驗證集的維度資訊
print(f"Test (submit) shape: {X_test.shape}")  #測試集的維度資訊

#畫迭代圖
    ##LGBM##
ax = lgb.plot_metric(lgb_evals_result, metric='rmse')
plt.title('LightGBM Metric during training (learning rate: 0.01)')
plt.show()
    ##XGB##
epochs = len(xgb_evals_result['train']['rmse'])
x_axis = range(0, epochs)
fig, ax = plt.subplots()
ax.plot(x_axis, xgb_evals_result['train']['rmse'], label='Train')
ax.plot(x_axis, xgb_evals_result['valid']['rmse'], label='Valid')
ax.legend()
# plt.xlabel('iterations')
plt.ylabel('rmse')
plt.title('XGBoost Metric during training (learning rate: 0.01)')
plt.grid(True)
plt.show()

#畫importance
    ##LGBM##
ax = lgb.plot_importance(lgb_model, max_num_features=28)
plt.title('LightGBM Feature Importance (learning rate: 0.01, best iteration: 200)')
plt.show()
    ##XGB##
ax = xgb.plot_importance(xgb_model, max_num_features=28)
plt.title('XGBoost Feature Importance (learning rate: 0.01, best iteration: 100)')
plt.show()

##GridSearchCV
# my_lgb=lgb.LGBMRegressor()
# param={ 'max_depth': [5, 7, 9],
#         'num_leaves': [32, 128, 512],
#         'learning_rate': [0.01, 0.05, 0.1],
#         'feature_fraction': [0.7, 0.8, 0.9] }
# param={ 'max_depth': [5, 7, 9],
#         'num_leaves': [32, 128, 256],
#         'learning_rate': [0.01, 0.1, 0.5],}
        # 'feature_fraction': [0.7, 0.8, 0.9] }
# param1={
#     'max_depth': range(3,5,7),
#     'num_leaves':range(50, 100, 150)
#        }
# grid=GridSearchCV(estimator=my_lgb, param_grid=param, scoring='neg_mean_squared_error',
#                   cv=10, verbose=True, return_train_score=True, n_jobs=-1)
# start=time.time()
# grid.fit(dev_X, dev_y)
# end=time.time()
# print("本次網格搜尋執行時間為: ", end-start) #24994.531247138977秒=416.5833分=6.943小時
#
# print("Valid+-Std     Train  :   Parameters")
# for i in np.argsort(grid.cv_results_['mean_test_score']):
#     print('{1:.3f}+-{3:.3f}     {2:.3f}   :  {0}'.format(grid.cv_results_['params'][i],
#                                     grid.cv_results_['mean_test_score'][i],
#                                     grid.cv_results_['mean_train_score'][i],
#                                     grid.cv_results_['std_test_score'][i]))

#hyperopt調參
# params_space = {'max_depth': hp.hp.quniform('max_depth',  3, 10, 1),
#                 'learning_rate': hp.hp.uniform('learning_rate', 0.001, 0.5),}
# trials = hp.Trials()
# best = hp.fmin(hyper_obj, space=params_space, algo=hp.tpe.suggest, max_evals=50, trials=trials, rstate=RandomState(123))
# print("\n展示hyperopt獲取的最佳結果，但是要注意的是我們對hyperopt最初的取值範圍做過一次轉換")
# print(best)

# def rmse(y_true, y_predict):
#     return 'rmse', round(np.sqrt(mean_squared_error(y_true, y_predict)), 3) #計算rmse取小數點後3位

# def objective(space):
#     reg = lgb.LGBMRegressor(
#                             # max_depth=int(space['max_depth']),
#                             num_leaves=int(space['num_leaves']),
#                             min_child_sample=space['min_child_sample'],
#                             learning_rate=space['learning_rate'],
#                             bagging_fraction=space['bagging_fraction'],
#                             feature_fraction=space['feature_fraction'],
#                             bagging_frequency=space['bagging_frequency'],
#                             bagging_seed=int(space['bagging_seed']),
#                             num_boost_round=int(space['num_boost_round']),
#                             objective='regression')
#     eval_set = [(dev_X, dev_y), (val_X, val_y)]
#     reg.fit(dev_X, dev_y, eval_set=eval_set, eval_metric = 'rmse')
#     pred = reg.predict(val_X)
#     performance = np.sqrt(mean_squared_error((val_y), (pred)))
#     return{'loss': performance, 'status': STATUS_OK}
# space = {
#          # 'max_depth': hp.choice('max_depth', range(3, 15)),
#          # 'max_depth': hp.quniform('max_depth', 3, 15, 1),
#          # 'num_leaves': hp.randint('num_leaves', 100),
#          # 'num_leaves': hp.choice('num_leaves', range(10, 40)),
#          'num_leaves': hp.quniform('num_leaves', 10, 40, 1),
#          'min_child_sample': hp.quniform('min_child_sample', 1, 150, 1),
#          'learning_rate': hp.quniform('learning_rate', 0.1, 0.6, 0.1),
#          'bagging_fraction': hp.quniform('bagging_fraction', 0.6, 1, 0.1),
#          'feature_fraction': hp.quniform('feature_fraction', 0.6, 1, 0.1),
#          'bagging_frequency': hp.quniform('bagging_frequency', 1, 6, 1),
#          # 'bagging_seed': hp.choice('bagging_seed', range(2018, 2020)),
#          'bagging_seed': hp.quniform('bagging_seed', 2018, 2020, 1),
#          # 'num_boost_round': hp.choice('num_boost_round', np.arange(500, 2000, 50, dtype=int)),
#          'num_boost_round': hp.quniform('num_boost_round', 500, 2000, 50),
#          }

# def objective(space):
#     reg = xgb.XGBRegressor(
#                             max_depth=int(space['max_depth']),
#                             eta=space['eta'],
#                             subsample=space['subsample'],
#                             colsample_bytree=space['colsample_bytree'],
#                             random_state=42,
#                             n_estimators=int(space['n_estimators']),
#                             objective='reg:linear')
#     eval_set = [(dev_X, dev_y), (val_X, val_y)]
#     reg.fit(dev_X, dev_y, eval_set=eval_set, eval_metric = 'rmse')
#     pred = reg.predict(val_X)
#     performance = np.sqrt(mean_squared_error((val_y), (pred)))
#     return{'loss': performance, 'status': STATUS_OK}
# space = {
#          'max_depth': hp.choice('max_depth', range(3, 15)),
#          'eta': hp.quniform('eta', 0.005, 1, 0.05),
#          'subsample': hp.quniform('subsample', 0.6, 1, 0.1),
#          'colsample_bytree': hp.quniform('colsample_bytree', 0.6, 1, 0.1),
#          'n_estimators': hp.quniform('n_estimators', 500, 2000, 25),
#          }
# trials = Trials()
# best = fmin(fn=objective,
#             space=space,
#             algo=tpe.suggest,
#             max_evals=3,  # change
#             trials=trials)
# print(best)