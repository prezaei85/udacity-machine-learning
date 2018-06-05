"""
inspired by:
https://www.kaggle.com/pranav84/lightgbm-fixing-unbalanced-data-lb-0-9680/code
"""

import pandas as pd
import time
import numpy as np
import lightgbm as lgb
import gc

n_rows = 184903891
n_train = 40000000
validation_percent = 5

path = '../input/'

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }


print('loading training data...')
train_df = pd.read_csv(path+"train.csv", skiprows=range(1,n_rows-n_train), nrows=n_train, dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])

print('loading testing data...')
test_df = pd.read_csv(path+"test.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])

len_train = len(train_df)
train_df = train_df.append(test_df, sort=False)

del test_df
gc.collect()

print('Extracting new features...')
train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
gc.collect()

print('grouping by ip-day-hour...')
gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_day_hour_count'})
train_df = train_df.merge(gp, on=['ip','day','hour'], how='left')
del gp
gc.collect()

print('grouping by ip-app...')
gp = train_df[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
train_df = train_df.merge(gp, on=['ip','app'], how='left')
del gp
gc.collect()

print('grouping by ip-app-os...')
gp = train_df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
del gp
gc.collect()

print('grouping by ip-app-device-os...')
gp = train_df[['ip','app','device','os', 'channel']].groupby(by=['ip', 'app','device','os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_device_os_count'})
train_df = train_df.merge(gp, on=['ip','app','device','os'], how='left')
del gp
gc.collect()

# Adding features with var and mean hour (inspired from nuhsikander's script)
print('grouping by ip_day_channel_var_hour...')
gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','channel'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_day_channel_var_hour'})
train_df = train_df.merge(gp, on=['ip','day','channel'], how='left')
del gp
gc.collect()

print('grouping by ip_app_os_var_hour...')
gp = train_df[['ip','app', 'os', 'hour']].groupby(by=['ip', 'app', 'os'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_app_os_var_hour'})
train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
del gp
gc.collect()

print('grouping by ip_app_channel_var_day...')
gp = train_df[['ip','app', 'channel', 'day']].groupby(by=['ip', 'app', 'channel'])[['day']].var().reset_index().rename(index=str, columns={'day': 'ip_app_channel_var_day'})
train_df = train_df.merge(gp, on=['ip','app', 'channel'], how='left')
del gp
gc.collect()

print('grouping by ip_app_channel_mean_hour...')
gp = train_df[['ip','app', 'channel','hour']].groupby(by=['ip', 'app', 'channel'])[['hour']].mean().reset_index().rename(index=str, columns={'hour': 'ip_app_channel_mean_hour'})
train_df = train_df.merge(gp, on=['ip','app', 'channel'], how='left')
del gp
gc.collect()

print("vars and data type: ")
train_df.info()
train_df['ip_day_hour_count'] = train_df['ip_day_hour_count'].astype('uint16')
train_df['ip_app_count'] = train_df['ip_app_count'].astype('uint16')
train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')
train_df['ip_app_device_os_count'] = train_df['ip_app_device_os_count'].astype('uint16')

test_df = train_df[n_train:]
val_df = train_df[int(n_train*(1-validation_percent/100)):n_train]
train_df = train_df[:int(n_train*(1-validation_percent/100))]

print("training size: ", len(train_df))
print("validation size: ", len(val_df))
print("testing size: ", len(test_df))

target = 'is_attributed'
predictors = ['app','device','os', 'channel', 'hour', #'day', 
              'ip_day_hour_count', 'ip_app_count', 'ip_app_os_count',
              'ip_app_device_os_count', 'ip_day_channel_var_hour', 
              'ip_app_os_var_hour', 'ip_app_channel_var_day', 'ip_app_channel_mean_hour']
categorical = ['app', 'device', 'os', 'channel', 'hour'] #, 'day']

gc.collect()

def lgb_fit_model(params, train_data, validation_data, predictors, target, 
        categorical_features, verbose_eval=10):

    xgtrain = lgb.Dataset(train_data[predictors].values, label=train_data[target].values,
                          feature_name=predictors, categorical_feature=categorical_features)
    xgvalid = lgb.Dataset(validation_data[predictors].values, label=validation_data[target].values,
                          feature_name=predictors, categorical_feature=categorical_features)

    evals_results = {}

    gbm = lgb.train(params, 
                     xgtrain, 
                     valid_sets=[xgtrain, xgvalid], 
                     valid_names=['train','valid'], 
                     evals_result=evals_results, 
                     num_boost_round=params['num_boost_round'],
                     early_stopping_rounds=params['early_stopping_rounds'],
                     verbose_eval=verbose_eval)

    n_estimators = gbm.best_iteration
    print("\nModel Report")
    print("n_estimators : ", n_estimators)
    print(params['metric']+":", evals_results['valid'][params['metric']][n_estimators-1])

    return gbm

print("Training...")
start_time = time.time()

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'learning_rate': 0.15,
    'scale_pos_weight':99, # because training data is extremely unbalanced 
    #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
    'num_leaves': 7,  # we should let it be smaller than 2^(max_depth)
    'max_depth': 3,  # -1 means no limit
    'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 100,  # Number of bucketed bin for feature values
    'subsample': 0.7,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequency of subsample, <=0 means no enable
    'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'nthread': 4,
    'verbose': 0,
    'metric': 'auc',
    'early_stopping_rounds': 30,
    'num_boost_round': 500,
}

gbm = lgb_fit_model(params, train_df, val_df, predictors, target, categorical)

print('training time: {}'.format(time.time() - start_time))
del train_df
del val_df
gc.collect()

print("Predicting...")
dt = pd.DataFrame()
dt['click_id'] = test_df['click_id'].astype('int')
dt['is_attributed'] = gbm.predict(test_df[predictors])
dt.to_csv('lgb_more_features.csv',index=False)
print("done.")s