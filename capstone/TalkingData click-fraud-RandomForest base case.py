""" data processing code is partially taken from this kernel:
https://www.kaggle.com/pranav84/lightgbm-fixing-unbalanced-data-lb-0-9680/code
"""

import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import gc

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

print('loading train data...')
train_df = pd.read_csv(path+"train.csv", skiprows=range(1,180903891), nrows=4000000, dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])

print('loading test data...')
test_df = pd.read_csv(path+"test.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])

print('creating training and validation data...')
x_train, x_val, y_train, y_val = train_test_split(train_df.loc[: , train_df.columns != 'is_attributed'], train_df['is_attributed'], test_size = 0.1, random_state = 100)

del train_df
gc.collect()

print("train size: ", len(x_train))
print("validation size: ", len(x_val))
print("test size : ", len(test_df))

predictors = ['ip','app','device','os', 'channel']

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')

gc.collect()

print("Training...")
start_time = time.time()

clf = RandomForestClassifier(max_depth=10, random_state=42, verbose=1)
clf.fit(x_train[predictors], y_train)

print('model training time: {} seconds'.format(time.time() - start_time))

print('validation auc score:', roc_auc_score(y_val, clf.predict_proba(x_val[predictors])[:,1]))

del x_train
del x_val
del y_train
del y_val
gc.collect()

print("Predicting test data...")
sub['is_attributed'] = clf.predict_proba(test_df[predictors])[:,1]
print("writing...")
sub.to_csv('sub_random_forest.csv',index=False)
print("done...")