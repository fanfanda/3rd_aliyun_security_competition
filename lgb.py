import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import sparse
from pandas import DataFrame
import sys
# all_data = sparse.load_npz("../ffddata/train_cv17_min5.npz")
# test_data = sparse.load_npz("../ffddata/test_cv17_min5.npz")

all_data = sparse.load_npz("../ffddata/train_base_cv14_min10.npz")
test_data = sparse.load_npz("../ffddata/test_base_cv14_min10.npz")

train_label = pd.read_csv("../ffddata/train_label.csv")
test_label = pd.read_csv("../ffddata/test_label.csv")

print("--- split ---")
x_train, x_test, y_train, y_test = train_test_split(all_data, train_label.label, test_size = 0.2, random_state = 20180921)

print("--- train ---")
clf = lgb.LGBMClassifier(
        boosting_type = 'gbdt', num_leaves = 31, reg_alpha = 1, reg_lambda = 1,
        max_depth = -1, n_estimators = 7000, objective = 'multi:softprob',
        subsample = 0.7, colsample_bytree = 0.7, subsample_freq = 1,
        learning_rate = 0.02, random_state = 8012, n_jobs = 25, verbose = -1, class_weight = {0:1, 1:3, 2:2, 3:3, 4:5, 5:1, 6:3, 7:2})

clf.fit(x_train, y_train, eval_set = [(x_train, y_train), (x_test, y_test)], early_stopping_rounds = 300)
sys.exit()
print("--- online ---")
clf = lgb.LGBMClassifier(
        boosting_type = 'gbdt', num_leaves = 31, reg_alpha = 1, reg_lambda = 1,
        max_depth = -1, n_estimators = 444, objective = 'multi:softprob',
        subsample = 0.7, colsample_bytree = 0.7, subsample_freq = 1,
        learning_rate = 0.02, random_state = 8012, n_jobs = 30, verbose = -1, class_weight = {0:1, 1:3, 2:2, 3:3, 4:5, 5:1, 6:3, 7:2})

clf.fit(all_data, train_label.label, eval_set = [(all_data, train_label.label)])

result = clf.predict_proba(test_data)

from utility import *
ss = clf.predict_proba(x_test)
print(multiclass_logloss(y_test, ss))


df_result = test_label[['file_id']]
result = DataFrame(result, columns = ['prob0', 'prob1', 'prob2', 'prob3', 'prob4', 'prob5', 'prob6', 'prob7'])

df_result = pd.concat([df_result, result], axis = 1)
# sort_result = df_result.groupby('file_id').apply(lambda x: x.sort_values(["file_id"], ascending = True)).reset_index(drop = True)

for i in range(7):
        df_result['prob' + str(i)] = df_result['prob' + str(i)].apply(lambda x: round(x, 6))
df_result['prob7'] = 1 - df_result['prob0'] - df_result['prob1'] - df_result['prob2'] - df_result['prob3'] - df_result['prob4'] - df_result['prob5'] - df_result['prob6']

df_result.to_csv("../result_ffd2.csv", index = False)
# sort_result.to_csv("../result_0821_nb15.csv", index = False)