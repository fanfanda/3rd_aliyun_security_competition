import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing  
import numpy as np
from pandas import DataFrame
from scipy import sparse

base_train = pd.read_csv("../ffddata/base_train.csv")
base_test = pd.read_csv("../ffddata/base_test.csv")

# base_train = pd.read_csv("../ffddata/train_cv17_min5.csv")
# base_test = pd.read_csv("../ffddata/test_cv17_min5.csv")

# train_label = base_train[['file_id', 'label']]
# test_label = base_test[['file_id']]

base_train = base_train.drop(['file_id', 'label'], axis = 1)
base_test = base_test.drop(['file_id', 'label'], axis = 1)

# cv_train = sparse.load_npz('../ffddata/train_cv14.npz')
# cv_test = sparse.load_npz('../ffddata/test_cv14.npz')

cv_train = sparse.load_npz('../ffddata/train_cv14_min10.npz')
cv_test = sparse.load_npz('../ffddata/test_cv14_min10.npz')

temp_train = sparse.hstack((base_train, cv_train))
temp_test = sparse.hstack((base_test, cv_test))

sparse.save_npz("../ffddata/train_base_cv14_min10.npz", temp_train)
sparse.save_npz("../ffddata/test_base_cv14_min10.npz", temp_test)

# train_label.to_csv("../ffddata/train_label.csv", index = False)
# test_label.to_csv("../ffddata/test_label.csv", index = False)