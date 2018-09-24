from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from scipy import sparse
import sys
import pandas as pd

print("--- load ---")
all_data = pd.read_csv("../final_train.csv", usecols = ['file_id','api'])
all_data = all_data.groupby("file_id").apply(lambda x: " ".join(x.api)).reset_index(name = "list_text")

test_data = pd.read_csv("../final_test.csv", usecols = ['file_id','api'])
test_data = test_data.groupby("file_id").apply(lambda x: " ".join(x.api)).reset_index(name = "list_text")

cv_vectorizer = CountVectorizer(ngram_range = (1, 4), min_df = 20)
print("--- fit ---")
cv_vectorizer.fit(all_data['list_text'])
print("--- convert ---")
cv_matrix_train = cv_vectorizer.transform(all_data.list_text)
cv_matrix_test = cv_vectorizer.transform(test_data.list_text)

print("--- save ---")
sparse.save_npz("../ffddata/train_cv14_min20.npz", cv_matrix_train)
sparse.save_npz("../ffddata/test_cv14_min20.npz", cv_matrix_test)