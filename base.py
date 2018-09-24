import pandas as pd

print("--- load ---")
train_data = pd.read_csv("../final_train.csv")
test_data = pd.read_csv("../final_test.csv")

train_data['file_id'] = train_data.file_id + 20000
all_data = train_data.append(test_data, ignore_index = True, sort = False)
print("--- 1 ---")
all_data['sequence_length'] = all_data.groupby("file_id").file_id.transform('count')
all_data['max_index_length'] = all_data.groupby("file_id").index.transform('max') + 1
print("--- 2 ---")
all_data['tid_count'] = all_data.groupby("file_id").tid.transform('nunique') # tid切换次数
all_data['mode_tid'] = all_data.groupby("file_id").tid.transform(lambda x: x.mode()[0]) # tid众数
all_data['max_tid'] = all_data.groupby("file_id").tid.transform('max')
all_data['min_tid'] = all_data.groupby("file_id").tid.transform('min')
print("--- 3 ---")
all_data['mean_seq_length'] = all_data['sequence_length'] / all_data['tid_count']

# api的统计特征
all_data['distinct_api_count'] = all_data.groupby("file_id").api.transform('nunique') # 总共出现了多少种api

featured_all_data = all_data.drop(['tid', 'api', 'index'], axis = 1).drop_duplicates().reset_index(drop = True)

out_train = featured_all_data[featured_all_data.file_id > 20000]
out_test = featured_all_data[featured_all_data.file_id < 20000]

print("--- save ---")
out_train.to_csv("../ffddata/base_train.csv", index = False)
out_test.to_csv("../ffddata/base_test.csv", index = False)