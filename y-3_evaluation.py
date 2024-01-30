#需修改8行路径

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report

path = './result/16_paper1_PMConv_L3_ECANet/'

idx_to_labels = np.load(path + 'idx_to_labels.npy', allow_pickle=True).item()
# 获得类别名称
classes = list(idx_to_labels.values())
print(classes)

#载入测试集预测表格
df = pd.read_csv(path + '测试集预测结果.csv')

#各类评价指标
report = classification_report(df['Label the category name'], df['top-1-预测名称'], target_names=classes, output_dict=True)
del report['accuracy']
df_report = pd.DataFrame(report).transpose()

accuracy_list = []
for fruit in tqdm(classes):
    df_temp = df[df['Label the category name']==fruit]
    accuracy = sum(df_temp['Label the category name'] == df_temp['top-1-预测名称']) / len(df_temp)
    accuracy_list.append(accuracy)

# 计算 宏平均准确率 和 加权平均准确率
acc_macro = np.mean(accuracy_list)
acc_weighted = sum(accuracy_list * df_report.iloc[:-2]['support'] / len(df))

accuracy_list.append(acc_macro)
accuracy_list.append(acc_weighted)

df_report['accuracy'] = accuracy_list

df_report.to_csv(path + '各类别准确率评估指标.csv', index_label='类别')