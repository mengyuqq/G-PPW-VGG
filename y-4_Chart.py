#需修改21行路径

import cv2
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from matplotlib import colors as mcolors
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# windows操作系统
plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号

path = './result/16_paper1_PMConv_L3_ECANet/'

#载入类别名称和ID
idx_to_labels = np.load(path + 'idx_to_labels.npy', allow_pickle=True).item()

# 获得类别名称
classes = list(idx_to_labels.values())
print(classes)

#载入测试集预测结果表格
df = pd.read_csv(path + '测试集预测结果.csv')

#生成混淆矩阵
confusion_matrix_model = confusion_matrix(df['Label the category name'], df['top-1-预测名称'])
import itertools


def cnf_matrix_plotter(cm, classes, cmap=plt.cm.Blues):
    """
    传入混淆矩阵和标签名称列表，绘制混淆矩阵
    """
    plt.figure(figsize=(10, 10))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    # plt.colorbar() # 色条
    tick_marks = np.arange(len(classes))

    plt.title('G-PPW-VGG11 confusion matrix', font={'family': 'Times New Roman', 'size': 36})
    plt.xlabel('Prediction category', font={'family': 'Times New Roman', 'size': 36})
    plt.ylabel('True category', font={'family': 'Times New Roman', 'size': 36})
    plt.tick_params(labelsize=25)  # 设置类别文字大小
    plt.xticks(tick_marks, classes, rotation=45, font={'family': 'Times New Roman', 'size': 25})  # 横轴文字旋转
    plt.yticks(tick_marks, classes, font={'family': 'Times New Roman', 'size': 25})

    # 写数字
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black",
                 fontsize=25)

    plt.tight_layout()

    plt.savefig(path + 'G-PPW-VGG11-混淆矩阵.png', dpi=300)  # 保存图像
    plt.show()
cnf_matrix_plotter(confusion_matrix_model, classes, cmap='Blues')

# 查看所有配色方案
# dir(plt.cm)

# #筛选出测试集中，真实为A类，但被误判为B类的图像
# true_A = 'gm6'
# pred_B = '101'
# wrong_df = df[(df['标注类别名称']==true_A)&(df['top-1-预测名称']==pred_B)]
# for idx, row in wrong_df.iterrows():
#     img_path = row['图像路径']
#     img_bgr = cv2.imread(img_path)
#     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#     plt.imshow(img_rgb)
#     title_str = img_path + '\nTrue:' + row['标注类别名称'] + ' Pred:' + row['top-1-预测名称']
#     plt.title(title_str)
#     plt.show()

#绘制PR曲线
#绘制所以类别地PR曲线

# random.seed(124)
# colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'black', 'indianred', 'brown', 'firebrick', 'maroon', 'darkred', 'red', 'sienna', 'chocolate', 'yellow', 'olivedrab', 'yellowgreen', 'darkolivegreen', 'forestgreen', 'limegreen', 'darkgreen', 'green', 'lime', 'seagreen', 'mediumseagreen', 'darkslategray', 'darkslategrey', 'teal', 'darkcyan', 'dodgerblue', 'navy', 'darkblue', 'mediumblue', 'blue', 'slateblue', 'darkslateblue', 'mediumslateblue', 'mediumpurple', 'rebeccapurple', 'blueviolet', 'indigo', 'darkorchid', 'darkviolet', 'mediumorchid', 'purple', 'darkmagenta', 'fuchsia', 'magenta', 'orchid', 'mediumvioletred', 'deeppink', 'hotpink']
# markers = [".",",","o","v","^","<",">","1","2","3","4","8","s","p","P","*","h","H","+","x","X","D","d","|","_",0,1,2,3,4,5,6,7,8,9,10,11]
# linestyle = ['--', '-.', '-']
#
# def get_line_arg():
#     '''
#     随机产生一种绘图线型
#     '''
#     line_arg = {}
#     line_arg['color'] = random.choice(colors)
#     # line_arg['marker'] = random.choice(markers)
#     line_arg['linestyle'] = random.choice(linestyle)
#     line_arg['linewidth'] = random.randint(1, 4)
#     # line_arg['markersize'] = random.randint(3, 5)
#     return line_arg
#
# plt.figure(figsize=(14, 10))
# plt.xlim([-0.01, 1.0])
# plt.ylim([0.0, 1.01])
# # plt.plot([0, 1], [0, 1],ls="--", c='.3', linewidth=3, label='随机模型')
# plt.xlabel('Recall', fontsize=30)
# plt.ylabel('Precision', fontsize=30)
# plt.title('PR曲线', fontsize=30)
# plt.rcParams['font.size'] = 25
# plt.grid(True)
#
# ap_list = []
# for each_class in classes:
#     y_test = list((df['Label the category name'] == each_class))
#     y_score = list(df['{}-预测置信度'.format(each_class)])
#     precision, recall, thresholds = precision_recall_curve(y_test, y_score)
#     AP = average_precision_score(y_test, y_score, average='weighted')
#     plt.plot(recall, precision, **get_line_arg(), label=each_class)
#     plt.legend()
#     ap_list.append(AP)
#
# plt.legend(loc='best', fontsize=25)
# plt.savefig(path + '各类别PR曲线.png'.format(classes), dpi=120, bbox_inches='tight')
# plt.show()
#
# #将AP增加至各类别准确率评估指标表格中
# df_report = pd.read_csv(path + '各类别准确率评估指标.csv')
# # 计算 AUC值 的 宏平均 和 加权平均
# macro_avg_auc_AP = np.mean(ap_list)
# weighted_avg_auc_AP = sum(ap_list * df_report.iloc[:-2]['support'] / len(df))
# ap_list.append(macro_avg_auc_AP)
# ap_list.append(weighted_avg_auc_AP)
# df_report['AP'] = ap_list
# df_report.to_csv(path + '各类别准确率评估指标.csv', index=False)
# # #绘制某一类PR曲线
# # specific_class = 'h_101_f'
# # # 二分类标注
# # y_test = (df['标注类别名称'] == specific_class)
# # # 二分类预测置信度
# # y_score = df['h_101_f-预测置信度']
# # precision, recall, thresholds = precision_recall_curve(y_test, y_score)
# # AP = average_precision_score(y_test, y_score, average='weighted')
# # plt.figure(figsize=(12, 8))
# # # 绘制 PR 曲线
# # plt.plot(recall, precision, linewidth=5, label=specific_class)
# #
# # # 随机二分类模型
# # # 阈值小，所有样本都被预测为正类，recall为1，precision为正样本百分比
# # # 阈值大，所有样本都被预测为负类，recall为0，precision波动较大
# # plt.plot([0, 0], [0, 1], ls="--", c='.3', linewidth=3, label='随机模型')
# # plt.plot([0, 1], [0.5, sum(y_test==1)/len(df)], ls="--", c='.3', linewidth=3)
# #
# # plt.xlim([-0.01, 1.0])
# # plt.ylim([0.0, 1.01])
# # plt.rcParams['font.size'] = 22
# # plt.title('{} PR曲线  AP:{:.3f}'.format(specific_class, AP))
# # plt.xlabel('Recall')
# # plt.ylabel('Precision')
# # plt.legend()
# # plt.grid(True)
# # plt.savefig(path + '{}-PR曲线.pdf'.format(specific_class), dpi=120, bbox_inches='tight')
# # plt.show()
#
# #绘制ROC曲线
# #绘制所有类别ROC曲线
# plt.figure(figsize=(14, 10))
# plt.xlim([-0.01, 1.0])
# plt.ylim([0.0, 1.01])
# plt.plot([0, 1], [0, 1],ls="--", c='.3', linewidth=3, label='随机模型')
# plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=30)
# plt.ylabel('True Positive Rate (Sensitivity)', fontsize=30)
# plt.title('ROC曲线')
# plt.rcParams['font.size'] = 25
# plt.grid(True)
#
# auc_list = []
# for each_class in classes:
#     y_test = list((df['Label the category name'] == each_class))
#     y_score = list(df['{}-预测置信度'.format(each_class)])
#     fpr, tpr, threshold = roc_curve(y_test, y_score)
#     plt.plot(fpr, tpr, **get_line_arg(), label=each_class)
#     plt.legend()
#     auc_list.append(auc(fpr, tpr))
#
# plt.legend(loc='best', fontsize=25)
# plt.savefig(path + '各类别ROC曲线.png'.format(classes), dpi=120, bbox_inches='tight')
# plt.show()
#
# #将AUC增加至各类别准确率评估指标表格中
# # 计算 AUC值 的 宏平均 和 加权平均
# macro_avg_auc_AUC = np.mean(auc_list)
# weighted_avg_auc_AUC = sum(auc_list * df_report.iloc[:-2]['support'] / len(df))
# auc_list.append(macro_avg_auc_AUC)
# auc_list.append(weighted_avg_auc_AUC)
# df_report['AUC'] = auc_list
# df_report.to_csv(path + '各类别准确率评估指标.csv', index=False)
#
#
#
# # #绘制某一类别的ROC曲线
# # specific_class = 'h_101_f'
# # # 二分类标注
# # y_test = (df['标注类别名称'] == specific_class)
# # # 二分类置信度
# # y_score = df['h_101_f-预测置信度']
# # fpr, tpr, threshold = roc_curve(y_test, y_score)
# # plt.figure(figsize=(12, 8))
# # plt.plot(fpr, tpr, linewidth=5, label=specific_class)
# # plt.plot([0, 1], [0, 1],ls="--", c='.3', linewidth=3, label='随机模型')
# # plt.xlim([-0.01, 1.0])
# # plt.ylim([0.0, 1.01])
# # plt.rcParams['font.size'] = 22
# # plt.title('{} ROC曲线  AUC:{:.3f}'.format(specific_class, auc(fpr, tpr)))
# # plt.xlabel('False Positive Rate (1 - Specificity)')
# # plt.ylabel('True Positive Rate (Sensitivity)')
# # plt.legend()
# # plt.grid(True)
# #
# # plt.savefig(path + '{}-ROC曲线.pdf'.format(specific_class), dpi=120, bbox_inches='tight')
# # plt.show()
# # auc(fpr, tpr)
# # # yticks = ax.yaxis.get_major_ticks()
# # # yticks[0].label1.set_visible(False)