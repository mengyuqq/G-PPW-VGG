import cv2
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# windows操作系统
plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号
warnings.filterwarnings("ignore")#忽略警告

path = './result/16_paper1_PMConv_L3_ECANet/'
#载入测试集图像语义特征
encoding_array = np.load(path + '测试集语义特征.npy', allow_pickle=True)
#载入测试集图像分类结果
df = pd.read_csv(path + '测试集预测结果.csv')

classes = df['Label the category name'].unique()
print(classes)

#可视化配置
# marker_list = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+', 'x', 'X', 'D', 'd', '|', '_', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
# class_list = np.unique(df['标注类别名称'])
marker_list = [1, 2, 3, 4, 5, 6]
class_list = np.unique(df['Label the category name'])

n_class = len(class_list) # 测试集标签类别数
palette = sns.hls_palette(n_class) # 配色方案
sns.palplot(palette)

# 随机打乱颜色列表和点型列表
# import random
# random.seed(1234)
# random.shuffle(marker_list)
# random.shuffle(palette)

#t-SNE降维至二维
# 降维到二维和三维

tsne = TSNE(n_components=2, n_iter=20000)
X_tsne_2d = tsne.fit_transform(encoding_array)

#可视化展示
# 不同的 符号 表示 不同的 标注类别
show_feature = 'Label the category name'
# plt.figure(figsize=(14, 14))
# for idx, fruit in enumerate(class_list): # 遍历每个类别
#     # 获取颜色和点型
#     color = palette[idx]
#     marker = marker_list[idx%len(marker_list)]
#
#     # 找到所有标注类别为当前类别的图像索引号
#     indices = np.where(df[show_feature]==fruit)
#     plt.scatter(X_tsne_2d[indices, 0], X_tsne_2d[indices, 1], color=color, marker=marker, label=fruit, s=150)
#
# plt.legend(fontsize=24, markerscale=1, bbox_to_anchor=(1, 1))
# plt.xticks([])
# plt.yticks([])
# plt.title('PPW_VGG11+PMConv+ECANet的t-SNE图', fontsize=30)
# plt.savefig(path + '语义特征t-SNE二维降维可视化.png', dpi=300) # 保存图像
# plt.show()

#plotply交互式可视化
df_2d = pd.DataFrame()
df_2d['X'] = list(X_tsne_2d[:, 0].squeeze())
df_2d['Y'] = list(X_tsne_2d[:, 1].squeeze())
df_2d['Label the category name'] = df['Label the category name']
df_2d['预测类别'] = df['top-1-预测名称']
df_2d['图像路径'] = df['图像路径']
df_2d.to_csv(path + 't-SNE-2D.csv', index=False)
fig = px.scatter(df_2d,
                 x='X',
                 y='Y',
                 color=show_feature,
                 labels=show_feature,
                 symbol=show_feature,
                 hover_name='图像路径',
                 opacity=0.8,
                 width=1000,
                 height=600
                )
# 设置排版
fig.update_layout(font_size=16, margin=dict(l=0, r=0, b=0, t=0))
fig.show()
fig.write_html(path + '语义特征t-SNE二维降维plotly可视化.html')
print("二维图已保存")

# # 查看图像
# img_path_temp = '../dataset/paper1/val/h_101_f/f_300_te_101_1.jpg'
# img_bgr = cv2.imread(img_path_temp)
# img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
# plt.imshow(img_rgb)
# temp_df = df[df['图像路径'] == img_path_temp]
# title_str = img_path_temp + '\nTrue:' + temp_df['标注类别名称'].item() + ' Pred:' + temp_df['top-1-预测名称'].item()
# plt.title(title_str)
# plt.show()
#
# #t-SNE降维至三维，并可视化
# # 降维到三维
# tsne = TSNE(n_components=3, n_iter=10000)
# X_tsne_3d = tsne.fit_transform(encoding_array)
# show_feature = '标注类别名称'
# # show_feature = '预测类别'
# df_3d = pd.DataFrame()
# df_3d['X'] = list(X_tsne_3d[:, 0].squeeze())
# df_3d['Y'] = list(X_tsne_3d[:, 1].squeeze())
# df_3d['Z'] = list(X_tsne_3d[:, 2].squeeze())
# df_3d['标注类别名称'] = df['标注类别名称']
# df_3d['预测类别'] = df['top-1-预测名称']
# df_3d['图像路径'] = df['图像路径']
# df_3d.to_csv(path + 't-SNE-3D.csv', index=False)
#
# fig = px.scatter_3d(df_3d,
#                     x='X',
#                     y='Y',
#                     z='Z',
#                     color=show_feature,
#                     labels=show_feature,
#                     symbol=show_feature,
#                     hover_name='图像路径',
#                     opacity=0.6,
#                     width=1000,
#                     height=800)
#
# # 设置排版
# fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
# fig.show()
# fig.write_html(path + '语义特征t-SNE三维降维plotly可视化.html')
# print("三维图已保存")
#
#

print("结束")