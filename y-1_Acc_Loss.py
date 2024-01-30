import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# windows操作系统
plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号

path = "./result/18_22m/"

train = pd.read_csv(path + '训练epoch日志-训练集.csv')
test = pd.read_csv(path + '训练日志-测试集.csv')


plt.figure()
plt.plot(np.arange(len(train['train_accuracy'])), train['train_accuracy'], label="train acc")
plt.plot(np.arange(len(train['train_loss'])), train['train_loss'], label="train loss")
plt.plot(np.arange(len(test['test_accuracy'])), test['test_accuracy'], label="test acc")
plt.plot(np.arange(len(test['test_loss'])), test['test_loss'], label="test loss")
plt.legend()  # 显示图例
plt.xlabel('epochs')
# plt.ylabel("epoch")
plt.title('PPW_VGG11-Model accuracy&loss')
plt.savefig(path + "acc&loss.png")
plt.show()

lr = pd.read_csv(path + '训练日志-学习率.csv')
plt.figure()
plt.plot(np.arange(len(lr['lr'])), lr['lr'], label="lr")
plt.legend()  # 显示图例
plt.xlabel('epochs')
plt.ylabel("lr")
plt.title('Model lr')
plt.savefig(path + "lr.png")
plt.show()