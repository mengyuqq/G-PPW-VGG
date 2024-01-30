import os
import json
import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchsummary import summary
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from torchvision import transforms, datasets
from torch.optim import lr_scheduler
from mobilenet_v3 import mobilenet_v3_large

if os.path.exists("result/2_mobilenet_v3_large") is False:
    os.makedirs("result/2_mobilenet_v3_large")
path = "result/2_mobilenet_v3_large/"

batch_size = 112
epochs = 120

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),   #随机裁剪
                                 transforms.RandomHorizontalFlip(),   #随机水平翻转
                                 transforms.ToTensor(),   #转换成Tensor格式
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),  #标准化处理
    "val": transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
# print(data_root)
image_path = os.path.join(data_root, "MRP", "21_6_3")  # flower data set path
# print(image_path)
assert os.path.exists(image_path), "{} path does not exist.".format(image_path)

train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                     transform=data_transform["train"])

validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                        transform=data_transform["val"])
train_num = len(train_dataset)
val_num = len(validate_dataset)
# print('训练集图像数量：', len(train_dataset))
# print('训练集类别个数：', len(train_dataset.classes))
# print('训练集类别名称：', train_dataset.classes)
# print(train_num)
# print('验证集图像数量：', len(validate_dataset))
# print('验证集类别个数：', len(validate_dataset.classes))
# print('验证集类别名称：', validate_dataset.classes)
# print(val_num)

# {'daisy':0, 'dandelion':1, 'roses':2, 'sunflower':3, 'tulips':4}
flower_list = train_dataset.class_to_idx  #映射关系：类别——索引号
cla_dict = dict((val, key) for key, val in flower_list.items())   #映射关系：索引号——类别
# write dict into json file
json_str = json.dumps(cla_dict, indent=4)
with open(path + 'class_indices.json', 'w') as json_file:
    json_file.write(json_str)

idx_to_labels = cla_dict
np.save(path + 'idx_to_labels.npy', idx_to_labels)
np.save(path + 'labels_to_idx.npy', train_dataset.class_to_idx)

#nw线程个数
nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
#nw = 4
print('Using {} dataloader workers every process'.format(nw))

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=batch_size, shuffle=False,
                                              num_workers=0)
print("using {} images for training, {} images for validation.".format(train_num,
                                                                       val_num))
# test_data_iter = iter(validate_loader)
# test_image, test_label = test_data_iter.next()

# #查看一个batch的图像和标注
# images,labels = next(iter(train_loader))
# print(images.shape)
# print(labels)
#
# #可视化一个batch的图像和标注
# images = images.numpy()
# idx=5
# print(images[idx].shape)   #选取索引为5的图片
# plt.hist(images[idx].flatten(),bins=50)
# plt.show()

model = mobilenet_v3_large(6)
model.to(device)
print(model)
summary(model, (3, 224, 224))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
# lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15, verbose=True, min_lr=0.0, eps=1e-8)

# 设置此参数，用于后期只保存最优的模型
best_acc = 0.0

def train_one_batch(images, labels):
    '''
    运行一个 batch 的训练，返回当前 batch 的训练日志
    '''

    # 获得一个 batch 的数据和标注
    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)  # 输入模型，执行前向预测
    loss = criterion(outputs, labels)  # 计算当前 batch 中，每个样本的平均交叉熵损失函数值

    # 优化更新权重
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 获取当前 batch 的标签类别和预测类别
    _, preds = torch.max(outputs, 1)  # 获得当前 batch 所有图像的预测类别
    preds = preds.cpu().numpy()
    loss = loss.detach().cpu().numpy()
    outputs = outputs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()



    log_train = {}
    log_train['epoch'] = epoch
    log_train['batch'] = batch_idx
    # 计算分类评估指标
    log_train['train_loss'] = loss
    log_train['train_accuracy'] = accuracy_score(labels, preds)
    # log_train['train_precision'] = precision_score(labels, preds, average='macro')
    # log_train['train_recall'] = recall_score(labels, preds, average='macro')
    # log_train['train_f1-score'] = f1_score(labels, preds, average='macro')

    return log_train


def evaluate_testset():
    '''
    在整个测试集上评估，返回分类评估指标日志
    '''

    loss_list = []
    labels_list = []
    preds_list = []

    with torch.no_grad():
        for images, labels in validate_loader:  # 生成一个 batch 的数据和标注
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)  # 输入模型，执行前向预测

            # 获取整个测试集的标签类别和预测类别
            _, preds = torch.max(outputs, 1)  # 获得当前 batch 所有图像的预测类别
            preds = preds.cpu().numpy()
            loss = criterion(outputs, labels)  # 由 logit，计算当前 batch 中，每个样本的平均交叉熵损失函数值
            loss = loss.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            loss_list.append(loss)
            labels_list.extend(labels)
            preds_list.extend(preds)

    log_test = {}
    log_test['epoch'] = epoch

    # 计算分类评估指标
    log_test['test_loss'] = np.mean(loss_list)
    log_test['test_accuracy'] = accuracy_score(labels_list, preds_list)
    log_test['test_precision'] = precision_score(labels_list, preds_list, average='macro')
    log_test['test_recall'] = recall_score(labels_list, preds_list, average='macro')
    log_test['test_f1-score'] = f1_score(labels_list, preds_list, average='macro')

    return log_test

#训练开始之前，记录日志
epoch = 0
batch_idx = 0
best_test_accuracy = 0
best_train_accuracy = 0

# 训练日志-训练集
df_train_log = pd.DataFrame()
df_epoch_train_log = pd.DataFrame()

log_train = {}
log_train['epoch'] = 0
log_train['batch'] = 0
log_epoch_train = {}
log_epoch_train['epoch'] = 0
images, labels = next(iter(train_loader))
log_train.update(train_one_batch(images, labels))
df_train_log = df_train_log.append(log_train, ignore_index=True)
df_epoch_train_log = df_epoch_train_log.append(log_epoch_train, ignore_index=True)

df_lr_log = pd.DataFrame()
log_lr = {}
log_lr['epoch'] = 0

# 训练日志-测试集
df_test_log = pd.DataFrame()
log_test = {}
log_test['epoch'] = 0
log_test.update(evaluate_testset())
df_test_log = df_test_log.append(log_test, ignore_index=True)


start = time.time()
#运行训练
for epoch in range(1, epochs + 1):

    start_epoch = time.time()
    print(f'Epoch {epoch}/{epochs}')

    log_lr = {}
    log_lr['epoch'] = epoch
    # 计算分类评估指标
    log_lr['lr'] = optimizer.param_groups[0]["lr"]
    df_lr_log = df_lr_log.append(log_lr, ignore_index=True)
    print(epoch, optimizer.param_groups[0]["lr"])

    train_epoch_acc = 0.0
    train_epoch_loss = 0.0
    ## 训练阶段
    model.train()
    for images, labels in tqdm(train_loader):  # 获得一个 batch 的数据和标注
        batch_idx += 1
        log_train = train_one_batch(images, labels)
        # train_epoch_acc = train_epoch_acc + log_train['train_accuracy']
        # train_epoch_loss= train_epoch_loss + log_train['train_loss']
        df_train_log = df_train_log.append(log_train, ignore_index=True)
        # wandb.log(log_train)

    train_epoch_acc = np.mean(log_train['train_accuracy'])
    train_epoch_loss = np.mean(log_train['train_loss'])
    log_epoch_train = {}
    log_epoch_train['epoch'] = epoch
    # 计算分类评估指标
    log_epoch_train['train_loss'] = train_epoch_loss
    log_epoch_train['train_accuracy'] = train_epoch_acc
    df_epoch_train_log = df_epoch_train_log.append(log_epoch_train, ignore_index=True)



    ## 测试阶段
    model.eval()
    log_test = evaluate_testset()
    df_test_log = df_test_log.append(log_test, ignore_index=True)
    # wandb.log(log_test)
    print('epoch: {}, Train Acc: {:.6f}, Train Loss: {:.6f}, Val_Acc: {:.6f}, Val_Loss: {:.6f}'
          .format(epoch, train_epoch_acc, train_epoch_loss, log_test['test_accuracy'], log_test['test_loss']))
    # lr_scheduler.step(log_test['test_accuracy'])  # 更新学习率

    # 保存最新的最佳模型文件
    if log_test['test_accuracy'] > best_test_accuracy:

        # 保存新的最佳模型文件
        best_test_accuracy = log_test['test_accuracy']
        new_best_checkpoint_path = path + 'best-{:.4f}.pth'.format(log_test['test_accuracy'])
        torch.save(model, new_best_checkpoint_path)
        print('保存新的最佳模型', 'best-{:.4f}.pth'.format(best_test_accuracy))
    if log_epoch_train['train_accuracy'] > best_train_accuracy:
        best_train_accuracy = log_epoch_train['train_accuracy']
    stop_epoch = time.time() - start_epoch
    print('第{}个epoch训练所需时间为：{}'.format(epoch, stop_epoch))


df_epoch_train_log.to_csv(path + '训练epoch日志-训练集.csv', index=False)
df_train_log.to_csv(path + '训练日志-训练集.csv', index=False)
df_test_log.to_csv(path + '训练日志-测试集.csv', index=False)
df_lr_log.to_csv(path + '训练日志-学习率.csv', index=False)

print("训练集最好准确率为：", best_train_accuracy)
print("测试集最好准确率为：", best_test_accuracy)
stop = time.time() - start
print('训练完所有epoch所需时间为：{}'.format(stop))


print('Finished Training')



