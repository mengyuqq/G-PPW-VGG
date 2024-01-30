#需要修改13行路径和44行模型路径

import os
import time
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from torchvision import transforms, datasets

path = './result/16_paper1_PMConv_L3_ECANet/'
model = torch.load('./result/16_paper1_PMConv_L3_ECANet/best-0.9353.pth')

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

#图像预处理
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}
#载入测试集
data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
image_path = os.path.join(data_root, "MRP", "21_6_3")  # flower data set path
assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                        transform=data_transform["val"])
print('测试集图像数量', len(validate_dataset))
print('类别个数', len(validate_dataset.classes))
print('各类别名称', validate_dataset.classes)
# 载入类别名称 和 ID索引号 的映射字典
idx_to_labels = np.load(path + 'idx_to_labels.npy', allow_pickle=True).item()
# 获得类别名称
classes = list(idx_to_labels.values())

#导入训练好的模型

model = model.eval().to(device)


#表格A-测试集图像路径及标注
img_paths = [each[0] for each in validate_dataset.imgs]
df = pd.DataFrame()
df['图像路径'] = img_paths
df['标注类别ID'] = validate_dataset.targets
df['Label the category name'] = [idx_to_labels[ID] for ID in validate_dataset.targets]
start_time = time.time()
#表格B-测试集每张图像的图像分类预测结果，以及各类别置信度
# 记录 top-n 预测结果
n = 3
df_pred = pd.DataFrame()
for idx, row in tqdm(df.iterrows()):
    img_path = row['图像路径']
    img_pil = Image.open(img_path).convert('RGB')
    input_img = data_transform["val"](img_pil).unsqueeze(0).to(device)  # 预处理
    pred_logits = model(input_img)  # 执行前向预测，得到所有类别的 logit 预测分数
    pred_softmax = F.softmax(pred_logits, dim=1)  # 对 logit 分数做 softmax 运算

    pred_dict = {}

    top_n = torch.topk(pred_softmax, n)  # 取置信度最大的 n 个结果
    pred_ids = top_n[1].cpu().detach().numpy().squeeze()  # 解析出类别

    # top-n 预测结果
    for i in range(1, n + 1):
        pred_dict['top-{}-预测ID'.format(i)] = pred_ids[i - 1]
        pred_dict['top-{}-预测名称'.format(i)] = idx_to_labels[pred_ids[i - 1]]
    pred_dict['top-n预测正确'] = row['标注类别ID'] in pred_ids
    # 每个类别的预测置信度
    for idx, each in enumerate(classes):
        pred_dict['{}-预测置信度'.format(each)] = pred_softmax[0][idx].cpu().detach().numpy()

    df_pred = df_pred.append(pred_dict, ignore_index=True)
end_time = time.time() - start_time
print("预测{}张图像所需时间{}".format(len(validate_dataset),end_time))
#拼接AB两张表格
df = pd.concat([df, df_pred], axis=1)
df.to_csv(path + '测试集预测结果.csv', index=False)

print("结束")