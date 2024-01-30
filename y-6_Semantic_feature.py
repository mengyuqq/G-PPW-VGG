#需修改31、33行路径

import cv2
import torch
import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image
from torchvision import transforms, datasets
from torchvision.models.feature_extraction import create_feature_extractor

# 忽略烦人的红色提示
warnings.filterwarnings("ignore")
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
#导入训练好的模型
model = torch.load('./result/16_paper1_PMConv_L3_ECANet/best-0.9353.pth')
model = model.eval().to(device)
path = './result/16_paper1_PMConv_L3_ECANet/'


#抽取模型中间层输出结果作为语义特征
model_trunc = create_feature_extractor(model, return_nodes={'fc': '8'})

# #计算单张图像的语义特征
# img_path = 'fruit30_split/val/菠萝/105.jpg'
# img_pil = Image.open(img_path)
# input_img = data_transform["val"](img_pil) # 预处理
# input_img = input_img.unsqueeze(0).to(device)
# # 执行前向预测，得到指定中间层的输出
# pred_logits = model_trunc(input_img)
# pred_logits['semantic_feature'].squeeze().detach().cpu().numpy().shape

#载入测试集图像分类结果
df = pd.read_csv(path + '测试集预测结果.csv')

#计算测试集每张图像的语义特征
encoding_array = []
img_path_list = []

for img_path in tqdm(df['图像路径']):
    img_path_list.append(img_path)
    img_pil = Image.open(img_path).convert('RGB')
    input_img = data_transform["val"](img_pil).unsqueeze(0).to(device) # 预处理
    feature = model_trunc(input_img)['8'].squeeze().detach().cpu().numpy() # 执行前向预测，得到 avgpool 层输出的语义特征
    encoding_array.append(feature)
encoding_array = np.array(encoding_array)
# 保存为本地的 npy 文件
np.save(path + '测试集语义特征.npy', encoding_array)