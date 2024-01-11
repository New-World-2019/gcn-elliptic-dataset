import numpy as np
import time
import sys
import pandas as pd
import os
from dataLoader import load_data
from model import GCN_2layer
import torch
import torch.nn as nn


dir = "./elliptic_bitcoin_dataset"
dataSet = load_data(dir, 0, 34)

num_features = 166   # 166 个特征
num_classes = 2      # 最后输出为两个类别
num_ts = 49          # 49 子图？
epochs = 15          # epochs
lr = 0.001           # 学习率
#max_train_ts = 34   # 0 ~ 34 用于训练，35 ~ 49 用于测试
max_train_ts = 5
train_ts = np.arange(max_train_ts)  # [0,1,......,33] 的列表 

adj_mats, features_labelled_ts, classes_ts = dataSet

# 转换标签为 0/1 的形式
# 0 - 非法, 1 - 合法
labels_ts = []
#print("classes_ts = ", classes_ts)
for c in classes_ts:
    labels_ts.append(np.array(c['class'] == '2', dtype = np.long))
#print("labels_ts = \n", labels_ts)
# num_features 输入/特征个数  100 隐藏单元个数  num_classes 输出类别个数
gcn = GCN_2layer(num_features, 100, num_classes)
# 损失函数
train_loss = nn.CrossEntropyLoss(weight = torch.DoubleTensor([0.7, 0.3]))
# 优化函数
optimizer = torch.optim.Adam(gcn.parameters(), lr = lr)

# Training
# for 0 - 33 根据features[1]取每个类别，数据预处理也是按照类别存储的 adj_mats/features_labelled_ts/labels_ts
for ts in train_ts:
    A = torch.tensor(adj_mats[ts].values)
    #print("A = \n", A, "len(A) = ", len(A))
    X = torch.tensor(features_labelled_ts[ts].values)
    #print("X = \n", X, "len(X) = ", len(X))
    L = torch.tensor(labels_ts[ts], dtype = torch.long)
    #print("labels_ts = \n", labels_ts, "len(labels_ts) = ", len(labels_ts))
    for ep in range(epochs):
        t_start = time.time()
        
        gcn.train()
        optimizer.zero_grad()
        out = gcn(A, X)

        loss = train_loss(out, L)
        train_pred = out.max(1)[1].type_as(L)
        acc = (train_pred.eq(L).double().sum())/L.shape[0]

        loss.backward()
        optimizer.step()

        sys.stdout.write("\r Epoch %d/%d Timestamp %d/%d training loss: %f training accuracy: %f Time: %s"
                         %(ep, epochs, ts, max_train_ts, loss, acc, time.time() - t_start)
                        )
modelDirPath = "./modelDir"
if not os.path.isdir(modelDirPath):
    print("创建目录：", modelDirPath)
    os.makedirs(modelDirPath)
torch.save(gcn.state_dict(), str(os.path.join("./modelDir", "gcn_weights.pth")))