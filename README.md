# gcn-elliptic-dataset
GCN 方法预测区块链中的非法节点。

# 简介
## 依赖
- Python
- PyTorch
- pandas
- scikit-learn

## 文件说明
- dataloader.py : 预处理数据；
- model.py : GCN 模型
- layer.py : Gcn 实现
- train.py : 训练模型；
- test.py : 测试模型；

## 使用方法
### 下载代码
```
git clone https://github.com/New-World-2019/gcn-elliptic-dataset.git
```
### 下载数据集
通过下载地址 [Elliptic Dataset](https://www.kaggle.com/ellipticco/elliptic-data-set#elliptic_bitcoin_dataset.zip) 下载数据集，将数据集放置到 gcn-elliptic-dataset 目录下。
### 创建 modelDir
在gcn-elliptic-dataset 目录下创建目录 modelDir 用于保存训练的模型。
### 训练
```
python train.py
```
### 测试
```
python test.py
```
