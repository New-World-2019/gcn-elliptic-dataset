# gcn-elliptic-dataset
GCN 方法预测比特币交易中的非法节点。

# 简介
## 依赖
- Python
- PyTorch
- pandas
- numpy
- scikit-learn

## 文件说明
- dataloader.py : 加载并预处理比特币交易数据；
- model.py : GCN 模型
- layer.py : GraphConv 实现
- train.py : 训练模型；
- test.py : 测试模型；
- modelDir ：保存训练的模型。

## 使用方法
### 1.下载代码
```
git clone https://github.com/New-World-2019/gcn-elliptic-dataset.git
```
### 2.下载数据集
点击链接 [Elliptic Dataset](https://www.kaggle.com/ellipticco/elliptic-data-set#elliptic_bitcoin_dataset.zip) 下载数据集，将数据集解压到 gcn-elliptic-dataset 目录下。
### 3.训练
```
python train.py
```
### 4.测试
```
python test.py
```
