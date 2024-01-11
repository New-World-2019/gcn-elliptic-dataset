from sklearn.metrics import f1_score, precision_score, recall_score
from model import GCN_2layer
import numpy as np
from dataLoader import load_data
import torch
import os

print("Start loading data......")
dir = "./elliptic_bitcoin_dataset"
test_ts = np.arange(14)
adj_mats, features_labelled_ts, classes_ts = load_data(dir, 35, 49)
print("Loading data completed!")

num_features = 166   # 166 个特征
num_classes = 2      # 最后输出为两个类别

# 0 - 非法, 1 - 合法
labels_ts = []
for c in classes_ts:
    labels_ts.append(np.array(c['class'] == '2', dtype = np.compat.long))

gcn = GCN_2layer(num_features, 100, num_classes)
gcn.load_state_dict(torch.load(os.path.join("./modelDir", "gcn_weights.pth")))

# Testing
test_accs = []
test_precisions = []
test_recalls = []
test_f1s = []

for ts in test_ts:
    A = torch.tensor(adj_mats[ts].values)
    X = torch.tensor(features_labelled_ts[ts].values)
    L = torch.tensor(labels_ts[ts], dtype = torch.long)
    
    gcn.eval()
    test_out = gcn(A, X)
    
    test_pred = test_out.max(1)[1].type_as(L)
    t_acc = (test_pred.eq(L).double().sum())/L.shape[0]
    test_accs.append(t_acc.item())
    test_precisions.append(precision_score(L, test_pred))
    test_recalls.append(recall_score(L, test_pred))
    test_f1s.append(f1_score(L, test_pred))

acc = np.array(test_accs).mean()
prec = np.array(test_precisions).mean()
rec = np.array(test_recalls).mean()
f1 = np.array(test_f1s).mean()

print("GCN - averaged accuracy: {}, precision: {}, recall: {}, f1: {}".format(acc, prec, rec, f1))