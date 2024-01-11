import numpy as np
import pandas as pd
import os

def load_data(data_dir, start_ts, end_ts):
	classes_csv = 'elliptic_txs_classes.csv'  # 203769 行，2列（id, lable）
	edgelist_csv = 'elliptic_txs_edgelist.csv' # 234355 行，2列 （id1, id2）
	features_csv = 'elliptic_txs_features.csv' # features 203769 行，每行 166 个特征

	classes = pd.read_csv(os.path.join(data_dir, classes_csv), index_col = 'txId') # labels for the transactions i.e. 'unknown', '1', '2'
	edgelist = pd.read_csv(os.path.join(data_dir, edgelist_csv), index_col = 'txId1') # directed edges between transactions
	features = pd.read_csv(os.path.join(data_dir, features_csv), header = None, index_col = 0) # features of the transactions
	
	num_features = features.shape[1]
	#print("num_features = ", num_features)
	#print("classes_csv calss = ", classes.shape[0])
	#print("edgelist_csv calss = ", edgelist.shape[0])
	num_tx = features.shape[0]  # 203769
	total_tx = list(classes.index)
	#print("num_tx = ", num_tx)  # features 203769 行，每行 166 个特征

	# select only the transactions which are labelled
	labelled_classes = classes[classes['class'] != 'unknown']
	labelled_tx = list(labelled_classes.index)
	#print("labelled_tx = ", len(labelled_tx))  # 46564 个有 lable 的行
	#print("features class = ", features[1].unique()) # 1 ~ 49 

	# to calculate a list of adjacency matrices for the different timesteps

	adj_mats = []
	features_labelled_ts = []
	classes_ts = []
	num_ts = 49 # number of timestamps from the paper
	print("共加载 %d 轮次数据(%d - %d)" % (end_ts - start_ts, end_ts, start_ts))
	for ts in range(start_ts, end_ts):
		print("加载 %d 轮次数据" % ts)
        # 选取 features[1] == ts+1 的所有行
	    features_ts = features[features[1] == ts+1]
        # 选取上一步行的 id
	    tx_ts = list(features_ts.index)
	    # 选取已标记的 id
	    labelled_tx_ts = [tx for tx in tx_ts if tx in set(labelled_tx)]
	    
	    # adjacency matrix for all the transactions
	    # we will only fill in the transactions of this timestep which have labels and can be used for training
	    # 创建二维数组，features 的全部的 id, 行和列分别命名 id 
	    adj_mat = pd.DataFrame(np.zeros((num_tx, num_tx)), index = total_tx, columns = total_tx)
	    #print("adj_mat.shape = ", adj_mat.shape) # (203769, 203769)
        # edgelist 中选取已经标记的 本轮 for 循环中的 id 列表
	    edgelist_labelled_ts = edgelist.loc[edgelist.index.intersection(labelled_tx_ts).unique()]
	    #print("edgelist_labbelled_ts: \n", edgelist_labelled_ts.head())
	    #print("edgelist_labelled_ts = ", edgelist_labelled_ts.shape[0]) # 2593
        # 将 edgelist 选择的边映射到二维矩阵中
	    for i in range(edgelist_labelled_ts.shape[0]):
	        adj_mat.loc[edgelist_labelled_ts.index[i], edgelist_labelled_ts.iloc[i]['txId2']] = 1
	    # 筛选出本轮 for 循环 【labelled_tx_ts, labelled_tx_ts】的 id 构成的矩阵      
	    adj_mat_ts = adj_mat.loc[labelled_tx_ts, labelled_tx_ts]
	    features_l_ts = features.loc[labelled_tx_ts]  # 选出对应的 features
	    
	    adj_mats.append(adj_mat_ts)
	    features_labelled_ts.append(features_l_ts)
	    classes_ts.append(classes.loc[labelled_tx_ts])
	    #print("adj_mats = ", adj_mats)
	    #print("features_labelled_ts = ", features_labelled_ts.shape)
	    #print("classes_ts = ", classes_ts.shape)

	return adj_mats, features_labelled_ts, classes_ts