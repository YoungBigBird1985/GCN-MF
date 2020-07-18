import numpy as np
import os
from util import *
from scipy import spatial
from scipy import sparse


def main():
    package_dir = os.path.dirname(os.path.abspath(__file__))
    data_name = 'data/ml-100k'
    datasize_file = 'datasize_percent.txt'
    train_file = 'train_percent.txt'
    user_count, item_count = load_datasize(os.path.join(package_dir,data_name+'/'+datasize_file))
    data = np.loadtxt(os.path.join(package_dir,data_name+'/'+train_file), dtype=int)
    visit_matrix = np.zeros((user_count,item_count))
    for i in range(data.shape[0]):
        userid = data[i, 0]
        itemid = data[i, 1]
        visit_matrix[userid, itemid] = 1

    user_graph = np.zeros((user_count,user_count))
    item_graph = np.zeros((item_count,item_count))
    for i in range(user_count-1):
        if sum(visit_matrix[i,:])>0:
            user_graph[i, i+1:] = 1-spatial.distance.cdist(visit_matrix[i+1:, :], visit_matrix[i, :].reshape(1,-1), 'cosine').reshape(-1,)
            user_graph[i+1:, i] = user_graph[i, i+1:]
        else:
            user_graph[i, i+1:] = 0
            user_graph[i+1:, i] = 0
    for i in range(item_count-1):
        if sum(visit_matrix[:,i])>0:
            item_graph[i, i+1:] = 1-spatial.distance.cdist(visit_matrix[:, i+1:].T, visit_matrix[:, i].reshape(1,-1), 'cosine').reshape(-1,)
            item_graph[i+1:, i] = item_graph[i, i+1:]
        else:
            item_graph[i, i+1:] = 0
            item_graph[i+1:, i] = 0
    where_are_NaNs = np.isnan(user_graph)
    user_graph[where_are_NaNs] = 0
    where_are_NaNs = np.isnan(item_graph)
    item_graph[where_are_NaNs] = 0

    user_sparse = sparse.coo_matrix(user_graph)
    item_sparse = sparse.coo_matrix(item_graph)
    sparse.save_npz(os.path.join(package_dir, data_name+'/user_graph.npz'), user_sparse)
    sparse.save_npz(os.path.join(package_dir, data_name+'/item_graph.npz'), item_sparse)
    print user_count, item_count



if __name__ == "__main__":
    main()