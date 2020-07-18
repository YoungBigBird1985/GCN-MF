
from util import *
from scipy import spatial
from scipy import sparse

def main():
    fold = 1
    knn = 10
    user_count, user_fea, item_count, item_fea, W, H = load_IMC_data(fold)

    user_graph = np.zeros((user_count,user_count))
    #for i in range(1):
    for i in range(user_count-1):
        user_graph[i, i+1:] = 1-spatial.distance.cdist(user_fea[i+1:, :], user_fea[i, :].reshape(1,-1), 'cosine').reshape(-1,)
        user_graph[i+1:, i] = user_graph[i, i+1:]
    idx_sort = np.argsort(-user_graph,axis = 1)
    user_knn_graph = np.zeros(user_graph.shape)
    for i in range(user_count):
        user_knn_graph[i,idx_sort[i,:knn+1]] = user_graph[i,idx_sort[i,:knn+1]]
        user_knn_graph[idx_sort[i,:knn+1],i] = user_graph[idx_sort[i,:knn+1],i]

    item_graph = np.zeros((item_count, item_count))
    for i in range(item_count-1):
        item_graph[i, i+1:] = 1-spatial.distance.cdist(item_fea[i+1:, :], item_fea[i, :].reshape(1,-1), 'cosine').reshape(-1,)
        item_graph[i+1:, i] = item_graph[i, i+1:]
    idx_sort = np.argsort(-item_graph,axis = 1)
    item_knn_graph = np.zeros(item_graph.shape)
    for i in range(item_count):
        item_knn_graph[i,idx_sort[i,:knn+1]] = item_graph[i,idx_sort[i,:knn+1]]
        item_knn_graph[idx_sort[i,:knn+1],i] = item_graph[idx_sort[i,:knn+1],i]

    user_sparse = sparse.coo_matrix(user_knn_graph)
    item_sparse = sparse.coo_matrix(item_knn_graph)
    sparse.save_npz('data/user_graph_fold'+str(fold)+'.npz', user_sparse)
    sparse.save_npz('data/item_graph_fold'+str(fold)+'.npz', item_sparse)
    print 1

if __name__ == '__main__':
    main()