
import scipy.io as sio
import pickle as cpk
import numpy as np
import time
import scipy as sp
import scipy.sparse
import math


def load_score_mat():
    katz_score = []
    sim = sio.loadmat('data/katzres.mat')
    katz_score.append(sim['ScoreMatrixKatz1'])
    katz_score.append(sim['ScoreMatrixKatz2'])
    katz_score.append(sim['ScoreMatrixKatz3'])
    catapult_score = []
    sim = sio.loadmat('data/catapultres.mat')
    catapult_score.append(sim['ScoreMatrixCatapult1'])
    catapult_score.append(sim['ScoreMatrixCatapult2'])
    catapult_score.append(sim['ScoreMatrixCatapult3'])
    return katz_score, catapult_score


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.sparse.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def load_IMC_data(fold=0):
    features = sio.loadmat('data/' + 'IMC' + str(fold) + '.mat')
    geneFeatures = features['Features']
    pheneFeatures = features['ColFeatures']
    numGenes = len(geneFeatures)
    numPhenes = len(pheneFeatures)
    W = features['W']
    H = features['H']
    return numPhenes, pheneFeatures, numGenes, geneFeatures,  W, H


def getSim(fold=0):
    user_adj = sp.sparse.load_npz('data/user_graph_fold'+str(fold)+'.npz')
    item_adj = sp.sparse.load_npz('data/item_graph_fold'+str(fold)+'.npz')
    return user_adj, item_adj



def normalize_adj(adj,w = 1.,knn = 10):
    """Symmetrically normalize adjacency matrix."""

    adj = sp.sparse.eye(adj.shape[0]) + w*adj
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.sparse.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocsr()



def getUserSimilarity():
    sim = sio.loadmat('data/PhenoSim.mat')['PhenoSim']
    return sim

def getItemSimilarity():
    sim = sio.loadmat('data/GeneSimilarities.mat')['GeneGene_Hs']
    return sim

def getSplit(fold,user_length,item_length):
    splits, negative = cpk.load(open('data/'+'splits.p', 'rb'))
    tr_data = set()
    te_data = set()
    for i in range(len(splits)):
        if i!=fold:
            tr_data = tr_data.union(splits[i])
        else:
            te_data = te_data.union(splits[i])

    Tr = {}
    Tr_neg = {}
    Te = {}
    for user_id in range(user_length):
        Tr[user_id] = []
    for tr_pair in tr_data:
        user_id = tr_pair[1]
        item_id = tr_pair[0]
        if user_id not in Tr:
            Tr[user_id] = [item_id]
        else:
            Tr[user_id].append(item_id)
    for te_pair in te_data:
        user_id = te_pair[1]
        item_id = te_pair[0]
        if user_id not in Te:
            Te[user_id] = [item_id]
        else:
            Te[user_id].append(item_id)
    items = set(range(item_length))
    for user_id in range(user_length):
        Tr_neg[user_id] = list(items-set(Tr[user_id]))
    tr_data_array = np.array(list(tr_data))
    tr_data = np.zeros(tr_data_array.shape,dtype= int)
    tr_data[:,0] = tr_data_array[:,1]
    tr_data[:,1] = tr_data_array[:,0]
    return tr_data, Tr, Tr_neg, Te

def getSplit_test(fold,user_length,item_length):
    splits, negative = cpk.load(open('data/'+'splits.p', 'rb'))
    tr_data = set()
    te_data = set()
    for i in range(len(splits)):
        if i!=fold:
            tr_data = tr_data.union(splits[i])
        else:
            te_data = te_data.union(splits[i])

    Tr = {}
    Tr_neg = {}
    Te = {}
    for user_id in range(user_length):
        Tr[user_id] = []
    for tr_pair in tr_data:
        user_id = tr_pair[1]
        item_id = tr_pair[0]
        if user_id not in Tr:
            Tr[user_id] = [item_id]
        else:
            Tr[user_id].append(item_id)
    for te_pair in te_data:
        user_id = te_pair[1]
        item_id = te_pair[0]
        if user_id not in Te:
            Te[user_id] = [item_id]
        else:
            Te[user_id].append(item_id)
    items = set(range(item_length))
    for user_id in range(user_length):
        Tr_neg[user_id] = list(items-set(Tr[user_id]))
    tr_data_array = np.array(list(tr_data))
    tr_data = np.zeros(tr_data_array.shape,dtype= int)
    tr_data[:,0] = tr_data_array[:,1]
    tr_data[:,1] = tr_data_array[:,0]
    te_data_array = np.array(list(te_data))
    te_data = np.zeros(te_data_array.shape,dtype= int)
    te_data[:,0] = te_data_array[:,1]
    te_data[:,1] = te_data_array[:,0]
    return tr_data, te_data, Tr, Tr_neg, Te

def generate_neg_sample(user,Tr_neg):
    neg_samples = np.zeros(user.shape,dtype=int)
    for idx,u in enumerate(user):
        neg_samples[idx] = np.random.choice(Tr_neg[u])
    return neg_samples

def construct_feeddict(batch,user_fea,item_fea,Tr_neg,placeholders,sparse=False):
    feed_dict = dict()
    support = {}
    user = batch[:,0]
    pos_item = batch[:,1]
    t1 = time.time()
    neg_item = generate_neg_sample(user,Tr_neg)
    weight_loss = np.ones(user.shape)
    #print 'generate_sample_time: ', time.time()-t1
    t1 = time.time()
    idx_p_user = user
    #print 'compute_adj_time: ', time.time() - t1
    t1 = time.time()
    idx_p_pos_item = pos_item
    idx_p_neg_item = neg_item
    #print 'construct_adj_time: ', time.time() - t1
    t1 = time.time()

    num_dict = {}
    if(sparse):
        input_user_feature = sparse_to_tuple(user_fea[idx_p_user,:])
        input_itempos_feature = sparse_to_tuple(item_fea[idx_p_pos_item,:])
        input_itemneg_feature = sparse_to_tuple(item_fea[idx_p_neg_item,:])
        num_dict['user'] = input_user_feature[1].shape
        num_dict['item_pos'] = input_itempos_feature[1].shape
        num_dict['item_neg'] = input_itemneg_feature[1].shape
    else:
        input_user_feature = user_fea[idx_p_user, :]
        input_itempos_feature = item_fea[idx_p_pos_item, :]
        input_itemneg_feature = item_fea[idx_p_neg_item, :]
        num_dict['user'] = input_user_feature.shape[0]
        num_dict['item_pos'] = input_itempos_feature.shape[0]
        num_dict['item_neg'] = input_itemneg_feature.shape[0]
    #print 'contruct_dict_time: ', time.time() - t1

    feed_dict.update({placeholders['user_AXfeatures']: input_user_feature})
    feed_dict.update({placeholders['itempos_AXfeatures']: input_itempos_feature})
    feed_dict.update({placeholders['itemneg_AXfeatures']: input_itemneg_feature})
    feed_dict.update({placeholders['user_field']: user})
    feed_dict.update({placeholders['itempos_field']: pos_item})
    feed_dict.update({placeholders['itemneg_field']: neg_item})
    feed_dict.update({placeholders['num_features_nonzero'][name]: num_dict[name] for name in num_dict.keys()})
    #feed_dict.update({placeholders['weight_loss']: weight_loss})
    feed_dict.update({placeholders['loss_decay']: 1.0*user.shape[0]/batch.shape[0]})

    return feed_dict

def construct_feeddict_val(batch,user_fea,item_fea,Tr_neg,placeholders,sparse=False):
    feed_dict = dict()
    support = {}
    user = batch[:,0]
    pos_item = batch[:,1]
    t1 = time.time()
    neg_item = batch[:,2]
    weight_loss = np.ones(user.shape)
    #print 'generate_sample_time: ', time.time()-t1
    t1 = time.time()
    idx_p_user = user
    #print 'compute_adj_time: ', time.time() - t1
    t1 = time.time()
    idx_p_pos_item = pos_item
    idx_p_neg_item = neg_item
    #print 'construct_adj_time: ', time.time() - t1
    t1 = time.time()

    num_dict = {}
    if(sparse):
        input_user_feature = sparse_to_tuple(user_fea[idx_p_user,:])
        input_itempos_feature = sparse_to_tuple(item_fea[idx_p_pos_item,:])
        input_itemneg_feature = sparse_to_tuple(item_fea[idx_p_neg_item,:])
        num_dict['user'] = input_user_feature[1].shape
        num_dict['item_pos'] = input_itempos_feature[1].shape
        num_dict['item_neg'] = input_itemneg_feature[1].shape
    else:
        input_user_feature = user_fea[idx_p_user, :]
        input_itempos_feature = item_fea[idx_p_pos_item, :]
        input_itemneg_feature = item_fea[idx_p_neg_item, :]
        num_dict['user'] = input_user_feature.shape[0]
        num_dict['item_pos'] = input_itempos_feature.shape[0]
        num_dict['item_neg'] = input_itemneg_feature.shape[0]
    #print 'contruct_dict_time: ', time.time() - t1

    feed_dict.update({placeholders['user_AXfeatures']: input_user_feature})
    feed_dict.update({placeholders['itempos_AXfeatures']: input_itempos_feature})
    feed_dict.update({placeholders['itemneg_AXfeatures']: input_itemneg_feature})
    feed_dict.update({placeholders['user_field']: user})
    feed_dict.update({placeholders['itempos_field']: pos_item})
    feed_dict.update({placeholders['itemneg_field']: neg_item})
    feed_dict.update({placeholders['num_features_nonzero'][name]: num_dict[name] for name in num_dict.keys()})
    #feed_dict.update({placeholders['weight_loss']: weight_loss})
    feed_dict.update({placeholders['loss_decay']: 1.0*user.shape[0]/batch.shape[0]})

    return feed_dict

def construct_feeddict_smodel(user_fea,item_fea,labels, label_weights,placeholders,support =None,sparse=False):
    feed_dict = dict()
    #print 'generate_sample_time: ', time.time()-t1
    t1 = time.time()
    #print 'compute_adj_time: ', time.time() - t1
    t1 = time.time()
    #print 'construct_adj_time: ', time.time() - t1
    t1 = time.time()



    num_dict = {}
    #print 'contruct_dict_time: ', time.time() - t1
    num_dict['user'] = user_fea.shape[0]
    num_dict['item_pos'] = item_fea.shape[0]

    #feed_dict.update({placeholders['support'][name]: support[name] for name in support.keys()})

    feed_dict.update({placeholders['user_AXfeatures']: user_fea})
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['label_weights']: label_weights})
    feed_dict.update({placeholders['itempos_AXfeatures']: item_fea})
    feed_dict.update({placeholders['num_features_nonzero'][name]: num_dict[name] for name in num_dict.keys()})
    feed_dict.update({placeholders['user_field']: np.array(range(user_fea.shape[0]))})
    feed_dict.update({placeholders['itempos_field']: np.array(range(item_fea.shape[0]))})

    #feed_dict.update({placeholders['weight_loss']: weight_loss})

    return feed_dict

def evaluate(M, Tr_neg, Te, positions=[5, 10, 15]):
    prec = np.zeros(len(positions))
    rec = np.zeros(len(positions))
    map_value, auc_value, ndcg = 0.0, 0.0, 0.0
    for u in Te:
        val = M[u, :]
        inx = np.array(Tr_neg[u])
        A = set(Te[u])
        B = set(inx) - A
        # compute precision and recall
        ii = np.argsort(val[inx])[::-1][:max(positions)]
        prec += precision(Te[u], inx[ii], positions)
        rec += recall(Te[u], inx[ii], positions)
        ndcg_user = nDCG(Te[u], inx[ii], 10)
        # compute map and AUC
        pos_inx = np.array(list(A))
        neg_inx = np.array(list(B))
        map_user, auc_user = map_auc(pos_inx, neg_inx, val)
        ndcg += ndcg_user
        map_value += map_user
        auc_value += auc_user
        # outf.write(" ".join([str(map_user), str(auc_user), str(ndcg_user)])+"\n")
    # outf.close()
    return map_value / len(Te.keys()), auc_value / len(Te.keys()), ndcg / len(Te.keys()), prec / len(
        Te.keys()), rec / len(Te.keys())

def precision(actual, predicted, N):
    if isinstance(N, int):
        inter_set = set(actual) & set(predicted[:N])
        return float(len(inter_set))/float(N)
    elif isinstance(N, list):
        return np.array([precision(actual, predicted, n) for n in N])


def recall(actual, predicted, N):
    if isinstance(N, int):
        inter_set = set(actual) & set(predicted[:N])
        return float(len(inter_set))/float(len(set(actual)))
    elif isinstance(N, list):
        return np.array([recall(actual, predicted, n) for n in N])


def nDCG(Tr, topK, num=None):
    if num is None:
        num = len(topK)
    dcg, vec = 0, []
    for i in range(num):
        if topK[i] in Tr:
            dcg += 1/math.log(i+2, 2)
            vec.append(1)
        else:
            vec.append(0)
    vec.sort(reverse=True)
    idcg = sum([vec[i]/math.log(i+2, 2) for i in range(num)])
    if idcg > 0:
        return dcg/idcg
    else:
        return idcg


def map_auc(pos_inx, neg_inx, val):
    map = 0.0
    pos_val, neg_val = val[pos_inx], val[neg_inx]
    ii = np.argsort(pos_val)[::-1]
    jj = np.argsort(neg_val)[::-1]
    pos_sort, neg_sort = pos_val[ii], neg_val[jj]
    auc_num = 0.0
    for i,pos in enumerate(pos_sort):
        num = 0.0
        for neg in neg_sort:
            if pos<=neg:
                num+=1
            else:
                auc_num+=1
        map += (i+1)/(i+num+1)
    return map/len(pos_inx), auc_num/(len(pos_inx)*len(neg_inx))
