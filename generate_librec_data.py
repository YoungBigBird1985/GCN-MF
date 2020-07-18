from util import *
import os


def main():
    for fold in [0,1,2]:
        dataset_str = 'IMC'
        user_count, user_fea, item_count, item_fea, W, H = load_IMC_data(fold)
        user_fea_dim = user_fea.shape[1]
        item_fea_dim = item_fea.shape[1]
        tr_data, te_data, Tr, Tr_neg, Te = getSplit_test(fold,user_count,item_count)
        tr_data_lib = np.ones((tr_data.shape[0],tr_data.shape[1]+1),dtype=int)
        tr_data_lib[:,:2] = tr_data
        te_data_lib = np.ones((te_data.shape[0],te_data.shape[1]+1),dtype=int)
        te_data_lib[:,:2] = te_data

        np.savetxt(os.path.join('data/' + dataset_str +'/fold_'+str(fold)+'/librec_implicit_training.txt'), tr_data_lib, fmt='%d')
        np.savetxt(os.path.join('data/' + dataset_str +'/fold_'+str(fold)+'/librec_implicit_testing.txt'), te_data_lib, fmt='%d')
        print 1

if  __name__ == '__main__':
    main()