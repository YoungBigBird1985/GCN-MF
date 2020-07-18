from util import *

def main():
    fold_list = [0,1,2]
    katz_score, catapult_score = load_score_mat()
    positions = [1,5,10,15]
    for fold in fold_list:
        user_count, user_fea, item_count, item_fea, W, H = load_IMC_data(fold)
        tr_data, Tr, Tr_neg, Te = getSplit(fold, user_count, item_count)
        k_score = katz_score[fold]
        c_score = catapult_score[fold]
        map_value, auc_value, ndcg_new, prec, rec = evaluate(k_score.transpose(), Tr_neg, Te, positions)
        results = 'TEST_MAP: %.4f & %.4f & %.4f ' % (map_value, auc_value, ndcg_new)
        results += ' '.join(['&%.4f' % (prec[i]) for i in xrange(len(positions))]) + ' '
        results += ' '.join(['&%.4f' % (rec[i]) for i in xrange(len(positions))])
        print results
        map_value, auc_value, ndcg_new, prec, rec = evaluate(c_score.transpose(), Tr_neg, Te, positions)
        results = 'TEST_MAP: %.4f & %.4f & %.4f ' % (map_value, auc_value, ndcg_new)
        results += ' '.join(['&%.4f' % (prec[i]) for i in xrange(len(positions))]) + ' '
        results += ' '.join(['&%.4f' % (rec[i]) for i in xrange(len(positions))])
        print results
        print fold

if __name__ == '__main__':
    main()


