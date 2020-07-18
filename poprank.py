
from util import *
import json
from networkx.readwrite import json_graph
import os
import tensorflow as tf
import time

def main():
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('dataset', 'ml-100k', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
    flags.DEFINE_string('model', 'pop_rank', 'Model string.')  # 'gcn', 'gcn_appr'
    flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
    flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
    flags.DEFINE_integer('num_nei', 10, 'Number of neighbours to use.')
    flags.DEFINE_integer('user_hidden1', 128, 'Number of units in user hidden layer 1.')
    flags.DEFINE_integer('item_hidden1', 128, 'Number of units in item hidden layer 1.')
    flags.DEFINE_integer('embedding_size', 50, 'Number of units in embedding.')
    flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
    flags.DEFINE_integer('early_stopping', 30, 'Tolerance for early stopping (# of epochs).')
    flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
    flags.DEFINE_integer('batch_size', 64, 'Number of batch size.')
    positions = [1, 5, 10, 15]
    fold = 2
    user_count, user_fea, item_count, item_fea, W, H = load_IMC_data(fold)
    user_fea_dim = user_fea.shape[1]
    item_fea_dim = item_fea.shape[1]
    data, Tr, Tr_neg, Te = getSplit(fold,user_count,item_count)
    package_dir = os.path.dirname(os.path.abspath(__file__))
    inf = open(os.path.join(package_dir, 'output/' + FLAGS.model + '_' + FLAGS.dataset + '_results.txt'), 'a+')
    inf.write("==================================Start=============================\r")
    parameter = 'learning_rate: ' + str(FLAGS.learning_rate) + ', num_nei: ' + str(
        FLAGS.num_nei) + ', user_hidden1: ' + str(FLAGS.user_hidden1) + \
                ', item_hidden1: ' + str(FLAGS.item_hidden1) + ', embedding_size: ' + str(
        FLAGS.embedding_size) + 'dropout: ' + str(FLAGS.dropout) + \
                'weight_decay: ' + str(FLAGS.weight_decay) + ', batch_size: ' + str(FLAGS.batch_size)

    M = np.zeros((user_count, item_count))
    for i in range(data.shape[0]):
        M[:,data[i,1]] =  M[:,data[i,1]] + 1

    map_value, auc_value, ndcg, prec, rec = evaluate(M, Tr_neg, Te, positions)
    results = 'TEST_MAP: %.4f & %.4f & %.4f ' % (map_value, auc_value, ndcg)
    results += ' '.join(['&%.4f' % (prec[i]) for i in xrange(len(positions))]) + ' '
    results += ' '.join(['&%.4f' % (rec[i]) for i in xrange(len(positions))])
    print results
    inf.write(results + '\r')
    inf.write("==================================End=============================\r")
    inf.close()

    print 1

if __name__ == '__main__':
    main()