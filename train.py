from util import *
import sys
import tensorflow as tf
import numpy as np
from models import NIMC_MF
import time



# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)


# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'IMC', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'nimc', 'Model string.')  # 'gcn', 'gcn_appr'
flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('num_nei', 10, 'Number of neighbours to use.')
flags.DEFINE_integer('user_hidden1', 200, 'Number of units in user hidden layer 1.')
flags.DEFINE_integer('item_hidden1', 200, 'Number of units in item hidden layer 1.')
flags.DEFINE_integer('embedding_size', 200, 'Number of units in embedding.')
flags.DEFINE_float('dropout', 0.0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 50, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')
flags.DEFINE_integer('batch_size', 256, 'Number of batch size.')

def iterate_minibatches_listinputs(data, batchsize, shuffle=False):
    assert data is not None
    numSamples = data.shape[0]
    if shuffle:
        indices = np.arange(numSamples)
        np.random.shuffle(indices)
    for start_idx in range(0, numSamples - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield data[excerpt]



def main(data_name, fold):
    positions = [1, 5,10, 15]

    user_count, user_fea, item_count, item_fea, W, H = load_IMC_data(fold)
    user_fea_dim = user_fea.shape[1]
    item_fea_dim = item_fea.shape[1]
    tr_data, Tr, Tr_neg, Te = getSplit(fold,user_count,item_count)

    model_func = NIMC_MF

    placeholders = {

        #'user_AXfeatures': tf.sparse_placeholder(tf.float32),
        'user_AXfeatures': tf.placeholder(tf.float32, shape=(None,user_fea_dim)),
        #'itempos_AXfeatures': tf.sparse_placeholder(tf.float32),
        'itempos_AXfeatures': tf.placeholder(tf.float32, shape=(None,item_fea_dim)),
        #'itemneg_AXfeatures': tf.sparse_placeholder(tf.float32),
        'itemneg_AXfeatures': tf.placeholder(tf.float32, shape=(None,item_fea_dim)),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': {'user': tf.placeholder(tf.int32), 'item_pos': tf.placeholder(tf.int32),
                                 'item_neg': tf.placeholder(tf.int32)},  # helper variable for sparse dropout
        'user_field': tf.placeholder(tf.int32, shape=(None)),
        'itempos_field': tf.placeholder(tf.int32, shape=(None)),
        'itemneg_field': tf.placeholder(tf.int32, shape=(None)),
        #'weight_loss': tf.placeholder(tf.float32, shape=(None)),
        'loss_decay': tf.placeholder_with_default(1., shape=())
    }


    model = model_func(placeholders, user_length=user_count, item_length=item_count, user_input_dim=user_fea_dim, item_input_dim=item_fea_dim, logging=True)


    config = tf.ConfigProto(inter_op_parallelism_threads=12, intra_op_parallelism_threads=12)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())



    max_count = max(user_count,item_count/2+item_count%2)
    val_data = np.zeros((max_count,3),dtype=int)
    for i in range(max_count):
        val_data[i,0] = i%user_count
        val_data[i,1] = i%item_count
        val_data[i,2] = item_count - 1 -val_data[i,1]
    val_feed_dict = None
    for batch in iterate_minibatches_listinputs(val_data, val_data.shape[0]):
        print 'generate_val_feed'
        val_feed_dict = construct_feeddict_val(batch, user_fea, item_fea, Tr_neg,placeholders, sparse=False)

    M = np.zeros((user_count,item_count))

    for epoch in range(FLAGS.epochs):
        t = time.time()
        train_loss = 0.
        train_acc = 0.
        train_num = 0
        val_loss = 0.
        val_num = 0
        val_acc = 0.
        for batch in iterate_minibatches_listinputs(tr_data, batchsize=FLAGS.batch_size, shuffle=True):
        #for batch in iterate_minibatchs_user(Tr, shuffle=True):
            #t2 = time.time()
            #print 'batch_time: ', t2-t1
            feed_dict = construct_feeddict(batch,user_fea,item_fea,Tr_neg,placeholders,sparse=False)
            if feed_dict is None:
                print "None"
                continue
            #t3 = time.time()
            #print 'feed_dict_time: ', t3-t2
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})
            outs = sess.run([model.train_op, model.loss, model.accuracy], feed_dict=feed_dict)
            #val_outs = sess.run([model.test_op, model.loss, model.accuracy, model.user_history, model.item_history],
            #                feed_dict=val_feed_dict)
            #M = np.dot(val_outs[3],val_outs[4].T)

            train_loss += outs[1]
            train_acc += outs[2]

           # t1 = time.time()
            train_num += 1



        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss / train_num),
              "train_acc=", "{:.5f}".format(train_acc / train_num),
              "time=", "{:.5f}".format(time.time() - t))
        t1 = time.time()
        val_outs = sess.run([model.test_op, model.loss, model.accuracy, model.user_history, model.item_history],
                            feed_dict=val_feed_dict)
        val_num += 1
        val_loss += val_outs[1]
        val_acc += val_outs[2]
        M = np.dot(val_outs[3], val_outs[4].T)
        print time.time() - t1
        map_value, auc_value, ndcg, prec, rec = evaluate(M, Tr_neg, Te, positions)
        results = 'TEST_MAP: %s AUC:%s nDCG:%s ' % (map_value, auc_value, ndcg)
        results += ' '.join(['P@%d:%.6f' % (positions[i], prec[i]) for i in xrange(len(positions))]) + ' '
        results += ' '.join(['R@%d:%.6f' % (positions[i], rec[i]) for i in xrange(len(positions))])
        print results
    print 1

if __name__ == '__main__':
    data_name = 'IMC'
    fold = 0
    if len(sys.argv) > 3:
        data_name = sys.argv[1]
        fold = sys.argv[2]

    main(data_name, fold)

