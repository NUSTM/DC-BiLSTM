#!/usr/bin/env python
# encoding: utf-8
# @author: zxding
# email: d.z.x@qq.com

import sys, os, time
import numpy as np
import tensorflow as tf
sys.path.append("..")
from utils.process_data import *
from utils.tf_funcs import *

FLAGS = tf.app.flags.FLAGS
# >>>>>>>>>>>>>>>>>>>> For Model <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_integer('num_gpu', 1, 'number of gpu')
## embedding parameters ##
tf.app.flags.DEFINE_string('w2v_file', 'data/sst-Glove-vectors.txt', 'embedding file')
tf.app.flags.DEFINE_integer('embedding_dim', 300, 'dimension of word embedding')
tf.app.flags.DEFINE_integer('embedding_type', 0, 'way of using word embedding, 0 for static')
## input struct ##
tf.app.flags.DEFINE_integer('max_doc_len', 50, 'max number of tokens per sentence')
## model struct ##
dense_h=[13]*15
tf.app.flags.DEFINE_integer('n_hidden', 100, 'number of hidden unit')
tf.app.flags.DEFINE_integer('n_class', 2, 'number of distinct class')
# >>>>>>>>>>>>>>>>>>>> For Data <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_string('train_file_path', 'data/rt-polarity', 'train file path')
tf.app.flags.DEFINE_string('test_index', 1, 'test index for n folds validation')
tf.app.flags.DEFINE_boolean('log_to_file', 0, ' whether log to file')
tf.app.flags.DEFINE_string('log_file_name', '', 'name of log file')
# >>>>>>>>>>>>>>>>>>>> For Training <<<<<<<<<<<<<<<<<<<< #
tf.app.flags.DEFINE_integer('batch_size', 200, 'number of example per batch')
tf.app.flags.DEFINE_float('learning_rate', 0.005, 'learning rate')
tf.app.flags.DEFINE_float('keep_prob1', 1.0, 'word embedding training dropout keep prob')
tf.app.flags.DEFINE_float('keep_prob2', 1.0, 'softmax layer dropout keep prob')
tf.app.flags.DEFINE_float('l2_reg', 0.000, 'l2 regularization')
tf.app.flags.DEFINE_integer('training_iter', 20, 'number of train iter')
tf.app.flags.DEFINE_string('scope', 'RNN', 'RNN scope')



def build_model(word_embedding,x,doc_len,keep_prob1,keep_prob2,RNN = biLSTM_multigpu):
    w = tf.nn.embedding_lookup(word_embedding, x)
    w = tf.nn.dropout(w, keep_prob=keep_prob1)
    # w shape:        [-1, FLAGS.max_doc_len, FLAGS.embedding_dim]
    with tf.name_scope('word_encode'):
        for i in range(len(dense_h)):
            h = RNN(w, doc_len, n_hidden=dense_h[i], scope=FLAGS.scope+'_word_dense_layer_'+str(i))
            w=tf.concat([w, h], 2)     
        w = RNN(w, doc_len, n_hidden=FLAGS.n_hidden, scope=FLAGS.scope+'word_layer_last')
    # w shape:        [-1, FLAGS.max_doc_len, FLAGS.n_hidden*2]
    with tf.name_scope('word_attention'):
        # average attention
        s = att_avg(w,doc_len)
        # varible attention
        # sh2 = FLAGS.n_hidden * 2
        # w1 = get_weight_varible('word_att_w1', [sh2, sh2])
        # b1 = get_weight_varible('word_att_b1', [sh2])
        # w2 = get_weight_varible('word_att_w2', [sh2, 1])
        # s = att_var(w,doc_len,w1,b1,w2)
    # s shape:        [-1, FLAGS.n_hidden*2]

    with tf.name_scope('softmax'):
        outputs = tf.nn.dropout(s, keep_prob=keep_prob2)
        w = get_weight_varible('softmax_w', [2 * FLAGS.n_hidden, FLAGS.n_class])
        b = get_weight_varible('softmax_b', [FLAGS.n_class])
        pred = tf.matmul(outputs, w) + b
        pred = tf.nn.softmax(pred)
    reg = tf.nn.l2_loss(w) + tf.nn.l2_loss(b)
    return pred, reg

def print_model_info():
    ans1=0
    d=FLAGS.embedding_dim
    for h in dense_h:
        ans1+=(d+h)*h*8
        d += h*2
    h = FLAGS.n_hidden
    ans2=(d+h)*h*8
    M=1000000.
    print('## input struct ##\nmax_doc_len-{}\n'.format(FLAGS.max_doc_len))
    print('## model struct ##\ndense_h-{}, n_hidden-{}, n_class-{}'.format(dense_h, FLAGS.n_hidden, FLAGS.n_class))
    print("Model Parameters: dense {}M, Bilstm {}M, all {}M\n".format(ans1/M,ans2/M,(ans1+ans2)/M))

def read_MR_Data(word_id_mapping):
    print('\n\n>>>>>>>>>>>>>>>>>>>>DATA INFO:\n')
    tr_x, tr_y, tr_doc_len, te_x, te_y, te_doc_len = load_data_for_DSCNN_sen(
        FLAGS.train_file_path,
        word_id_mapping,
        FLAGS.max_doc_len,
        FLAGS.n_class,
        10,
        FLAGS.test_index
    )
    print('train docs: {}    test docs: {}    test_index-{}\n'.format(len(tr_y), len(te_y), FLAGS.test_index))
    return tr_x, tr_y, tr_doc_len,te_x, te_y, te_doc_len

def print_training_info():
    print('\n\n>>>>>>>>>>>>>>>>>>>>TRAINING INFO:\n')
    print('batch-{}, lr-{}, kb1-{}, kb2-{}, l2_reg-{}'.format(
        FLAGS.batch_size,  FLAGS.learning_rate, FLAGS.keep_prob1, FLAGS.keep_prob2, FLAGS.l2_reg))
    print('training_iter-{}, scope-{}\n'.format(FLAGS.training_iter, FLAGS.scope))

def feed_all_gpu(inp_dict, models, payload_per_gpu, train):
    for i in range(len(models)):
        start_pos = i * payload_per_gpu
        stop_pos = (i + 1) * payload_per_gpu
        if i==FLAGS.num_gpu-1 and stop_pos<len(train['x']):
            stop_pos=len(train['x'])

        x,doc_len,kp1,kp2,y = models[i][:5]
        inp_dict[x] = train['x'][start_pos:stop_pos]
        inp_dict[doc_len] = train['doc_len'][start_pos:stop_pos]
        inp_dict[kp1] = train['keep_prob1']
        inp_dict[kp2] = train['keep_prob2']
        inp_dict[y] = train['y'][start_pos:stop_pos]
    return inp_dict

def get_gpu_batch_data(x, doc_len, keep_prob1, keep_prob2, y, batch_size, test=False):
    for index in batch_index(len(y), batch_size, test):
        feed_dict = {
            'x': x[index],
            'doc_len': doc_len[index],
            'keep_prob1': keep_prob1,
            'keep_prob2': keep_prob2,
            'y': y[index],
        }
        yield feed_dict, len(index)

def multi_gpu():
    tf.reset_default_graph()
    with tf.device('/cpu:0'):
        # Model Code Block
        word_id_mapping, word_embedding = tf_load_w2v(FLAGS.w2v_file, FLAGS.embedding_dim, FLAGS.embedding_type)
        print_model_info()
        print('build model...')
        print('build model on gpu tower...')
        models = []
        opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        for gpu_id in range(FLAGS.num_gpu):
            with tf.device('/gpu:%d' % gpu_id):
                print('tower:%d...'% gpu_id)
                with tf.name_scope('tower_%d' % gpu_id):
                    with tf.variable_scope('cpu_variables', reuse=gpu_id>0):
                        x = tf.placeholder(tf.int32, [None, FLAGS.max_doc_len])
                        doc_len = tf.placeholder(tf.int32, [None])
                        keep_prob1 = tf.placeholder(tf.float32)
                        keep_prob2 = tf.placeholder(tf.float32)
                        y = tf.placeholder(tf.float32, [None, FLAGS.n_class])

                        pred, reg = build_model(word_embedding,x,doc_len,keep_prob1,keep_prob2)
                        loss = - tf.reduce_mean(y * tf.log(pred)) + reg * FLAGS.l2_reg
                        grads = opt.compute_gradients(loss)
                        models.append((x,doc_len,keep_prob1,keep_prob2,y,pred,loss,grads))
        print('build model on gpu tower done.')
        print('reduce model on cpu...')
        tower_y, tower_preds, tower_losses, tower_grads = zip(*models)[-4:]
        all_y = tf.concat(tower_y, 0) 
        all_pred = tf.concat(tower_preds, 0)
        aver_loss_op = tf.reduce_mean(tower_losses)
        apply_gradients_op = opt.apply_gradients(average_gradients(tower_grads))

        true_y_op = tf.argmax(all_y, 1)
        pred_y_op = tf.argmax(all_pred, 1)
        correct_pred = tf.equal(tf.argmax(all_y, 1), tf.argmax(all_pred, 1))
        acc_op = tf.reduce_mean(tf.cast(correct_pred, 'float'))
        acc_num_op = tf.reduce_sum(tf.cast(correct_pred, 'float'))
        print('reduce model on cpu done.')
        # Data Code Block
        tr_x, tr_y, tr_doc_len, te_x, te_y, te_doc_len = read_MR_Data(word_id_mapping)
        # Training Code Block
        print_training_info()
        tf_config = tf.ConfigProto()  
        tf_config.gpu_options.allow_growth = True
        with tf.Session(config=tf_config) as sess:
        # with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            max_acc, step, test_step = 0., -1, len(te_y)/(FLAGS.batch_size*FLAGS.num_gpu)
            for i in xrange(FLAGS.training_iter):
                start_time = time.time()
                for train, _ in get_gpu_batch_data(tr_x, tr_doc_len, FLAGS.keep_prob1, FLAGS.keep_prob2, tr_y, FLAGS.batch_size*FLAGS.num_gpu):
                    inp_dict = feed_all_gpu({}, models, FLAGS.batch_size, train)
                    _, loss, acc = sess.run([apply_gradients_op, aver_loss_op, acc_op], feed_dict=inp_dict)
                    step = step + 1
                    
                    if step % test_step == 0:
                        acc_cnt=0.
                        for train, num in get_gpu_batch_data(te_x, te_doc_len, 1.0, 1.0, te_y, FLAGS.batch_size*FLAGS.num_gpu, test=True):
                            if num<FLAGS.num_gpu:
                                print('num < FLAGS.num_gpu')
                                break
                            inp_dict = feed_all_gpu({}, models, num/FLAGS.num_gpu, train)
                            acc_num = sess.run(acc_num_op, feed_dict=inp_dict)
                            acc_cnt = acc_cnt + acc_num
                        test_acc = acc_cnt/len(te_y)
                        if test_acc > max_acc:
                            max_acc = test_acc
                        print 'Iter {}: mini-batch loss={:.6f}, acc={:.6f} test_acc={:.6f}'.format(step, loss, acc, test_acc)
                    else :
                        print 'Iter {}: mini-batch loss={:.6f}, acc={:.6f}'.format(step, loss, acc)
                
                print '----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime()))
                print 'epoch {}: max_test_acc {}  cost_time {}s  test_acc {:.6f} '.format(i, max_acc, time.time()-start_time, test_acc)
                print_training_info()
        print '----------{}----------'.format(time.strftime("%Y-%m-%d %X", time.localtime()))
        print 'Optimization Finished!'
        return max_acc


def main(_):
    if FLAGS.log_to_file:
        sys.stdout = open(FLAGS.log_file_name, 'w')
    results = []
    for i in range(10):
        FLAGS.test_index = i
        FLAGS.scope='MR_{}'.format(i)
        results.append(multi_gpu())
    print 'results: {}\navg: {}'.format(results, sum(results)/len(results))

if __name__ == '__main__':
    tf.app.run()