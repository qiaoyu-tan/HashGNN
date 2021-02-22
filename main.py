from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from input_graph_feed_beta import GraphInput, data_load
import time
from model import HashGNN_model
from metrics_rs import *
import datetime
from evaluate_recall import _evaluate_embedding_hash_faiss
# Set random seed
seed = 12345
np.random.seed(seed)
tf.set_random_seed(seed)

GPU_MEM_FRACTION = 0.8

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
# core params..
flags.DEFINE_string('checkpointDir', '', '')
flags.DEFINE_string('outputDir', 'hashgnn', '')
flags.DEFINE_string('version', '', '')
flags.DEFINE_integer('slice_parts', 32, 'final dim')
flags.DEFINE_boolean('trace', False, 'whether trace.')
flags.DEFINE_string('hdfs_path', '', '')
flags.DEFINE_string('job_name', '', 'job name')
flags.DEFINE_boolean('write_nbrs', False, 'if true, u_emb_table needs 3columns')
flags.DEFINE_boolean('use_protobuf_input', False, '')
flags.DEFINE_string('dataset', 'ml-1m', 'model names. See README for possible values.')
flags.DEFINE_string('encoding_schema', 'u-i-u', 'user->item->user.')
flags.DEFINE_string('activation', 'leaky_relu', 'activation function.')
flags.DEFINE_float('learning_rate', 0.001, 'initial learning rate.')
flags.DEFINE_float('lambda_', 0.1, 'initial learning rate.')
flags.DEFINE_string('learning_algo', 'adam', 'adam or sgd')
flags.DEFINE_string('u_neighs_num', '10,5', 'adam or sgd')
flags.DEFINE_string('i_neighs_num', '10', 'adam or sgd')
flags.DEFINE_string('u_embed', '68,68,68', 'adam or sgd')
flags.DEFINE_string('i_embed', '68,68,68', 'adam or sgd')

flags.DEFINE_integer('epochs', 20, 'number of epochs to train.')
flags.DEFINE_float('dropout', 0.0, 'dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('neigh_hop1', 10, 'number of neighbors in layer 1')
flags.DEFINE_integer('neigh_hop2', 5, 'number of neighbors in layer 2')
flags.DEFINE_integer('neg_num', 10, 'negative number per sample')
flags.DEFINE_integer('id_dim', 120, 'negative number per sample')
flags.DEFINE_integer('hash_size', 25, 'output ')
flags.DEFINE_integer('sparse_k', 11, 'output ')
flags.DEFINE_boolean('hash_sparse', True, 'whether to save embeddings for all nodes after training')

flags.DEFINE_integer('batch_size', 256, 'minibatch size.')
flags.DEFINE_integer('out_dense', 512, 'first_dense ')
flags.DEFINE_integer('final_dim', 32, 'output ')

# logging, saving, validation settings etc.
flags.DEFINE_boolean('save_embeddings', True, 'whether to save embeddings for all nodes after training')
flags.DEFINE_boolean('use_input_bn', False, 'whether to save embeddings for all nodes after training')
flags.DEFINE_string('base_log_dir', '.', 'base directory for logging and saving embeddings')
flags.DEFINE_integer('print_every', 1000, "How often to print training info.")
flags.DEFINE_integer('validate_iter', 100, "how many batches to run validation.")
flags.DEFINE_integer('validate_local_every', 1000, "how often to run a validation minibatch.")
flags.DEFINE_integer('gpu', 0, "which gpu to use.")
flags.DEFINE_integer('summary_every', 10000, "How often to print training info.")
flags.DEFINE_integer('save_embedding_every', 4000, "How often to print training info.")
flags.DEFINE_integer('save_lines', -1, "How many lines to write to table.")
flags.DEFINE_integer('max_total_steps', -1, "Maximum total number of iterations")

os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

def file_check():
    models = FLAGS.outputDir
    # models = FLAGS.outputDir + str(FLAGS.lambda_) # for grid search purpose
    out_dir = models
    models = FLAGS.outputDir
    # models = FLAGS.outputDir + str(FLAGS.lambda_) # for grid search purpose
    data_name = FLAGS.dataset
    if not os.path.isdir(os.path.join(os.getcwd(), models)):
        os.mkdir(os.path.join(os.getcwd(), models))

    if data_name == 'ali':
        pass
    else:
        model_file = os.path.join(os.getcwd(), models + '/' + data_name)
        model_file_embedding = os.path.join(os.getcwd(), models + '/' + data_name + '/embedding')
        out_dir = models + '/' + data_name
        if not os.path.isdir(model_file):
            os.mkdir(model_file)
            os.mkdir(model_file_embedding)

    FLAGS.outputDir = out_dir
    if data_name == 'ml-10m':
        FLAGS.final_dim = 32
    if data_name == 'gowalla':
        FLAGS.final_dim = 32
    if data_name == 'ml-1m':
        FLAGS.final_dim = 32


def train_and_eval(output_dim=12):
    file_check()
    parse_config(FLAGS, verbose=True)
    input_data = GraphInput(flag=FLAGS)
    input_data.init_server()
    input_data.init_u_i_iter()

    global_step = tf.train.get_or_create_global_step()
    model = HashGNN_model(
        FLAGS, global_step,
        input_data.u_categoricals, input_data.u_continuous,
        input_data.i_categoricals, input_data.i_continuous,
        graph_input=input_data.features,
        u_id_encode=(input_data.user_len, FLAGS.id_dim),
        i_id_encode=(input_data.item_len, FLAGS.id_dim),
        mode='train')

    print('All global variables:')
    for v in tf.global_variables():
        if v not in tf.trainable_variables():
            print('\t', v)
        else:
            print('\t', v, 'trainable')

    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    xx_range = generate_beta()

    summary_hook = tf.train.SummarySaverHook(
        save_steps=FLAGS.summary_every, output_dir=FLAGS.outputDir,
        summary_op=model.summary_op)

    nan_hook = tf.train.NanTensorHook(model.loss)
    hooks = [nan_hook, summary_hook]
    if FLAGS.max_total_steps != -1:
        stop_hook = tf.train.StopAtStepHook(last_step=FLAGS.max_total_steps)
        hooks.append(stop_hook)

    with tf.train.MonitoredTrainingSession(
            checkpoint_dir=FLAGS.outputDir,
            hooks=hooks) as mon_sess:
        print("start training.")
        local_step = 0
        t = time.time()
        last_print_step = 0
        epoch = 0
        it = 0
        preds = []
        labels = []
        beta_init = 0.
        while not mon_sess.should_stop():
            try:
                try:
                    beta = xx_range[epoch]
                except:
                    beta = beta_init
                t1 = time.time()
                sample_dict = input_data.feed_next_sample_train_continuous(beta)
                t2 = time.time()
            except StopIteration:
                epoch += 1
                print('end of an epoch')
                if epoch >= FLAGS.epochs:
                    print('\n start saving embedding at epoch={}'.format(epoch))
                    save_embedding(mon_sess, input_data, model, FLAGS, epoch, beta)
                    break
                else:
                    print('\n start saving embedding at epoch={}'.format(epoch))
                    save_embedding(mon_sess, input_data, model, FLAGS, epoch, beta)
                    continue

            it += 1
            outs_all = mon_sess.run([model.train_op], feed_dict=sample_dict)
            outs = outs_all[0]
            train_cost = outs[1]
            auc = outs[2]
            pos_pred = outs[3]
            label = outs[4]
            global_step = outs[-1]

            preds.append(pos_pred)
            labels.append(label)
            # Print results
            if local_step % FLAGS.print_every == 0:

                labels = np.concatenate(labels, axis=-1)
                preds = np.concatenate(preds, axis=-1)
                batch_auc = metrics.roc_auc_score(labels, preds)
                batch_recall = metrics.recall_score(labels, preds > 0.5)
                batch_precision = metrics.precision_score(labels, preds > 0.5)
                batch_f1 = metrics.f1_score(label, pos_pred > 0.5)
                labels = []
                preds = []
                print(datetime.datetime.now(),
                      "type-%d-Iter:" % 0, 'global-%04d' % global_step,
                      'local-%04d' % local_step,
                      'epoch-%04d' % epoch,
                      "train_loss=", "{:.5f}".format(train_cost),
                      "avg time =", "{:.5f}".format((time.time() - t) * 1.0 / FLAGS.print_every),
                      "global step/sec =",
                      "{:.2f}".format((global_step - last_print_step) * 1.0 / (time.time() - t)),
                      'auc =', auc,
                      'batch auc: {:.4f}'.format(batch_auc),
                      'batch f1: {:.4f}'.format(batch_f1),
                      'batch rec: {:.4f}'.format(batch_recall),
                      'batch prec: {:.4f}'.format(batch_precision)
                      )
                t = time.time()
                last_print_step = global_step
            local_step += 1

        print('learning finished')


def save_embedding(mon_sess, input_data, model, flag, epoch, beta):
    temp_user = sp.dok_matrix((input_data.user_len, flag.final_dim), dtype=np.float)
    temp_item = sp.dok_matrix((input_data.item_len, flag.final_dim), dtype=np.float)
    user_index = []
    item_index = []
    input_data.init_u_i_iter()
    t = time.time()
    while not mon_sess.should_stop():
        try:
            dicts, uids, unbrs_id = input_data.feed_next_user_continuous(beta)
            outs = mon_sess.run([model.user, model.global_step], feed_dict=dicts)
            temp_user[uids, :] = outs[0]
            user_index.extend(list(uids))
        except StopIteration:
            print('generated user embeddings time={}'.format(time.time()-t))
            break
    t = time.time()
    while not mon_sess.should_stop():
        try:
            dicts, uids, unbrs_id = input_data.feed_next_item_continuous(beta)
            outs = mon_sess.run([model.item, model.global_step], feed_dict=dicts)
            temp_item[uids, :] = outs[0]
            item_index.extend(list(uids))
        except StopIteration:
            print('generated item embeddings time={}'.format(time.time()-t))
            break
    temp_user = temp_user.tocsr()
    temp_item = temp_item.tocsr()
    source_file = FLAGS.outputDir + '/embedding/'
    source_file = os.path.join(os.getcwd(), source_file)
    user_file = os.path.join(source_file, 'user_embedding{}_{}.npz'.format(flag.final_dim, epoch))
    item_file = os.path.join(source_file, 'item_embedding{}_{}.npz'.format(flag.final_dim, epoch))
    sp.save_npz(user_file, temp_user)
    print('saved {} user embedding at epoch={}'.format(len(user_index), epoch))
    sp.save_npz(item_file, temp_item)
    print('saved {} item embedding at epoch={}'.format(len(item_index), epoch))

    # evaluate performance
    topk = [50, 100]
    dataset = FLAGS.dataset
    train_dict, test_dict = data_load(dataset)
    user_emb_csr_spar = sp.csr_matrix(np.sign(temp_user.toarray().astype('float32')))
    item_emb_csr_spar = sp.csr_matrix(np.sign(temp_item.toarray().astype('float32')))
    t1 = time.time()
    recall_, ndcg, _, _ = _evaluate_embedding_hash_faiss(temp_user, temp_item, user_emb_csr_spar,
                                                                  item_emb_csr_spar, train_dict, test_dict)
    print('testing results at time={} \n'.format(time.time() - t1))
    for k in range(len(topk)):
        print('\n epoch={} topk={} recall={} ndcg={}'.format(epoch, topk[k], recall_[k], ndcg[k]))


def main(argv=None):

    train_and_eval()

if __name__ == '__main__':
    tf.app.run()
