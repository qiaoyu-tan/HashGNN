from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import time
from metrics_rs import *
import multiprocessing as mp
from graph import EdgeTable
import faiss
# Set random seed
seed = 12345
np.random.seed(seed)
tf.set_random_seed(seed)

GPU_MEM_FRACTION = 0.8

# Settings

def _evaluate_embedding_hash_faiss(user_emb_csr, item_emb_csr, user_emb_spar_csr, item_emb_spar_csr,
                                   train_dict, test_dict):
    k = 500
    if mp.cpu_count() > 4:
        cores = 10
    else:
        cores = 1
    pool = mp.Pool(cores)
    t1 = time.time()
    index = faiss.IndexFlatIP(item_emb_spar_csr.shape[1])
    index.add(item_emb_spar_csr.toarray().astype('float32'))
    _, i_recall = index.search(user_emb_spar_csr.toarray().astype('float32'), k)
    print('time for knn construct={} k={}'.format(time.time()-t1, k))

    results_ = [pool.apply(evaluate_embedding_parellel_faiss_final, args=(user_emb_csr[user_], item_emb_csr, i_recall[user_],
                                                              test_dict[user_])) for user_ in test_dict.keys()]
    pool.close()
    recall_ = [user_result[0] for user_result in results_]
    ndcg = [user_result[1] for user_result in results_]
    precision = [user_result[2] for user_result in results_]
    f1 = [user_result[3] for user_result in results_]

    recall_ = np.array(recall_)
    ndcg = np.array(ndcg)
    precision = np.array(precision)
    f1 = np.array(f1)

    recall_ = np.mean(recall_, axis=0)
    precision = np.mean(precision, axis=0)
    f1 = np.mean(f1, axis=0)
    ndcg = np.mean(ndcg, axis=0)
    # auc = np.mean(auc)

    return recall_, ndcg, precision, f1

