import numpy as np
from sklearn.metrics import roc_auc_score
from collections import OrderedDict
import scipy.sparse as sp


def generate_beta():
    # The interval varies from datasets and could be determined by the convergence of GraphSage
    aa = 0.
    xx_range = []
    while aa <= 1.01:
        xx_range.append(aa)
        xx_range.append(aa)
        aa = aa + 0.02
    return xx_range

def recall(rank, ground_truth, N):
    return len(set(rank[:N]) & set(ground_truth)) / float(len(set(ground_truth)))
from sklearn import metrics

def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)


def average_precision(r,cut):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    Returns:
        Average precision
    """
    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(cut) if r[k]]
    if not out:
        return 0.
    return np.sum(out)/float(min(cut, np.sum(r)))

def construct_bit_buckets(item_sparse_code):
    item_len, dim = item_sparse_code.shape

    interest_group = {i: [] for i in range(dim)}

    for i in range(dim):
        sparse_code_col = np.squeeze(item_sparse_code[:, i].toarray())
        group_index = np.where(sparse_code_col == 1)
        group_index = list(group_index[0])

        interest_group[i] = group_index

    return interest_group

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def recall_at_k(r, k, all_pos_num):
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num


def hit_at_k(r, k):
    r = np.array(r)[:k]
    if np.sum(r) > 0:
        return 1.
    else:
        return 0.

def F1(pre, rec):
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.

def auc(ground_truth, prediction):
    try:
        res = roc_auc_score(y_true=ground_truth, y_score=prediction)
    except Exception:
        res = 0.5
    return res


def predict_matrix(out_test):
    pred_matrix = out_test[0]
    train_label = out_test[1]
    train_label = 1 - train_label
    pred_matrix = np.multiply(pred_matrix, train_label)
    return pred_matrix


def evaluate_one_user(topk, ranking_user, pos_test_items):
    recall_, ndcg_, precision, f1 = [], [], [], []
    r = []
    max_k = max(topk)
    ranking_user = ranking_user[0:max_k]
    for i in range(len(ranking_user)):
        item_ = ranking_user[i]
        if item_ in pos_test_items:
            r.append(1)
        else:
            r.append(0)
    for K in topk:
        recall_tem = recall_at_k(r, K, len(pos_test_items))
        recall_.append(recall_tem)
        ndcg_.append(ndcg_at_k(r, K))
        precision_tem = precision_at_k(r, K)
        precision.append(precision_tem)

        f1.append(F1(precision_tem, recall_tem))

    return recall_, ndcg_, precision, f1


def parse_config(flags, verbose=False):
    config = OrderedDict(sorted(flags.__flags.items()))
    # if flags.python_version == 3:
    for k, v in config.items():
        config[k] = v.value
    if verbose:
        print(">>>>> params setting: ")
        for k, v in config.items():
            print('{}:'.format(k), config[k])


def evaluate_embedding_parellel_faiss(embed_user, item_embedding, user_recall, pos_test_items):
    topk = [20, 50, 100, 150, 200]
    item_embedding = item_embedding[user_recall]
    pred_user = embed_user.dot(item_embedding.transpose()).toarray()[0]
    pred_user = sigmoid(pred_user)
    ranking_user_ = list(np.argsort(- pred_user))
    ranking_user = [user_recall[index_] for index_ in ranking_user_]

    recall_, ndcg, precision, f1 = evaluate_one_user(topk, ranking_user, pos_test_items)
    return recall_, ndcg, precision, f1


def evaluate_embedding_parellel_faiss_final(embed_user, item_embedding, user_recall, pos_test_items):
    topk = [50, 100]
    item_embedding = item_embedding[user_recall]
    pred_user = embed_user.dot(item_embedding.transpose()).toarray()[0]
    pred_user = sigmoid(pred_user)
    ranking_user_ = list(np.argsort(- pred_user))
    ranking_user = [user_recall[index_] for index_ in ranking_user_]

    recall_, ndcg, precision, f1 = evaluate_one_user(topk, ranking_user, pos_test_items)
    return recall_, ndcg, precision, f1
