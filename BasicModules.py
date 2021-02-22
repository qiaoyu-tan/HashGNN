import tensorflow as tf


def get_metric(tag, pre_label, pre_probability):
    def auc_pr(predictions=None, labels=None):
        return tf.contrib.metrics.streaming_auc(predictions=predictions, labels=labels, curve="PR")

    def auc_roc(predictions=None, labels=None):
        return tf.contrib.metrics.streaming_auc(predictions=predictions, labels=labels, curve="ROC")

    def precision_thr(predictions=None, labels=None):
        return tf.contrib.metrics.streaming_precision_at_thresholds(predictions=predictions, labels=labels,
                                                                    thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                                                                                0.9, 1.0])

    def recall_thr(predictions=None, labels=None):
        return tf.contrib.metrics.streaming_recall_at_thresholds(predictions=predictions, labels=labels,
                                                                 thresholds=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                                                                             0.9, 1.0])

    metrics = {
        tag + "_accuracy":
            tf.contrib.learn.MetricSpec(
                metric_fn=tf.metrics.accuracy,
                prediction_key=pre_label),
        tag + "_precision":
            tf.contrib.learn.MetricSpec(
                metric_fn=tf.metrics.precision,
                prediction_key=pre_label),
        tag + "_recall":
            tf.contrib.learn.MetricSpec(
                metric_fn=tf.metrics.recall,
                prediction_key=pre_label),
        tag + "_precision_thr":
            tf.contrib.learn.MetricSpec(
                metric_fn=precision_thr,
                prediction_key=pre_probability),
        tag + "_recall_thr":
            tf.contrib.learn.MetricSpec(
                metric_fn=recall_thr,
                prediction_key=pre_probability),
        tag + "_auc_roc":
            tf.contrib.learn.MetricSpec(
                metric_fn=auc_roc,
                prediction_key=pre_probability),
        tag + "_auc_pr":
            tf.contrib.learn.MetricSpec(
                metric_fn=auc_pr,
                prediction_key=pre_probability)
    }

    return metrics


def inner_evaluation(probs, true_label, pos_prob, tag):
    def predict(probs):
        # softmax = tf.nn.softmax(logits, name="softmax")
        label = tf.argmax(probs, 1, name="label")
        return probs, label

    with tf.variable_scope(tag + "_evaluation"):
        softmax, predict_label = predict(probs)
        true_label_onehot = tf.one_hot(indices=true_label, depth=2)

        acc, update_op_acc = tf.metrics.accuracy(
            true_label, predict_label)
        precision, update_op_precision = tf.metrics.precision(
            true_label, predict_label)
        recall, update_op_recall = tf.metrics.recall(
            true_label, predict_label)
        precision_thr, update_op_precision_thr = tf.metrics.precision_at_thresholds(
            softmax, true_label_onehot, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        recall_thr, update_op_recall_thr = tf.metrics.recall_at_thresholds(
            softmax, true_label_onehot, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        auc_roc, update_op_auc_roc = tf.metrics.auc(
            true_label, pos_prob, curve='ROC')
        auc_pr, update_op_auc_pr = tf.metrics.auc(
            true_label, pos_prob, curve='PR')
        confusion_matrix = tf.contrib.metrics.confusion_matrix(true_label, predict_label, 2)

    return acc, update_op_acc, precision, update_op_precision, recall, update_op_recall, \
           precision_thr, update_op_precision_thr, recall_thr, update_op_recall_thr, \
           auc_roc, update_op_auc_roc, auc_pr, update_op_auc_pr, confusion_matrix
