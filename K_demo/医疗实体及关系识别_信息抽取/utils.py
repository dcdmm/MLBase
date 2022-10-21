import torch
import numpy as np


def multilabel_categorical_crossentropy(y_pred, y_true):
    """多标签分类的交叉熵.原理见https://kexue.fm/archives/7359"""
    # 必须确保y_pred=y_true
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = y_pred - (1 - y_true) * 1e12  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()


def loss_fun(y_true, y_pred):
    """
    y_true.shape=[batch_size, ent_type_size, seq_len, seq_len]
    y_pred.shape=[batch_size, ent_type_size, seq_len, seq_len]
    """
    batch_size, ent_type_size = y_pred.shape[:2]
    # y_true.shape=[batch_size * ent_type_size, seq_len, seq_len]
    y_true = y_true.reshape(batch_size * ent_type_size, -1)
    y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
    loss = multilabel_categorical_crossentropy(y_true, y_pred)
    return loss


class MetricsCalculator:
    """查准率、查全率、F1 score"""

    def __init__(self):
        self.TP, self.TP_FP, self.TP_FN = 0, 0, 0

    def calc_confusion_matrix_ner(self, y_pred, y_true):
        y_pred = y_pred.data.cpu().numpy()
        y_true = y_true.data.cpu().numpy()
        pred, true = [], []
        for b, l, start, end in zip(*np.where(y_pred > 0.0)):  # 阈值(threshold)设置为0.0
            pred.append((b, l, start, end))
        for b, l, start, end in zip(*np.where(y_true > 0.0)):
            true.append((b, l, start, end))

        R = set(pred)
        T = set(true)
        self.TP += len(R & T)
        self.TP_FP += len(R)
        self.TP_FN += len(T)

    def calc_confusion_matrix_rel(self, y_pred, y_true):
        y_pred = y_pred.data.cpu().numpy() > 0
        y_pred = y_pred.astype(np.float32)
        y_true = y_true.data.cpu().numpy()

        self.TP += np.sum(y_true * y_pred)
        self.TP_FP += np.sum(y_pred)
        self.TP_FN += np.sum(y_true)

    @property
    def precision(self):
        return 0 if self.TP_FP == 0 else self.TP / self.TP_FP  # 查准率

    @property
    def recall(self):
        return 0 if self.TP_FP == 0 else self.TP / self.TP_FN  # 查全率

    @property
    def f1(self):
        return 0 if (self.TP_FP + self.TP_FN) == 0 else 2 * self.TP / (self.TP_FP + self.TP_FN)  # f1 score
