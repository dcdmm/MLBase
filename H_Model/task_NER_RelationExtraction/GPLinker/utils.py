import torch


def sparse_multilabel_categorical_crossentropy(y_true, y_pred, mask_zero=True):
    """稀疏多标签交叉熵损失的torch实现"""
    # y_true.shape=[batch_size, ?, longest sequence, 2]
    # y_pred.shape=[batch_size, ?, seq_len, seq_len]

    shape = y_pred.shape
    # y_true.shape=[batch_size, ? longest sequence]
    # 解析:省略前两个维度batch_size, ?,则y_true'[0] = y_true[0, 0] * seq_len + y_true[0, 1];y_true'[1] = y_true[1, 0] * seq_len + y_true[1, 1]
    y_true = y_true[..., 0] * shape[2] + y_true[..., 1]  # shape[2]=seq_len
    # y_pred.shape=[batch_size, ?, seq_len * seq_len]
    y_pred = y_pred.reshape(shape[0], -1, torch.prod(torch.tensor(shape[2:])).item())

    # zeros.shape=[batch_size, ?, 1]
    zeros = torch.zeros_like(y_pred[..., :1])
    # infs.shape=[batch_size, ?, 1]
    infs = zeros + 1e12
    # y_pred.shape=[batch_size, ?, seq_len * seq_len + 1]
    y_pred = torch.cat([y_pred, zeros], dim=-1)
    if mask_zero:
        y_pred = torch.cat([infs, y_pred[..., 1:]], dim=-1)  # y_pred.shape=[batch_size, ?, 1 + seq_len * seq_len + 1]
    y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
    y_pos_1 = torch.cat([y_pos_2, zeros], dim=-1)
    if mask_zero:
        y_pred = torch.cat([-infs, y_pred[..., 1:]], dim=-1)
        y_pos_2 = torch.gather(y_pred, index=y_true, dim=-1)
    pos_loss = torch.logsumexp(-y_pos_1, dim=-1)
    all_loss = torch.logsumexp(y_pred, dim=-1)
    aux_loss = torch.logsumexp(y_pos_2, dim=-1) - all_loss
    aux_loss = torch.clip(1 - torch.exp(aux_loss), 1e-10, 1)
    neg_loss = all_loss + torch.log(aux_loss)

    loss = torch.mean(torch.sum(pos_loss + neg_loss))
    return loss


class MetricsCalculator_CMeIE:
    """查准率、查全率、F1 score"""

    def __init__(self):
        self.TP = 0
        self.TP_FP = 0
        self.TP_FN = 0

    def calc_confusion_matrix(self, y_pred, y_true):
        for b in range(len(y_pred)):
            b_pred = y_pred[b]
            b_true = y_true[b]
            # 集合元素形式为:(subject, predicate, object)
            R = set([(spo[0], spo[1], spo[2]) for spo in b_pred])
            T = set([(spo[0], spo[3] + "_" + spo[1] + "_" + spo[-1], spo[2]) for spo in b_true])
            self.TP += len(R & T)
            self.TP_FP += len(R)
            self.TP_FN += len(T)

    @property
    def precision(self):
        return 0 if self.TP_FP == 0 else self.TP / self.TP_FP  # 查准率

    @property
    def recall(self):
        return 0 if self.TP_FP == 0 else self.TP / self.TP_FN  # 查全率

    @property
    def f1(self):
        return 0 if (self.TP_FP + self.TP_FN) == 0 else 2 * self.TP / (self.TP_FP + self.TP_FN)  # f1 score


class MetricsCalculator_bdci:
    """查准率、查全率、F1 score"""

    def __init__(self):
        self.TP = 0
        self.TP_FP = 0
        self.TP_FN = 0

    def calc_confusion_matrix(self, y_pred, y_true):
        for b in range(len(y_pred)):
            b_pred = y_pred[b]
            b_true = y_true[b]
            # 集合元素形式为:(subject, object, predicate)
            R = set([(spo[0], spo[2], spo[-1]) for spo in b_pred])
            T = set([(spo[0], spo[2], spo[-1]) for spo in b_true])
            self.TP += len(R & T)
            self.TP_FP += len(R)
            self.TP_FN += len(T)

    @property
    def precision(self):
        return 0 if self.TP_FP == 0 else self.TP / self.TP_FP  # 查准率

    @property
    def recall(self):
        return 0 if self.TP_FP == 0 else self.TP / self.TP_FN  # 查全率

    @property
    def f1(self):
        return 0 if (self.TP_FP + self.TP_FN) == 0 else 2 * self.TP / (self.TP_FP + self.TP_FN)  # f1 score
