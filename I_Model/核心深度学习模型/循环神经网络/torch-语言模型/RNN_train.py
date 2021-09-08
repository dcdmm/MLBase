import torch.nn as nn
import torch
import torch.optim as optim


def repackage_hidden(h):
    """不共享内存和脱离计算图"""
    if isinstance(h, torch.Tensor):  # RNN,GUR情况下
        return h.clone().detach()
    else:
        return tuple(repackage_hidden(v) for v in h)  # LSMT情况下


class RNN_train:
    def __init__(self, model, optimizer, criterion, device, batch_size):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.batch_size = batch_size

    def init_hidden(self):
        """初始化hx"""
        init_zeros = torch.zeros((self.model.num_layers, self.batch_size, self.model.hidden_size))
        init_zeros = init_zeros.to(self.device)  # 这个地方也要执行此语句
        if self.model.rnn_type == 'LSTM':
            return init_zeros, init_zeros
        else:
            return init_zeros

    def repeat(self, batch, hidden):
        """训练和验证的重复代码"""
        text, target = batch.text, batch.target
        text, target = text.to(self.device), target.to(self.device)

        output, hidden = self.model(text, hidden)
        hidden = repackage_hidden(hidden)

        output = output.view(-1, output.shape[-1])
        target = target.view(-1)
        loss = self.criterion(output, target)  # 注意pytorch交叉熵损失函数的格式
        return loss, hidden

    def train(self, train_iterator, clip):
        """
        在训练过程中,使用的输入和输出是同一个句子
        但是输入的句子和输出的句子之间相差一个单词,既在循环神经网络的计算中,使用上一个单词预测下一个单词
        从而计算相应的损失函数
        """
        self.model.train()  # 进入训练模式
        epoch_loss = 0
        hidden = self.init_hidden()

        for i, batch in enumerate(train_iterator):
            self.optimizer.zero_grad()
            loss, hidden = self.repeat(batch, hidden)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), clip)  # 梯度裁剪
            self.optimizer.step()
            epoch_loss += loss.item()

            if i % 1000 == 0:
                print('train loss:', loss)
        return epoch_loss / len(train_iterator)

    def evaluate(self, val_iterator):
        """验证过程"""
        self.model.eval()
        epoch_loss = 0
        hidden = self.init_hidden()

        with torch.no_grad():  # 上下文管理器,脱离计算图
            for j, batch in enumerate(val_iterator):
                loss, hidden = self.repeat(batch, hidden)
                epoch_loss += loss.item()

                scheduler_lr = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.1, mode='min',
                                                                    patience=5, min_lr=0.000001, eps=1e-12)  # 设置学习率衰减
                scheduler_lr.step(loss)
                if j % 1000 == 0 and val_iterator != 'test_iter':  # 每1000次输出一次验证数据集的损失
                    print('valid loss:', loss)
        self.model.train()
        return epoch_loss / len(val_iterator)
