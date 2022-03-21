import numpy as np
from Optimizer import SGD


class Trainer:
    """训练神经网络"""

    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100, optimizer_param=None, verbose=True):
        if optimizer_param is None:
            optimizer_param = {'lr': 0.01}
        self.network = network
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.train_size = x_train.shape[0]  # 样本大小
        self.batch_size = mini_batch_size
        self.optimizer = SGD(**optimizer_param)
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []
        self.current_iter = 0
        self.verbose = verbose  # 是否打印训练信息
        self.epochs = epochs  # 训练批次
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)

    def train_step(self):
        batch_mask = np.random.choice(self.train_size, self.batch_size)  # 随机选择数据进行训练
        x_batch = self.x_train[batch_mask]
        t_batch = self.t_train[batch_mask]

        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)  # 更新梯度信息

        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)
        if self.verbose:
            print("current_iter:" + str(self.current_iter), ', train loss:' + str(loss))

        if self.current_iter % self.iter_per_epoch == 0:  # 每self.iter_per_epoch次保存一次训练信息
            x_train_sample, t_train_sample = self.x_train[:3000], self.t_train[:3000]  # 使用一半的数据对每轮神经网络进行验证
            x_test_sample, t_test_sample = self.x_test[:500], self.t_test[:500]
            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)

            if self.verbose:
                print("======current_iter:" + str(self.current_iter) + ", train acc:" + str(
                    train_acc) + ", dir_example acc:" + str(test_acc) + "=======")
        self.current_iter += 1

    def train(self):
        for i in range(self.max_iter):
            self.train_step()

        test_acc = self.network.accuracy(self.x_test, self.t_test)

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("dir_example acc:" + str(test_acc))  # 最终测试数据集的精度
