import torch
from collections import OrderedDict


class Trainer:
    """
    pytorch模型训练与评估组件(功能模仿`transformers.Trainer`)
    """

    def __init__(self, model, optimizer, criterion, epochs=5, device=None):
        """
        模型初始化

        Parameters
        ---------
        model :
            神经网络模型
        optimizer : torch optim
            优化器
        criterion : torch loss
            损失函数(损失值必须为标量)
        epochs : int
            训练轮数
        device : torch device(default=None)
            设备
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model = model
        self.model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs

    def train_step(self, train_loader, epoch=0, verbose=20, compute_metrics=None):
        """
        单次模型训练

        Parameters
        ---------
        train_loader : torch dataloader.DataLoader
            训练数据集;遍历结果为:(features, label)
        epoch: int
            当前为第几轮训练
        verbose: int
            每多少个批次打印一次中间结果
        compute_metrics : func
            其他评估函数(函数返回值必须为字典)
        """
        self.model.train()  # Sets the module in training mode
        train_loader_len = len(train_loader.dataset)

        # 具体任务对应修改
        for batch_idx, (text, lengths, target) in enumerate(train_loader):
            target, text, = target.to(self.device), text.to(self.device)

            # 反向传播固定格式
            self.optimizer.zero_grad()  # 梯度清零
            output = self.model(text, lengths)

            loss = self.criterion(output, target)
            loss.backward()  # 梯度计算
            self.optimizer.step()  # 执行一次优化步骤

            # 每verbose次输出一次中间结果 and 输出第一个批次的结果 and 输出所有数据时的结果
            if (batch_idx + 1) % verbose == 0 or batch_idx == 0 or batch_idx == (len(train_loader) - 1):
                trained_num = (batch_idx + 1) * train_loader.batch_size
                if trained_num >= train_loader_len:
                    trained_num = train_loader_len  # 当drop_last=False时,trained_num可能会大于train_loader_len
                if compute_metrics is not None:  # 带额外评估指标的输出
                    metric_result = OrderedDict(compute_metrics(output.detach().clone().cpu().numpy(),
                                                                target.detach().clone().cpu().numpy()))  # 字典转换为有序字典
                    metric_result_keys = list(metric_result.keys())
                    metric_result_values = list(metric_result.values())

                    # metric_result = {'acc': num1, 'f1': num2)
                    # k_v_list = {'acc', num1, 'f1', num2}
                    k_v_lst = []
                    for t_i in zip(metric_result_keys, metric_result_values):
                        k_v_lst.extend(t_i)

                    string = 'Train Epoch: {:<2} [{:<5}/{} ({:<3.0f}%)]\tLoss: {:.6f}' + '\t{}: {:.6f}' * len(
                        metric_result_keys)
                    if batch_idx == 0:
                        print(string.format(epoch, 0, train_loader_len, 0, loss.item(), *k_v_lst))
                    else:
                        print(string.format(epoch, trained_num, train_loader_len,
                                            (100. * (batch_idx + 1)) / len(train_loader), loss.item(), *k_v_lst))
                else:
                    if batch_idx == 0:
                        print('Train Epoch: {:<2} [{:<5}/{} ({:<3.0f}%)]\tLoss: {:.6f}'.
                              format(epoch, 0, train_loader_len, 0, loss.item()))
                    else:
                        print('Train Epoch: {:<2} [{:<5}/{} ({:<3.0f}%)]\tLoss: {:.6f}'.
                              format(epoch, trained_num, train_loader_len, (100. * (batch_idx + 1)) / len(train_loader),
                                     loss.item()))

        print('-' * 100)

    def predict(self, data_loader, status='Training', compute_metrics=None):
        """
        模型验证与测试

        Parameters
        ---------
        data_loader : torch dataloader.DataLoader
            数据集;训练或验证数据集(遍历结果为:(features, label));测试数据集(遍历结果为:(features, ))
        status : str
            当前进行的操作;训练(Training)、验证(Validation)、测试(Test)
        compute_metrics : func
            其他评估函数(函数返回值必须为字典)

        Returns
        -------
        predict_all : torch tensor
            模型预测结果
        metrics : dict
            模型评估结果(仅训练与验证阶段)
        """
        self.model.eval()  # Sets the module in evaluation mode.

        predict_list = []
        y_true_list = []

        # 具体任务对应修改
        with torch.no_grad():
            for data in data_loader:
                data_X = data[0].to(self.device)
                predict = self.model(data_X, data[1])
                predict_list.append(predict)
                if status != 'Test':  # 测试数据集不含标签
                    y_true_list.extend(data[2].tolist())

        predict_all = torch.cat(predict_list, dim=0)  # 合并所有批次的预测结果

        if status == 'Test':
            return predict_all  # 测试数据集只返回预测值
        else:
            y_true = torch.tensor(y_true_list).to(self.device)  # 数据集真实标签
            loss = self.criterion(predict_all, y_true).item()  # 损失值

            if compute_metrics is not None:
                # 模型其他评估函数的评估结果
                compute_metrics_dict = compute_metrics(predict_all.cpu().numpy(), y_true.cpu().numpy())
            else:
                compute_metrics_dict = {}

            if status == 'Training':
                metrics = {'Training loss': loss}
                for key_name in list(compute_metrics_dict.keys()):  # 合并损失值与模型其他评估函数的评估结果
                    new_name = 'Training ' + key_name  # 训练阶段评估指标以Training开头
                    metrics[new_name] = compute_metrics_dict.pop(key_name)
                return predict_all, metrics

            if status == 'Validation':
                metrics = {'Validation loss': loss}
                for key_name in list(compute_metrics_dict.keys()):
                    new_name = 'Validation ' + key_name  # 训练阶段评估指标以Validation开头
                    metrics[new_name] = compute_metrics_dict.pop(key_name)
                return predict_all, metrics

    def train(self, train_loader, valid_loader=None, estimate_train=True, compute_metrics=None, verbose=20):
        """
        模型训练和评估

        Parameters
        ---------
        train_loader : torch dataloader.DataLoader
            训练数据集;遍历结果为:(features, label)
        valid_loader : torch dataloader.DataLoader
            验证数据集;遍历结果为:(features, label)
        estimate_train : bool
            是否评估训练数据集
        compute_metrics : func
            其他评估函数(函数返回值必须为字典)
        verbose : int
            每多少个批次打印一次中间结果

        Returns
        -------
        history_train : dict
            训练数据集和验证数据集(若有)评估结果
        """
        history_train = {}
        history_valid = {}

        for epoch in range(self.epochs):
            self.train_step(train_loader, epoch=epoch, verbose=verbose, compute_metrics=compute_metrics)
            if estimate_train:
                train_result = OrderedDict(self.predict(train_loader, compute_metrics=compute_metrics)[1])
                if epoch == 0:
                    # 不要通过fromkeys构造字典(所有键都指向了同一个内存地址)
                    history_train = dict(zip(list(train_result.keys()), [[], [], []]))
                for key_name in list(train_result.keys()):
                    history_train[key_name].append(train_result[key_name])  # 添加每轮模型的评估结果

            if valid_loader is not None:
                valid_result = OrderedDict(
                    self.predict(valid_loader, status='Validation', compute_metrics=compute_metrics)[1])
                if epoch == 0:
                    history_valid = dict(zip(list(valid_result.keys()), [[], [], []]))
                for key_name in list(valid_result.keys()):
                    history_valid[key_name].append(valid_result[key_name])

        history_train.update(history_valid)

        return history_train
