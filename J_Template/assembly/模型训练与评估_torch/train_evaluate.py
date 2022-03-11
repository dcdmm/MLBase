import torch


class Train_Evaluate:
    """
    pytorch模型训练与评估组件
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

    def train(self, train_loader, epoch, verbose, metric):
        """
        模型训练组件

        Parameters
        ---------
        train_loader : torch dataloader.DataLoader
            训练数据集
        epoch: int
            当前为第几轮训练
        verbose: int
            每多少个批次打印一次中间结果
        metric: func
            其他评估指标
        """
        self.model.train()  # Sets the module in training mode
        train_loader_len = len(train_loader.dataset)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # 反向传播固定格式
            self.optimizer.zero_grad()  # 梯度清零
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()  # 梯度计算
            self.optimizer.step()  # 执行一次优化步骤

            # 每verbose次输出一次中间结果
            # 输出第一个批次的结果
            # 输出所有数据的结果
            if (batch_idx + 1) % verbose == 0 or batch_idx == 0 or batch_idx == (len(train_loader) - 1):
                trained_num = (batch_idx + 1) * train_loader.batch_size
                if trained_num >= train_loader_len:
                    trained_num = train_loader_len  # 当drop_last=False时,trained_num可能会大于train_loader_len
                if metric is not None:  # 带额外评估指标的输出
                    metric_name = metric.__name__
                    metric_result = metric(output, target)
                    if batch_idx == 0:
                        print('Train Epoch: {:<2} [{:<5}/{} ({:<3.0f}%)]\tLoss: {:.6f}\t{}: {:.6f}'.
                              format(epoch, 0,
                                     train_loader_len,
                                     0,
                                     loss.item(),
                                     metric_name,
                                     metric_result))
                    else:
                        print('Train Epoch: {:<2} [{:<5}/{} ({:<3.0f}%)]\tLoss: {:.6f}\t{}: {:6f}'.
                              format(epoch,
                                     trained_num,
                                     train_loader_len,
                                     (100. * (batch_idx + 1)) / len(train_loader),
                                     loss.item(), metric_name, metric_result))
                else:
                    if batch_idx == 0:
                        print('Train Epoch: {:<2} [{:<5}/{} ({:<3.0f}%)]\tLoss: {:.6f}'.
                              format(epoch,
                                     0,
                                     train_loader_len,
                                     0,
                                     loss.item()))
                    else:
                        print('Train Epoch: {:<2} [{:<5}/{} ({:<3.0f}%)]\tLoss: {:.6f}'.
                              format(epoch,
                                     trained_num,
                                     train_loader_len,
                                     (100. * (batch_idx + 1)) / len(train_loader),
                                     loss.item()))

        print('-' * 100)

    def eval(self, data_loader, metric):
        """
        模型验证与测试

        Parameters
        ---------
        data_loader : torch dataloader.DataLoader
            验证数据集
        metric : func
            其他评估指标
        """
        self.model.eval()  # Sets the module in evaluation mode.
        predict_list = []
        y_true_list = []
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.device)
                predict = self.model(data)
                predict_list.append(predict)
                y_true_list.extend(target.tolist())
        predict_all = torch.cat(predict_list, dim=0)  # 合并每个批次的预测值
        y_true = torch.tensor(y_true_list).to(self.device)
        if metric is not None:
            return self.criterion(predict_all, y_true).item(), metric(predict_all, y_true).item()
        else:
            return self.criterion(predict_all, y_true).item()

    def train_eval(self, train_loader, valid_loader=None, metric=None, verbose=20):
        """
        模型训练和评估
        Parameters
        ---------
        train_loader : torch dataloader.DataLoader
            训练数据集
        valid_loader : torch dataloader.DataLoader
            验证数据集
        metric : func
            其他评估指标
        verbose : int
            每多少个批次打印一次中间结果

        Returns
        -------
        history : dict
            不同轮次下训练数据集和验证数据集的损失值(和其他评估指标)
        """
        history = {'train_loss': [], 'val_loss': []}
        for epoch in range(self.epochs):
            self.train(train_loader, epoch=epoch, verbose=verbose, metric=metric)
            if metric is not None:
                history['train_loss'].append(self.eval(train_loader, metric=metric)[0])
                history['train_' + metric.__name__].append(self.eval(train_loader, metric=metric)[1])
                if valid_loader is not None:
                    history['val_loss'].append(self.eval(valid_loader, metric=metric)[0])
                    history['val_' + metric.__name__].append(self.eval(valid_loader, metric=metric)[1])
            else:
                history['train_loss'].append(self.eval(train_loader, metric=metric))
                if valid_loader is not None:
                    history['val_loss'].append(self.eval(valid_loader, metric=metric))
        if not history['val_loss']:
            history.pop('val_loss')
        return history
