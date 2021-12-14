import torch


class Train_Evaluate:
    """
    pytorch模型训练与评估组件具体实现
    """

    def __init__(self, model, optimizer, criterion, epochs=5, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model = model
        self.model.to(device)
        self.optimizer = optimizer  # 优化器
        self.criterion = criterion  # 损失函数losses
        self.epochs = epochs  # 训练轮数

    def train(self, train_loader, epoch, verbose, metric):
        """模型训练"""
        self.model.train()  # Sets the module in training mode
        train_loader_len = len(train_loader.dataset)
        for batch_idx, (target, text, lengths) in enumerate(train_loader):
            target, text, = target.to(self.device), text.to(self.device)

            # 反向传播固定格式
            self.optimizer.zero_grad()  # 梯度清零
            output = self.model(text, lengths).squeeze()  # 转换为向量
            loss = self.criterion(output, target)
            loss.backward()  # 反向传播
            self.optimizer.step()  # 执行一次优化步骤
            if (batch_idx + 1) % verbose == 0 or batch_idx == 0:
                trained_num = (batch_idx + 1) * train_loader.batch_size
                if trained_num >= train_loader_len:
                    trained_num = train_loader_len
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
        """模型评估"""
        self.model.eval()  # Sets the module in training mode
        predict_list = []
        y_true_list = []
        with torch.no_grad():
            for label, text, lengths in data_loader:
                text = text.to(self.device)
                predict = self.model(text, lengths).squeeze()
                predict_list.append(predict)
                y_true_list.extend(label.tolist())
        predict_all = torch.cat(predict_list, dim=0)  # 合并每个批次的预测值
        y_true = torch.tensor(y_true_list).to(self.device)
        if metric is not None:
            return self.criterion(predict_all, y_true).item(), metric(predict_all, y_true).item()
        else:
            return self.criterion(predict_all, y_true).item()

    def train_eval(self, train_loader, valid_loader=None, metric=None, verbose=20):
        """
        模型训练和评估
        """
        history = {'train_loss': [], 'val_loss': []}
        if metric is not None:
            history['train_' + metric.__name__] = []
            if valid_loader is not None:
                history['val_' + metric.__name__] = []
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
