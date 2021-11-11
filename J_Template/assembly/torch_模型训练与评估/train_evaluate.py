import torch
import torch.utils.data as Data


class Train_Evaluate:
    """
    pytorch模型训练与评估组件
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
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # 反向传播固定格式
            self.optimizer.zero_grad()  # 梯度清零
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()  # 反向传播
            self.optimizer.step()  # 执行一次优化步骤
            if (batch_idx + 1) % verbose == 0 or batch_idx == 0:
                if metric is not None:  # 带额外评估指标的输出
                    metric_name = metric.__name__
                    metric_result = metric(output, target)
                    if batch_idx == 0:
                        print('Train Epoch: {:<2} [{:<5}/{} ({:<3.0f}%)]\tLoss: {:.6f}\t{}: {:.6f}'.format(epoch, 0,
                                                                                                           len(train_loader.dataset),
                                                                                                           0,
                                                                                                           loss.item(),
                                                                                                           metric_name,
                                                                                                           metric_result))
                    else:
                        print('Train Epoch: {:<2} [{:<5}/{} ({:<3.0f}%)]\tLoss: {:.6f}\t{}: {:6f}'.
                              format(epoch,
                                     (batch_idx + 1) * len(data),
                                     len(train_loader.dataset),
                                     (100. * (batch_idx + 1)) / len(train_loader),
                                     loss.item(), metric_name, metric_result))
                else:
                    if batch_idx == 0:
                        print('Train Epoch: {:<2} [{:<5}/{} ({:<3.0f}%)]\tLoss: {:.6f}'.
                              format(epoch,
                                     0,
                                     len(train_loader.dataset),
                                     0,
                                     loss.item()))
                    else:
                        print('Train Epoch: {:<2} [{:<5}/{} ({:<3.0f}%)]\tLoss: {:.6f}'.
                              format(epoch,
                                     (batch_idx + 1) * len(data),
                                     len(train_loader.dataset),
                                     (100. * (batch_idx + 1)) / len(train_loader),
                                     loss.item()))

        print('-' * 100)

    def eval(self, data_loader, metric):
        """模型评估"""
        data_loader_no_shuffle = Data.DataLoader(data_loader.dataset, batch_size=data_loader.batch_size,
                                                 shuffle=False)  # 必须设定shuffle=False
        self.model.eval()  # Sets the module in training mode
        predict_list = []
        with torch.no_grad():
            for data, _ in data_loader_no_shuffle:
                data = data.to(self.device)
                predict = self.model(data)
                predict_list.append(predict)
        predict_all = torch.cat(predict_list, dim=0)  # 合并每个批次的预测值
        y_true = data_loader_no_shuffle.dataset.tensors[1].to(self.device)
        if metric is not None:
            return self.criterion(predict_all, y_true).item(), metric(predict_all, y_true).item()
        else:
            return self.criterion(predict_all, y_true).item()

    def train_eval(self, train_loader, valid_loader=None, metric=None, verbose=20):
        """
        模型训练和评估
        """
        history = {'train_loss': [], 'val_loss': []}
        for epoch in range(self.epochs):
            self.train(train_loader, epoch=epoch, verbose=verbose, metric=metric)
            if metric is not None:
                history['train_loss'].append(self.eval(train_loader, metric=metric)[0])
                history['train_' + metric.__name__].append(self.eval(train_loader, metric=metric)[1])
                if valid_loader is not None:
                    history['val_loss'].append(self.eval(valid_loader, metric=metric)[0])
                    history['val_' + metric.__name__].append(self.eval(valid_loader, metric=metric)[0])
                else:
                    history.pop('val_loss')
            else:
                history['train_loss'].append(self.eval(train_loader, metric=metric))
                if valid_loader is not None:
                    history['val_loss'].append(self.eval(valid_loader, metric=metric))
                else:
                    history.pop('val_loss')
        return history

