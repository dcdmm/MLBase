import torch


class Train_Evaluate:
    def __init__(self, model, optimizer, criterion, epochs=5, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model = model
        self.model.to(device)
        self.optimizer = optimizer  # 优化器
        self.criterion = criterion  # 损失函数
        self.epochs = epochs  # 训练轮数

    def fit(self, train_loader, epoch, verbose=10):
        """模型训练"""
        self.model.train()  # Sets the module in training mode
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            if (batch_idx + 1) % verbose == 0 or batch_idx == 0:
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

    def predict(self, X_loader):
        """模型预测"""
        self.model.eval()  # Sets the module in training mode
        predict_result = []
        with torch.no_grad():
            for data in X_loader:
                data = data.to(self.device)  # 对X_loader分批次预测(避免全量预测显存不够)
                output = self.model(data)
                predict_result.append(output.tolist())
        # predict_result.shape=(len(X_loader), X_loader.batch_size, 模型分类数)
        predict_result = torch.tensor(predict_result)
        predict_result = predict_result.reshape(len(X_loader.dataset), -1)
        return predict_result

    def score(self, X_loader, y):
        """验证数据集评估"""
        predict_result = self.predict(X_loader)
        predict_result, y = predict_result.to(self.device), y.to(self.device)
        return self.criterion(predict_result, y)

    def train_eval(self, train_loader, verbose=10, valid_sets=None):
        metric_lst = []
        for i in range(self.epochs):
            self.fit(train_loader, verbose=verbose, epoch=i)
            if valid_sets is not None:
                temp = []
                for val in valid_sets:
                    metric_result = self.score(val[0], val[1])
                    temp.append(metric_result.item())
                metric_lst.append(temp)

        return metric_lst
