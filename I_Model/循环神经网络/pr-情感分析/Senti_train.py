import torch


class Train_Evaluate:
    def __init__(self, model, optimizer=None, criterion=None, device=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model.to(device)

    @staticmethod
    def accuracy(preds, y):
        rounded_preds = torch.round(torch.sigmoid(preds))
        correct = (rounded_preds == y).to(torch.float32)
        acc = correct.sum() / len(correct)
        return acc

    def repeat_code(self, batch):
        """train,evaluta重复的代码"""
        (text, text_lengths), label = batch.text, batch.label
        text, text_lengths, label = text.to(self.device), text_lengths.to(self.device), label.to(self.device)
        predictions = self.model(text, text_lengths).squeeze()  # 转换为向量
        loss = self.criterion(predictions, label)
        acc = self.accuracy(predictions, label)
        return loss, acc

    def train(self, train_data):
        self.model.train()
        epoch_loss = 0
        epoch_acc = 0
        length = len(train_data)

        for batch in train_data:
            self.optimizer.zero_grad()
            loss, acc = self.repeat_code(batch)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            epoch_acc += acc.item()

        loss_result = epoch_loss / length
        acc_result = epoch_acc / length
        return loss_result, acc_result

    def evaluate(self, test_data):
        self.model.eval()
        epoch_loss = 0
        epoch_acc = 0
        length = len(test_data)

        with torch.no_grad():
            for batch in test_data:
                loss, acc = self.repeat_code(batch)
                epoch_loss += loss.item()
                epoch_acc += acc.item()
        loss_result = epoch_loss / length
        acc_result = epoch_acc / length
        self.model.train()  # 重新进入训练模式
        return loss_result, acc_result
