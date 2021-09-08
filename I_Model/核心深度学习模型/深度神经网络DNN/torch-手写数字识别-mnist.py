from abc import ABC
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.utils.data as Data


batch_size = 200
learning_rate = 0.01
epochs = 100

# 手写数字训练数据集
train_loader = Data.DataLoader(
    datasets.MNIST('../../Other/datasets/PyTorch',
                   train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),  # 转换为tensor
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)

# 手写数字测试数据集
test_loader = Data.DataLoader(
    datasets.MNIST('../../Other/datasets/PyTorch',
                   train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)


class MLP(nn.Module, ABC):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 10),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.model(x)
        return x


net = MLP()
optimizer = optim.SGD(net.parameters(), lr=learning_rate)
criteon = nn.CrossEntropyLoss()  # 损失函数为交叉熵

for epoch in range(1, epochs + 1):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28 * 28)
        logits = net(data)
        loss = criteon(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (batch_idx + 1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.view(-1, 28 * 28)
        logits = net(data)
        test_loss += criteon(logits, target).item()

        pred = logits.data.max(1)[1]
        correct += pred.eq(target.data).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
