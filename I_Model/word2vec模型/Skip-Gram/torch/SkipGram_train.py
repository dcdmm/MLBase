import torch
import time
from SkipGram_loss import SkipGram_loss
from SkipGramModel import SkipGramModel


def train(dataset, embed_size, embed_dimension, lr=0.01, num_epochs=10):
    """训练SkipGram模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("train on", device)
    net_ = SkipGramModel(embed_size, embed_dimension)  # 模型初始化
    net_ = net_.to(device)
    optimizer = torch.optim.SparseAdam(net_.parameters(), lr=lr)  # 优化器
    criteon = SkipGram_loss()  # 损失函数(自定义损失函数)

    for epoch in range(num_epochs):
        start, l_sum, n = time.time(), 0.0, 0
        for batch in dataset:
            center, context_negative, mask, label = [d.to(device) for d in batch]
            pred = net_(center, context_negative)
            l = criteon(pred.view(label.shape), label, mask).mean()  # 一个batch的平均loss
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            l_sum += l.cpu().item()
            n += 1
        print('epoch %d, loss %.2f, time %.2fs'
              % (epoch + 1, l_sum / n, time.time() - start))

    return net_
