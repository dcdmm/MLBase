import torchvision.models as models
from torchsummary import summary

# 打印显示网络结构和参数
print(summary(model=models.alexnet(),
              batch_size=400,
              input_size=(3, 320, 320), # 输入形状;input_size=(channels, H, W)
              device="cpu")) # 'cpu'或'cuda'

