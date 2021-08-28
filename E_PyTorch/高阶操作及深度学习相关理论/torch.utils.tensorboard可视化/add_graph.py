import torchvision.models as models
import torch
from torch.utils.tensorboard import SummaryWriter

# 可视化展示:命令行执行tensorboard --logdir="graph文件夹所在的相对路径"

writer = SummaryWriter(log_dir='./data/graph', filename_suffix="tb")

fake_img = torch.randn(1, 3, 320, 320)
al = models.alexnet()

writer.add_graph(model=al, # 模型
                 input_to_model=fake_img) # 模型的输入符合形状即可
writer.close()

