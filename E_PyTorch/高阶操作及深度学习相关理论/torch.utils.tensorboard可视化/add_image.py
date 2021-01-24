import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import scipy.misc
import torch

trans = transforms.ToPILImage()
lena = scipy.misc.face()
img = transforms.ToTensor()(lena)
img_batch = torch.stack((img, img, img, img, img, img.clone().fill_(-10)), 0) # 6张图片

img_grid0 = make_grid(img_batch)
img_grid1 = make_grid(img_batch, padding=50)
img_grid2 = make_grid(img_batch, normalize=True)

writer = SummaryWriter(log_dir='./data/image', filename_suffix="tb")
writer.add_image(tag='img grid', # Data identifier
                 img_tensor=img_grid0, # Image data
                 global_step=0, # Global step value to record
                 dataformats='CHW') # 数据格式:CHW(通道数,高,宽),HWC,HW(灰度图)
writer.add_image(tag='img grid',
                 img_tensor=img_grid1,
                 global_step=1,
                 dataformats='CHW')
writer.add_image(tag='img grid',
                 img_tensor=img_grid2,
                 global_step=2,
                 dataformats='CHW')

writer.close()