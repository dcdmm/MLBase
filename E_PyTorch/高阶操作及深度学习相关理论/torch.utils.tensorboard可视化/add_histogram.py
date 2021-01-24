from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter(log_dir='./data/histogram', filename_suffix='tb')

for x in range(2):
    data_union = np.arange(100)
    data_normal = np.random.normal(size=1000)
    writer.add_histogram(tag='distribution union', #  Data identifier
                         values=data_union, #  Values to build histogram
                         global_step=x, #  Global step value to record
                         bins='auto') #  One of {'tensorflow', 'auto, 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt'}. This determines how the bins are made.

    writer.add_histogram('distribution normal', data_normal, x, 'sqrt')


writer.close()

