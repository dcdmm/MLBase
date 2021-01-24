from torch.utils.tensorboard import SummaryWriter
import numpy as np

max_epoch = 100
writer = SummaryWriter(log_dir='./data/scalars', filename_suffix="tb")

for x in range(max_epoch):
    # Adds many scalar data to summary.
    writer.add_scalars(main_tag='data/scalar_group', # The parent name for the tag
                       tag_scalar_dict={"xsinx": x * np.sin(x), # Key-value pair storing the tag and corresponding values
                                        "xcosx": x * np.cos(x)},
                       global_step=x) # Global step value to record
writer.close()