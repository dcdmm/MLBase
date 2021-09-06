from torch.utils.tensorboard import SummaryWriter
import numpy as np

labels = np.random.randint(2, size=100)  # binary label
predictions = np.random.rand(100)

log_dir = "./data/pr_curve"
writer = SummaryWriter(log_dir=log_dir,
                       filename_suffix="tb")

# Adds precision recall curve.
'''
tag (string) – Data identifier

labels (torch.Tensor, numpy.array, or string/blobname) – Ground truth data. Binary label for each element.

predictions (torch.Tensor, numpy.array, or string/blobname) – The probability that an element be classified as true. Value should be in [0, 1]

global_step (int) – Global step value to record
'''
writer.add_pr_curve(tag='pr_curve', labels=labels, predictions=predictions, global_step=0)
writer.close()
