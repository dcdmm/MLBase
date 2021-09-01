from torch.utils.tensorboard import SummaryWriter

'''
log_dir代表可视化的数据将会写入哪个文件夹,如果传入的参数是None,
这个类将会在当前目录下自动创建runs文件夹,然后在runs文件夹中创建一个新的文件夹,
文件夹名称和当前日期时间,以及当前的主机名有关.接着在这个新的文件夹中写入可视化数据
'''
log_dir = "./data/scalar"

# Creates a SummaryWriter that will write out events and summaries to the event file.
writer = SummaryWriter(log_dir=log_dir,
                       filename_suffix="tb")  # 可视化文件的后缀

for x in range(100):
    # Add scalar data to summary.
    writer.add_scalar(tag='y=pow_2_x',  # Data identifier
                      scalar_value=2 ** x,  # Value to save
                      global_step=x)  # Global step value to record

writer.close()
