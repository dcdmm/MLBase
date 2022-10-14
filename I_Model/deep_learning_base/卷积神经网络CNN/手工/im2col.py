import numpy as np


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    input_data:由(数据量,通道,高,宽)组成的4维数组构成的输入数据
    filter_h:滤波器的高
    filter_w:滤波器的宽
    stride:步幅(这里简化为高宽步幅相等)
    pad:填充(这里简化为高宽填充相等)
    col:input_data合适的二维展开
    """
    N, C, H, W = input_data.shape  # batch_num, channel, height, width
    out_h = int(np.floor((H + 2 * pad - filter_h + stride) / stride))  # 输出height
    out_w = int(np.floor((W + 2 * pad - filter_w + stride) / stride))  # 输出width

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')  # 填充数据(默认使用0进行填充)
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose((0, 4, 5, 1, 2, 3)).reshape(N * out_h * out_w,
                                                    -1)  # col.shape=(N*out_h*out_w, C*filter_h*filter_w)

    return out_h, out_w, col


if __name__ == '__main__':
    image = np.arange(1080).reshape((10, 3, 6, 6))
    print('(batch_num, channel, height, width)分别为:', image.shape)
    KernelOrPool_size = (3, 3)
    print('滤波器或池化窗口的高宽分别为:', KernelOrPool_size)
    out_height, out_widht, imcol = im2col(image, *KernelOrPool_size)
    print('输出高宽分别为:', (out_height, out_widht))
    print('图片经过转换后有:', imcol.shape)
