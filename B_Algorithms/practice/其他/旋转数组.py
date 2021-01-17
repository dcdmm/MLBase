# 给一个n×n的数组,旋转90度

import numpy as np
import copy


def rotate(matrix):
    """从最外层一层逐一层的进行旋转"""
    form = matrix.shape[0]
    for layer in range(form // 2):
        temper0 = copy.copy(matrix[0 + layer:form - layer, form - 1 - layer])  # 最后一列(注意array(numpy)共享数据的物理地址)
        temper1 = copy.copy(matrix[form - 1 - layer, 0 + layer:form - layer])  # 最后一行
        temper2 = copy.copy(matrix[0 + layer:form - layer, 0 + layer])  # 第一列

        matrix[0 + layer:form - layer, form - 1 - layer] = matrix[0 + layer, 0 + layer:form - layer]  # 最后一列等于第一行
        matrix[form - 1 - layer, 0 + layer:form - layer] = temper0[::-1]  # 最后一行等于最后一列
        matrix[0 + layer:form - layer, 0 + layer] = temper1  # 第一列等于最后一行
        matrix[0 + layer, 0 + layer:form - layer] = temper2[::-1]  # 第一行等于第一列


if __name__ == '__main__':
    arr = np.arange(81).reshape(9, 9)  # 大小为9的数组只要旋转4层
    print(arr, end='\n\n')
    rotate(arr)
    print(arr)
