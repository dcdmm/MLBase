import pandas as pd
import os


def read_csv_file(path):
    df = pd.read_csv(path, index_col=0)
    return df


# 根据当前文件获取目标文件的绝对路径(其他目录结构导入该路径也不会改变)
DIR = os.path.abspath(os.path.dirname(__file__))
DIR = os.path.abspath(os.path.join(DIR, 'data'))
CONFIG_FILE_PATH = os.path.abspath(os.path.join(DIR, 'data.csv'))


def read_csv_file_():
    print(CONFIG_FILE_PATH)
    df = pd.read_csv(CONFIG_FILE_PATH, index_col=0)
    return df


if __name__ == '__main__':
    # module_import\file_operate\data\data.csv与module_import\file_operate\read_csv_file.py的相对位置
    path = 'data/data.csv'

    dataframe = read_csv_file(path)
    print(dataframe)

    print('#' * 100)

    print(CONFIG_FILE_PATH)
    dataframe_ = read_csv_file_()
    print(dataframe_)
