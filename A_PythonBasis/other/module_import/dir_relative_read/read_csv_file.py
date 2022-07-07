import pandas as pd


def read_csv_file(path):
    df = pd.read_csv(path, index_col=0)
    return df


if __name__ == '__main__':
    # module_import\dir_relative_read\data.csv与module_import\dir_relative_read\read_csv_file.py的相对位置
    path = './data.csv'
    dataframe = read_csv_file(path)
    print(dataframe)
