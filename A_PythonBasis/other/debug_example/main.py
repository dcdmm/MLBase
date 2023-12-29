import pandas as pd
import numpy as np

from utilty import square_xy, map_x


def sum_xy(x, y):
    z = x + y
    return z


def debug_t(lst0, lst1):
    shape0 = len(lst0)
    shape1 = len(lst1)
    new_lst0, new_lst1 = [], []
    for i in range(shape0):
        for j in range(shape1):
            new_lst0.append(square_xy(lst0[i], lst1[j]))
            new_lst1.append(sum_xy(lst0[i], lst1[j]))
    new_array0 = np.array(new_lst0)
    new_array0 = np.diff(new_array0)
    new_df0 = pd.DataFrame(new_array0)
    new_df1 = pd.DataFrame(new_lst1)
    result = new_df0 + new_df1
    result.columns = ["A"]
    last_result = result.map(map_x)
    return last_result


lst0 = [1, 2, 3]
lst1 = [-3, -2, -1]
print(debug_t(lst0, lst1))
