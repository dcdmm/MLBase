
"""
选择排序:
* 不稳定(不能保证相同数字原有顺序)
* 比较:O(n^2)
* 交换:O(n)
* 额外的空间:O(1)
"""


def selection_sort(lst):
    for i in range(len(lst)):
        pos_min = i
        for j in range(i + 1, len(lst)):
            if lst[j] < lst[pos_min]:
                pos_min = j  # 最小值的索引

        lst[i], lst[pos_min] = lst[pos_min], lst[i]  # 将最小的元素交换到最前

    return lst


if __name__ == '__main__':
    disorder_lst = [3, 4, 51, 5, 77, 2, 8, 0, -1, 1]
    print(selection_sort(disorder_lst))
