
"""
冒泡排序:
* 稳定(不改变相同数字原有顺序)
* 比较和交换:O(n^2);最好O(n),此时初始数组已基本排序或已完全排序
* 额外的空间:O(1)
"""


def bubble_sort(lst):
    for i in range(len(lst)):
        for j in range(len(lst) - i - 1):
            if lst[j] > lst[j + 1]:
                temp = lst[j + 1]
                lst[j + 1] = lst[j]
                lst[j] = temp # 交换顺序
    return lst


def bubble_sort_mod(lst):
    for i in range(len(lst)):
        is_sorted = True
        for j in range(1, len(lst) - i):
            if lst[j] < lst[j - 1]:
                lst[j], lst[j - 1] = lst[j - 1], lst[j]
                is_sorted = False
        if is_sorted:  # 若已经是排好序的,则可以提早跳出循环
            break
    return lst


if __name__ == '__main__':
    disorder_lst = [3, 4, 51, 5, 77, 2, 8, 0, -1, 1]

    print(bubble_sort(disorder_lst))

    print(bubble_sort_mod(disorder_lst))
