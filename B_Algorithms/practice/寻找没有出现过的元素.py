
def findDisappearedNumbers(nums):
    """
    给定一个长度为n的列表(元素范围为[1, n]),有些元素出现了>=1次,而有些元素则没有出现
    寻找哪些元素没有出现,要求空间复杂度为O(1)
    """
    for i in range(len(nums)):
        index = abs(nums[i]) - 1  # 将出现过的元素作为列表的下标
        nums[index] = - abs(nums[index])  # 将该位置元素设置为对应的负值

    return [i + 1 for i in range(len(nums)) if nums[i] > 0]


if __name__ == '__main__':
    lst = [4, 3, 2, 7, 8, 2, 3, 1, 3]
    print(findDisappearedNumbers(lst))
