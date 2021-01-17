
def comb(nums, t):
    result = []
    tmp = []
    combHelper(result, tmp, nums, t, 0)
    return result

def combHelper(result, tmp, nums, remains, start):
    if remains < 0: # 该组合之和不等于该数字
        return
    if remains == 0: # 该组合之和等于该数字
        result.append(tmp[:])
    else:
        for i in range(start, len(nums)):
            tmp.append(nums[i])
            combHelper(result, tmp, nums, remains - nums[i], i)
            tmp.pop()


if __name__ == '__main__':
    candidates = [2, 3, 4, 5, 6, 7, 8]
    total = 8

    # 8可以分解成candidates里哪些数字之和,如[2,2,2,2],[2,2,4],[2,3,3],[2,6]......
    print(comb(candidates, total))
