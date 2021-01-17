"""
给定一个正整数n,计算出小于等于n的质数有多少个
比如17,则返回7,因为小于等于7的质数有2, 3, 5, 7, 13, 17
"""

import math


def count_prime(n):
    """筛选法求质数"""
    is_prime = [True] * (n + 1)

    i = 2
    t_n = math.sqrt(n)
    while i <= t_n:  # 只需i*i<=n即可
        if is_prime[i]:
            j = i
            while j * i <= n:
                is_prime[i * j] = False  # i的倍数的数不可能为质数
                j += 1
        i += 1

    prime_lst = []
    for j in range(2, n + 1):
        if is_prime[j]:
            prime_lst.append(j)

    return prime_lst


if __name__ == '__main__':
    find_prime = count_prime(49)
    print(find_prime)
    print(len(find_prime))
