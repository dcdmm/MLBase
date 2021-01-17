
# 逆置含有n个元素的序列S,即第一个元素成为最后一个元素,第二个元素成为倒数第二个元素,以此类推.

def reverse(S, start,  # 开始位置索引
            stop):  # 结束位置索引
    """Return elements in implicit slice S[start, stop]"""
    if start < stop:
        S[start], S[stop] = S[stop], S[start]
        reverse(S, start + 1, stop - 1)


if __name__ == '__main__':
    seq = [4, 3, 6, 2, 8, 9, 5]
    print(seq)
    reverse(seq, 0, 6)
    print(seq)
