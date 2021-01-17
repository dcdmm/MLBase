
# 给定一个字符串和需要替换的字符.列出所有(不同)的转换(需要替换字符大小写转换)可能
"""
For example:

sentence = "medium-one"
Rule = "io"

solutions = ["medium-one", "medIum-one", "medium-One", "medIum-One"]
"""


def replace(string, pl_index, low_or_upp):
    """将字符串pl_index位置处的字符大写或小写"""
    pre = string[:pl_index]
    behand = string[pl_index + 1:]
    if low_or_upp == 'lower':
        return pre + string[pl_index].lower() + behand
    else:
        return pre + string[pl_index].upper() + behand


def enter(sentence, seq_, max_index_):
    result = []
    recursive_finc(result, sentence, seq_, 0, max_index_)
    return result


def recursive_finc(result, sentence, nums, pos, max_index_):
    if sentence[pos] in nums:
        if True:
            sentence = replace(sentence, pos, low_or_upp='lower')
            if pos == max_index_:
                result.append(sentence[:])
            if pos <= len(sentence) - 2:  # 注意字符串index溢出
                # 字符串为不可变数据类型,故此处不需要进行回溯
                recursive_finc(result, sentence, nums, pos + 1, max_index_)
        if True:
            sentence = replace(sentence, pos, low_or_upp='upper')
            if pos == max_index_:
                result.append(sentence[:])  # 全部替换完成
            if pos <= len(sentence) - 2:
                recursive_finc(result, sentence, nums, pos + 1, max_index_)
    else:
        if pos <= len(sentence) - 2:
            recursive_finc(result, sentence, nums, pos + 1, max_index_)


if __name__ == '__main__':
    rule = 'abcd'  # 替换字符
    seq = list(rule) + list(rule.upper())  # 替换字符大写和小写组成的列表

    example_sentence = "apple crazy better"  # 句子包含4个替换字符,故有2^4=16种替换可能(乘法公式)
    length, example_re = len(example_sentence), example_sentence[::-1]

    max_index = 0  # 字符串中最后一个替换字符的位置
    for char in seq:
        find_index = example_re.find(char)
        if find_index >= 0:
            index = length - 1 - find_index
            if index > max_index:
                max_index = index

    last_result = enter(example_sentence, seq, max_index)
    print(last_result)
