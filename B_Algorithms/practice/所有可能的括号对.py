
def generateParenthesis(n):
    result = []

    def generate(prefix, left, right):
        if right == 0:
            result.append(prefix)
        if left > 0:
            generate(prefix + '(', left-1, right)  # 字符串为不可变数据类型,故此处不需要进行回溯
        if right > left:
            generate(prefix + ')', left, right-1)
    generate('', n, n)
    return result


if __name__ == '__main__':
    print(generateParenthesis(3))  # 所有可能的括号对
