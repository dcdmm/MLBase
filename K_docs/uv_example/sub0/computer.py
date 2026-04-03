def sumx(a, b):
    return a + b


def multix(a, b):
    return a * b


def divx(a, b):
    if b == 0:
        raise ValueError("除数不能为零")
    return a / b


def subtractx(a, b):
    return a - b
