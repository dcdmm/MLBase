def bar(str_bar):
    str_bar = str_bar * 3
    return str_bar

def baz(str):
    str_baz = "hello " + str + " world!"
    return str_baz


def trun(str_, n):
    str_trun = str_[:n]
    return str_trun

def foo(str1, str2):
    str_merge = trun(bar(str1) + baz(str2), 20)
    str_length = len(str_merge)
    return str_length


print(foo("duan", "chao"))