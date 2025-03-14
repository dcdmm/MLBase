import contextvars
import time

a_var = contextvars.ContextVar('a')  # 上下文变量
n_var = contextvars.ContextVar('n')


def a_func(a=0):
    # Return a value for the context variable for the current context.
    a_get = a_var.get(None)
    if a_get is not None:
        a = a_get
    return a


def my_sum():
    n_get = n_var.get(1)
    time.sleep(n_get)
    return n_get
