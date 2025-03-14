from a import a_func, my_sum, n_var
import concurrent.futures


def b_func(m=0):
    af_result = a_func()
    return af_result + m


def initialize_worker(n_var_value):
    """初始化(所有)工作线程"""
    n_var.set(n_var_value)


def pp_my_sum(n=10):
    n_get = n_var.get(None)
    # 多线程环境中需要通过initializer+initargs设置上下文变量n_var的值
    with concurrent.futures.ThreadPoolExecutor(max_workers=None,
                                               initializer=initialize_worker,
                                               initargs=(n_get,)) as executor:
        f0_result = [executor.submit(my_sum) for _ in range(n)]
    return [f0r.result() for f0r in f0_result]
