from b import b_func, pp_my_sum
from a import a_var, n_var


# Call to set a new value for the context variable in the current context.
a_var.set(999)
n_var.set(3.3)

if __name__ == '__main__':
    print(b_func(1))  # print->1000
    print(pp_my_sum(10))
