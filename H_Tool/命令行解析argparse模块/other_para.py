import argparse

def my_arg():
    parse = argparse.ArgumentParser(description='计算长方体体积的参数')
    parse.add_argument('--length',
                       type=float, default=1.0, help='这是长方体的长')
    parse.add_argument('width', # 没有横杆开头的是必填参数,并且按照输入顺序进行设定
                       type=float, default=1., help='这是长方体的宽')
    parse.add_argument('hight',
                       choices=[1., 2., 3., 4.], # 允许的参数值
                       type=float, default=1., help='这是长方体的高')
    return parse


if __name__ == '__main__':
    # 在当前目录下命令行执行:python other_para.py 5.2 3.
    args = my_arg().parse_args()
    print(vars(args)) # 返回参数的字典