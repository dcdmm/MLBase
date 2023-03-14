import argparse


def my_arg():
    parse = argparse.ArgumentParser(description='计算长方体体积的参数')
    parse.add_argument('--length',  # 可选参数(--开头)
                       type=float, default=1.0, help='这是长方体的长')
    parse.add_argument('width',  # 位置参数
                       type=float, default=1., help='这是长方体的宽')
    parse.add_argument('hight',
                       choices=[1., 2., 3., 4.],
                       type=float, default=1., help='这是长方体的高')
    return parse


if __name__ == '__main__':
    args = my_arg().parse_args()
    print(vars(args))
