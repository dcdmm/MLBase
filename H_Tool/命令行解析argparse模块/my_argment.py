import argparse


def my_argment():
    parse = argparse.ArgumentParser(description='计算长方体体积的参数') # 参数描述
    parse.add_argument('--length',
                       type=float, # 把从命令行输入的结果转成设置的类型
                       default=1., # 参数默认值
                       choices=[1., 2., 3., 4., 5.], # 允许的参数值
                       help='这是长方体的长') # 参数命令的介绍
    parse.add_argument('--width', type=float, default=1.0, help='这是长方体的宽')
    parse.add_argument('--hight', type=float, default=1., help='这是长方体的高')
    return parse


if __name__ == '__main__':
    para = my_argment()
    print(para.description)
    print(vars(para.parse_args()))
    print(para.parse_args().length,
          para.parse_args().width,
          para.parse_args().hight)
    print(para.print_help())
