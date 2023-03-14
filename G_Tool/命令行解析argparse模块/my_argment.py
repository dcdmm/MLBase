import argparse


def my_argment():
    # Object for parsing command line strings into Python objects.
    parse = argparse.ArgumentParser(
        description='计算长方体体积的参数'  # A description of what the program does
    )
    # adding argument actions
    parse.add_argument('--length',
                       # The type to which the command-line argument should be converted.
                       type=float,
                       # The value produced if the argument is absent from the command line and if it is absent from the namespace object.
                       default=1.,
                       # A sequence of the allowable values for the argument.
                       choices=[1., 2., 3., 4., 5.],
                       # A brief description of what the argument does.
                       help='这是长方体的长')
    parse.add_argument('--width', type=float, default=1.0, help='这是长方体的宽')
    parse.add_argument('--hight', type=float, help='这是长方体的高')
    return parse


if __name__ == '__main__':
    para = my_argment()
    print(para.description)

    '''
    ArgumentParser.print_help(file=None)
        Print a help message, including the program usage and information about the arguments registered with the ArgumentParser. If file is None, sys.stdout is assumed.
    '''
    para.print_help()

    print(vars(para.parse_args(args=[])))  # 命令行参数字典
    print("length:", para.parse_args().length,
          "width:", para.parse_args().width,
          "hight:", para.parse_args().hight)
