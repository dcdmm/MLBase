from  my_argment import  my_argment

para = my_argment()
le = my_argment().parse_args().length
wi = my_argment().parse_args().width
hi = my_argment().parse_args().hight

var_dict = vars(para.parse_args()) # 返回参数的字典

class volume_arg:
    def __init__(self, length, width, higth):
        self.length = length
        self.width = width
        self.hight = higth

    def calcu(self):
        return self.length * self.width * self.hight


if __name__ == '__main__':
    print(var_dict)
    vo = volume_arg(le, wi, hi)
    print(vo.calcu())