import sys


class Empty(Exception):
    pass


class ArrayStack(object):
    def __init__(self):
        self._data = []

    def __len__(self):
        return len(self._data)

    def is_empty(self):
        return len(self._data) == 0

    def push(self, e):
        self._data.append(e)

    def top(self):
        if self.is_empty():
            raise Empty('Stack is empty')
        return self._data[-1]

    def pop(self):
        if self.is_empty():
            raise Empty('Stack is empty')
        return self._data.pop()

    def print_element(self):
        return [i for i in self._data[::-1]]


class MinStack(ArrayStack):
    """扩展栈的功能,得到栈中当前的最小值"""
    class NodeWithMin:  # 也可以使用元组,列表等数据类型代替
        def __init__(self, v, min_v):
            self._value = v
            self._min = min_v

    def __init__(self):
        super(MinStack, self).__init__()

    def get_min(self):
        """得到当前栈中的最小值"""
        if super(MinStack, self).is_empty():
            return sys.maxsize
        else:
            return self.top()._min  # 直接调用父类的push方法

    def push(self, v):
        """push当前进入栈的值和当前整个栈中的最小值(一一对应,通过NodeWithMin类)"""
        newMin = min(v, self.get_min())
        super(MinStack, self).push(self.NodeWithMin(v, newMin))  # 显示地调用父类的push方法(子类进行了重写)

    def print_element(self):
        return [i._value for i in self._data[::-1]]


class MinStack2(ArrayStack):
    """扩展栈的功能,得到栈中当前的最小值,相比于MinStack可以稍微节约内存空间"""

    def __init__(self):
        super(MinStack2, self).__init__()
        self.min_stack = ArrayStack()

    def get_min(self):
        if len(self.min_stack) == 0:
            return sys.maxsize
        else:
            return self.min_stack.top()

    def push(self, value):
        if value <= self.get_min():  # 若当前push进栈的值小于或等于self.min_stack中最小的值,则也将该值push进self._min_stack中.故比MinStack可以稍微节约内存空间
            self.min_stack.push(value)
        super(MinStack2, self).push(value)

    def pop(self):
        value = super(MinStack2, self).pop()
        if value == self.get_min():
            self.min_stack.pop()  # 若当前pop出栈的值等于self.min_stack中最小的值,则也将该值pop出self._min_stack中
        return value


def stack_test(minstack):
    """测试函数"""
    minstack.push(0)
    minstack.push(1)
    minstack.push(2)
    minstack.push(3)  # 栈顶元素
    print(minstack.print_element())
    print(minstack.get_min())  # 最小值为0

    minstack.push(-1)
    print(minstack.print_element())
    print(minstack.get_min())  # 最小值为-1

    minstack.push(-2)
    print(minstack.print_element())
    print(minstack.get_min())  # 最小值为-2

    minstack.pop()
    print(minstack.print_element())
    print(minstack.get_min())  # 最小值为-1


if __name__ == '__main__':
    test = 1
    if test == 0:
        minstack1 = MinStack()
        stack_test(minstack1)  # MinStack测试成功
    else:
        minstack2 = MinStack2()
        stack_test(minstack2)  # MinStack2测试成功
