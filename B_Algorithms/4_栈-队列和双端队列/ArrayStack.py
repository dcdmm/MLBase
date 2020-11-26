
# 栈是由一系列对象组合的一个集合,这些对象的插入和删除操作遵循后进先出(LIFO)的原则.


# 当一个空的python列表调用pop方法是,正常情况下会触发一个IndexError,因为列表是基于索引的序列.
# 对于栈而言,这个选择并不恰当,因为这里并没有假定的所有.这里,定义一个新的异常类更为恰当.
class Empty(Exception):
    """Error attempting to access an element from an empty container"""
    pass


class ArrayStack(object):
    """用Python列表作为储存实现一个栈"""
    def __init__(self):
        """Create an empty stack"""
        self._data = []

    # 返回栈中元素的数量
    def __len__(self):
        """Return the number of elements in the stack"""
        return len(self._data)

    # 如果栈中不包含任何元素,则返回一个布尔值"True"
    def is_empty(self):
        """Return True if the stack is empty"""
        return len(self._data) == 0

    # O(1)
    # 将一个元素e添加到栈的栈顶
    def push(self, e):
        """Add element e to the top of the stack"""
        self._data.append(e)

    # O(1)
    # 在不移除栈顶元素的前提下,返回栈的栈顶元素;若栈为空,这个操作将会出错
    def top(self):
        """Return (but do not remove) the element at the top of the stack"""
        if self.is_empty():
            raise Empty('Stack is empty')
        return self._data[-1]

    # O(1)
    # 从栈中移除并返回栈顶的元素,如果此时栈顶是空的,这个操作将出错
    def pop(self):
        """Remove and return the element from the pop of the stack"""
        if self.is_empty():
            raise Empty('Stack is empty')
        return self._data.pop()

    def printstack(self):
        print(self._data[::-1])
        print()


if __name__ == '__main__':
    mystack = ArrayStack()
    print('size was: ', str(len(mystack)))
    mystack.push(1)
    mystack.push(2)
    mystack.push(3)
    mystack.push(4)
    mystack.push(5)
    print('size was: ', str(len(mystack)))
    mystack.printstack()

    mystack.pop()
    mystack.pop()
    print('size was: ', str(len(mystack)))
    mystack.printstack()

    print(mystack.top())
    mystack.pop()
    mystack.pop()
    mystack.pop()
    mystack.pop()

