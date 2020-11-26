
class Empty(Exception):
    """Error attempting to access an element from an empty container"""
    pass


class QueueWithTwoStacks:
    """使用两个栈来实现一个队列"""

    def __init__(self):
        # 这里用list表示栈
        self.insertStack = []
        self.popStack = []

    def enqueue(self, e):
        self.insertStack.append(e)

    def dequeue(self):
        if len(self.insertStack) == 0 and len(self.popStack) == 0:
            raise Empty('Stack is empty')

        if len(self.popStack) == 0:
            while len(self.insertStack) != 0:
                self.popStack.append(self.insertStack.pop())

        return self.popStack.pop()


if __name__ == '__main__':
    myqueue = QueueWithTwoStacks()
    myqueue.enqueue(3)
    print(myqueue.insertStack)
    myqueue.enqueue(2)
    print(myqueue.insertStack)
    myqueue.enqueue(1)
    print(myqueue.insertStack)

    print()

    # 符合先进先出的原则
    myqueue.dequeue()
    print(myqueue.popStack)
    myqueue.dequeue()
    print(myqueue.popStack)
    myqueue.dequeue()
    print(myqueue.popStack)
