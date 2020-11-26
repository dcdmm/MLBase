class Empty(Exception):
    pass


class LinkedQueue:
    """单向链表实现队列"""
    class _Node:
        def __init__(self, element, pointer):
            self._element, self._pointer = element, pointer

    def __init__(self):
        self._head = self._tail = None
        self._size = 0  # number of queue elements

    def __len__(self):
        return self._size

    def is_empty(self):
        return self._size == 0

    def dequeue(self):
        """Remove and return the first element of the queue"""
        if self.is_empty():
            raise Empty('Queue is empty')
        answer = self._head._element
        node = self._head
        self._head = self._head._pointer
        node._pointer = node._element = None  # 显式地删除该节点
        self._size -= 1
        if self.is_empty():  # special case as queue is empty
            self._tail = None  # removed head had been the tail
        return answer

    # 在链表尾节点后添加节点,尾节点为最后进入的元素
    def enqueue(self, e):  # 不清楚画图
        """Add an element to the back of queue"""
        newest = self._Node(e, None)  # node will be new tail node
        if self.is_empty():
            self._head = newest  # special case:previously empty
        else:
            self._tail._pointer = newest
        self._tail = newest
        self._size += 1

    def printer_queue(self):
        node = self._head
        while node:
            print(node._element, end=" ")
            node = node._pointer
        print()


class StackWithQueue:
    """使用队列实现栈"""

    def __init__(self):
        self.queue = LinkedQueue()

    def push(self, x):
        self.queue.enqueue(x)

    def pop(self):
        size = len(self.queue)
        for i in range(1, size):
            self.queue.enqueue(self.queue.dequeue())  # 将队列最后元素之前的元素依次添加到该元素之后
        self.queue.dequeue()


if __name__ == '__main__':
    swq = StackWithQueue()
    swq.push(0)
    swq.push(1)
    swq.push(2)
    swq.push(3)
    swq.push(4)
    swq.push(5)  # 5为最后进入的元素
    swq.queue.printer_queue()

    swq.pop()
    swq.queue.printer_queue()

    swq.pop()
    swq.queue.printer_queue()  # 满足栈后进先出的原则
