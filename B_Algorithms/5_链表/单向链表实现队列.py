
class Empty(Exception):
    """Error attempting to access an element from an empty container"""
    pass


class LinkedQueue:
    class _Node:
        def __init__(self, element, pointer):  # initialize node's fields
            self._element = element  # reference to user's element
            self._pointer = pointer  # reference th next node

    def __init__(self):
        """Create an empty queue"""
        self._head = None
        self._tail = None
        self._size = 0  # number of queue elements

    def __len__(self):
        """Return the number of elements in the queue"""
        return self._size

    def is_empty(self):
        """Return True if the queue is empty"""
        return self._size == 0

    def first(self):
        """Return (but do not remove) the element at the front of the queue """
        if self.is_empty():
            raise Empty('Queue is empty')
        return self._head._element

    def dequeue(self):
        """Remove and return the first element of the queue"""
        if self.is_empty():
            raise Empty('Queue is empty')
        answer = self._head._element
        node = self._head
        self._head = self._head._pointer
        node._pointer = node._element = None  # 显式的删除该节点
        self._size -= 1
        if self.is_empty():  # special case as queue is empty
            self._tail = None  # removed head had been the tail
        return answer

    # 在链表尾节点后添加节点,尾节点为最后进入的元素
    def enqueue(self, e):  # 不理解画图
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


if __name__ == '__main__':
    lq = LinkedQueue()
    lq.enqueue(0)
    lq.enqueue(1)
    lq.enqueue(2)
    lq.enqueue(3)
    lq.printer_queue()  # 0, 1, 2, 3

    lq.dequeue()
    lq.dequeue()
    lq.printer_queue()  # 0, 1先进先出

    lq.dequeue()
    lq.dequeue()
    lq.dequeue()
