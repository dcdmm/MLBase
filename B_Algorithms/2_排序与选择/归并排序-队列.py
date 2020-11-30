
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


def merge(S1, S2, S):
    """merge two sorted queue instance S1 and S2 into empty queus S"""
    while not S1.is_empty() and not S2.is_empty():
        if S1.first() < S2.first():
            S.enqueue(S1.dequeue())
        else:
            S.enqueue(S2.dequeue())
    while not S1.is_empty():
        S.enqueue(S1.dequeue())  # move remaining elements of S1 to S
    while not S2.is_empty():
        S.enqueue(S2.dequeue())  # move remaining elements of S2 to S


def merge_sort(S):  # 使用队列实现归并排序
    """Sort the elements of queue S using the merge-sort algorithm"""
    n = len(S)
    if n < 2:
        return
    # divide
    S1 = LinkedQueue()  # 额外的内存空间
    S2 = LinkedQueue()

    # 功能等价于于归并排序-list.ipynb中49-53行代码
    while len(S1) < n // 2:
        S1.enqueue(S.dequeue())
    while len(S2) < n // 2:
        S2.enqueue(S.dequeue())

    # conquer(with recursion)
    merge_sort(S1)
    merge_sort(S2)
    # merge results
    merge(S1, S2, S)


if __name__ == '__main__':
    lq = LinkedQueue()
    lq.enqueue(24)
    lq.enqueue(45)
    lq.enqueue(63)
    lq.enqueue(85)
    lq.enqueue(17)
    lq.enqueue(31)
    lq.enqueue(50)
    lq.enqueue(96)
    lq.printer_queue()

    merge_sort(lq)
    lq.printer_queue()
