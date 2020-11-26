
class Empty(Exception):
    """Error attempting to access an element from an empty container"""
    pass

class CircularQueue:
    class _Node:
        def __init__(self, element, pointer):  # initialize node's fields
            self._element = element  # reference to user's element
            self._pointer = pointer  # reference th next node
    # 简单的设置self._tail=self._tail._pointer,使原来的头部变成新的尾部
    # 对于循环链表,显然不需要同时保存指向头部和尾部的引用(指针).只要保存一个指向尾部的引用,就总能通过尾部的"pointer"引用找到头部
    def __init__(self):
        """Create an empty queue"""
        self._tail = None
        self._size = 0

    def __len__(self):
        """Return the number of elements in the queue"""
        return self._size

    def is_empty(self):
        """Return True if the queue is empty"""
        return self._size == 0

    def first(self):
        """Return (but not remove) the element at the front the queue"""
        if self.is_empty():
            raise Empty('Queue is empty')
        head = self._tail._pointer
        return head._element

    def enqueue(self, e):
        """Add an element to the back of queue"""
        newest = self._Node(e, None) # node will be new tail node
        if self.is_empty():
            newest._pointer = newest # initialize circularly
        else:
            newest._pointer = self._tail._pointer # new node points to head
            self._tail._pointer = newest # old tail points to new node
        self._tail = newest # new node becomes the tail
        self._size += 1

    def dequeue(self):
        """Remove and return the first element of the queue"""
        if self.is_empty():
            raise Empty('Queue is empty')
        oldhead = self._tail._pointer
        oldelement = oldhead._element
        if self._size == 1:
            self._tail = None
        else:
            self._tail._pointer = oldhead._pointer # bypass the old head
        self._size -= 1
        oldhead._pointer = oldhead._element = None # 显式地删除该节点
        return oldelement

    def rotate(self):
        """Rotate ront element to the back of the queue"""
        if self._size > 0:
            self._tail = self._tail._pointer

    def element(self):
        """使用生成器遍历链表"""
        p = self._tail._pointer
        while p is not None:
            yield p._element
            p = p._pointer
            if p == self._tail:
                yield p._element
                break # 循环链表注意遍历到最后时跳出循环

if __name__ == '__main__':
    cq = CircularQueue()
    cq.enqueue(0)
    cq.enqueue(1)
    cq.enqueue(2)
    print(list(cq.element()))

    print(cq.dequeue())
    print(list(cq.element()))

