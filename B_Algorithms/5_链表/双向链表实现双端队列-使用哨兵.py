
class Empty(Exception):
    """Error attempting to access an element from an empty container"""
    pass


class DoublyLindedBase:
    class _Node:
        def __init__(self, element, prev, pointer):
            self._element = element
            self._prev = prev
            self._pointer = pointer

        @property
        def element(self):
            return self._element

    def __init__(self):
        self._header = self._Node(None, None, None)
        self._trailer = self._Node(None, None, None)
        self._header._pointer = self._trailer
        self._trailer._prev = self._header
        self._size = 0

    def __len__(self):
        return self._size

    def is_empty(self):
        return self._size == 0

    def _insert_between(self, e, predecessor, successor):
        newest = self._Node(e, predecessor, successor)
        predecessor._pointer = newest
        successor._prev = newest
        self._size += 1
        return newest

    def _delete_node(self, node):
        predecessor = node._prev
        successor = node._pointer
        predecessor._pointer = successor
        successor._prev = predecessor
        self._size -= 1
        element = node._element # recode deleted element

        # 该节点与其他节点不必要的链接和储存元素将会被消除,从而帮助Python进行垃圾回收
        node._prev = node._pointer = node._element = None # ★★★★★deprecate node
        return element

# 在使用哨兵时,实现的关键时要记住队列的第一个元素并不存储头节点,
# 而是储存在头节点后的第一个节点(假设双端队列时非空的).
# 同时,尾结点之前的第一个节点中存储的时双端队列的最后一个元素
class LinedDeque(DoublyLindedBase):
    """Double-ended queue implementation based on aa doubly linked list"""
    def first(self):
        if self.is_empty():
            raise Empty('Deque is empty')
        return self._header._pointer._element # real item just after header

    def last(self):
        if self.is_empty():
            raise Empty('Deque is empty')
        return self._trailer._prev._element

    def insert_first(self, e):
        self._insert_between(e, self._header, self._header._pointer)

    def insert_last(self, e):
        self._insert_between(e, self._trailer._prev, self._trailer)

    def delete_fist(self):
        if self.is_empty():
            raise Empty('Deque is empty')
        return self._delete_node(self._header._pointer)

    def delete_last(self):
        if self.is_empty():
            raise Empty('Deque is empty')
        return self._delete_node(self._trailer._prev)

    def printer_deque(self):
        node = self._header
        while node:
            if node._element is not None:
                print(node.element, end =" ")
            node = node._pointer
        print()

if __name__ == '__main__':
    lq = LinedDeque()
    lq.insert_first(0)
    lq.insert_first(1)
    lq.insert_first(2)
    lq.printer_deque()

    lq.insert_last(-1)
    lq.insert_last(-2)
    lq.insert_last(-3)
    lq.printer_deque()

    lq.delete_fist()
    lq.delete_last()
    lq.printer_deque()

