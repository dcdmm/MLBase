
class MyLindedList:
    class _Node:
        def __init__(self, element, pointer):
            self._element, self._pointer = element, pointer

    def __init__(self):
        self._head = None  # 链表的头节点
        self._tail = None  # 链表的尾节点

    def is_empty(self):
        return self._head is None

    def add_first(self, value):
        """在链表的头部插入一个元素"""
        newest = self._Node(value, None)
        if self.is_empty():
            self._head = self._tail = newest
        else:
            newest._pointer = self._head
            self._head = newest

    def element(self):
        """使用生成器遍历链表"""
        p = self._head
        while p is not None:
            yield p._element
            p = p._pointer


def find_middle(lst):
    """寻找一个单向链表的中间节点(假设链表没有._size属性)"""
    assert lst._head is not None and lst._head._pointer is not None

    fast = slow = lst._head  # 开始时都处于头节点
    # 使用双指针
    while fast._pointer is not None and fast._pointer._pointer is not None:
        fast = fast._pointer._pointer  # 块的指针
        slow = slow._pointer  # 慢的指针

    return slow._element


if __name__ == '__main__':
    ll = MyLindedList()
    ll.add_first(0)
    ll.add_first(1)
    ll.add_first(2)
    ll.add_first(3)
    ll.add_first(4)
    ll.add_first(5)
    ll.add_first(6)
    print(list(ll.element()))
    print(find_middle(ll))

    ll.add_first(7)
    print(list(ll.element()))
    print(find_middle(ll))

    ll.add_first(8)
    print(list(ll.element()))
    print(find_middle(ll))
