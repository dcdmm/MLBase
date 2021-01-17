
class Empty(Exception):
    """Error attempting to access an element from an empty container"""
    pass


class Outbound(Exception):
    pass


class MyLindedList:
    class _Node:
        def __init__(self, element, pointer):
            self._element = element
            self._pointer = pointer

    def __init__(self):
        self._head = None  # 链表的头节点
        self._tail = None  # 链表的尾节点
        self._size = 0

    def __len__(self):
        return self._size

    def is_empty(self):
        return self._size == 0

    def frist(self):
        """返回链表头节点的值"""
        if self.is_empty():
            raise Empty('LinedList is empty')
        return self._head._element

    def add_first(self, value):
        """在链表的头部插入一个元素"""
        newest = self._Node(value, None)
        if self.is_empty():
            self._head = newest
            self._tail = newest
            self._size += 1
        else:
            newest._pointer = self._head
            self._head = newest
            self._size += 1

    def add_last(self, value):
        """在链表的尾部插入一个元素"""
        newest = self._Node(value, None)
        if self.is_empty():
            self._head = newest
            self._tail = newest
            self._size += 1
        else:
            self._tail._pointer = newest
            self._tail = newest
            self._size += 1

    def remove_first(self):
        """在链表的头部删除一个节点"""
        if self._head is None:
            raise Empty('LinedList is empty')
        node = self._head
        self._head = self._head._pointer
        node._pointer = node._element = None  # 显式地删除该节点
        self._size -= 1

    def add(self, value, index):
        """在链表的任意index处添加一个节点"""
        if index < 0 or index > self._size:
            raise Outbound('index is out of bound')
        newest = self._Node(value, None)
        find_node = self._head
        if index == 0:
            self.add_first(value)
        elif index == self._size:
            self.add_last(value)
        else:
            for _ in range(index - 1):
                find_node = find_node._pointer
            newest._pointer = find_node._pointer
            find_node._pointer = newest
            self._size += 1

    def remove(self, index=0):
        """删除链表任意index处的一个节点"""
        if index < 0 or index >= self._size:
            raise Outbound('index is out of bound')
        if self._head is None:
            raise Empty('LinkedList is empty')
        if index == 0:
            self.remove_first()
        else:
            fond_node = self._head
            for i in range(index - 1):
                fond_node = fond_node._pointer
            fond_node._pointer = fond_node._pointer._pointer
            self._size -= 1

    def element(self):
        """使用生成器遍历链表"""
        p = self._head
        while p is not None:
            yield p._element
            p = p._pointer


if __name__ == '__main__':
    print('add_first方法测试***********************************')
    my_linkedlist = MyLindedList()
    my_linkedlist.add_first(0)
    my_linkedlist.add_first(1)
    my_linkedlist.add_first(2)
    my_linkedlist.add_first(3)
    my_linkedlist.add_first(4)
    print(list(my_linkedlist.element()), end='\n\n')

    print('add_last方法测试***********************************')
    my_linkedlist.add_last(-1)
    my_linkedlist.add_last(-2)
    my_linkedlist.add_last(-3)
    my_linkedlist.add_last(-4)
    print(list(my_linkedlist.element()), end='\n\n')

    print('remove_first方法测试***********************************')
    my_linkedlist.remove_first()
    my_linkedlist.remove_first()
    print(list(my_linkedlist.element()), end='\n\n')

    print('add方法测试***********************************')
    my_linkedlist.add(777, 0)
    print(list(my_linkedlist.element()))
    my_linkedlist.add(888, 1)
    print(list(my_linkedlist.element()))
    my_linkedlist.add(999, 7)
    print(list(my_linkedlist.element()), end='\n\n')

    print('remove方法测试***********************************')
    my_linkedlist.remove(0)
    print(list(my_linkedlist.element()))
    my_linkedlist.remove(3)
    print(list(my_linkedlist.element()), end='\n\n')
