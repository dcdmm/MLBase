
class Empty(Exception):
    """Error attempting to access an element from an empty container"""
    pass


class LinkedStack:
    class _Node:
        def __init__(self, element, pointer):  # initialize node's fields
            self._element = element  # reference to user's element
            self._pointer = pointer  # reference th next node

    def __init__(self):
        """Create an empty stack"""
        self._head = None  # referecne to the head node
        self._size = 0  # number of stack elements

    # O(1)
    def __len__(self):
        """Return the number of elements in the stack"""
        return self._size

    # O(1)
    def is_empty(self):
        """Return True if the stack is empty"""
        return self._size == 0

    # O(1)
    # ★★★★★由于所有栈的操作都会影响栈顶,因此规定栈顶在链表的头部
    def push(self, e):
        """Add element e to the top the stack"""
        self._head = self._Node(e, self._head)  # create and link a new node
        self._size += 1

    # O(1)
    def top(self):
        """Return (but no remove) the element at the top of the stack"""
        if self.is_empty():
            raise Empty('Stack is empty')
        return self._head._element  # top of stack i at head of list

    # O(1)
    def pop(self):
        """Remove and return the element from the top of the stack"""
        if self.is_empty():
            raise Empty('Stack is empty')
        answer = self._head._element
        node = self._head
        self._head = self._head._pointer  # bypass the former top node
        self._size -= 1
        node._pointer = node._element = None  # 显式地删除该节点
        return answer

    def printer(self):
        node = self._head
        while node:
            print(node._element, end=" ")
            node = node._pointer
        print()

    def printer_stack(self):
        node = self._head
        while node:
            print(node._element, end=" ")
            node = node._pointer
        print()


if __name__ == '__main__':
    ls = LinkedStack()
    ls.push(1)
    ls.push(2)
    ls.push(3)
    ls.push(4)
    ls.printer_stack()  # 4, 3, 2, 1

    ls.pop()
    ls.printer_stack()  # 4后进先出

    ls.pop()
    ls.pop()
    ls.pop()
    ls.pop()
    ls.printer_stack()
