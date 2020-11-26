class Node:
    def __init__(self, element, pointer):
        self._element = element
        self._pointer = pointer


def has_cycle(head):
    """判断链表是否有环"""
    slow = fast = head  # 链表的头节点
    # 使用双指针
    while fast._pointer is not None and fast._pointer._pointer is not None:
        fast = fast._pointer._pointer  # 快的指针
        slow = slow._pointer  # 慢的指针
        if fast == slow:  # 如果存在存在环,必能使得fast==slow(易证)
            return True

    return False


if __name__ == '__main__':
    node0 = Node(0, None)
    node1 = Node(1, node0)
    node2 = Node(2, node1)
    node3 = Node(3, node2)
    node4 = Node(4, node3)
    node5 = Node(5, node4)
    node6 = Node(5, node5)
    print(has_cycle(node6))

    node0._pointer = node4  # 此时有环
    print(has_cycle(node6))
