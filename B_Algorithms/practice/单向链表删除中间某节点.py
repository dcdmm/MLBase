
def delete_node(node):
    """单向链表删除node节点(已知node节点,间接进行删除)"""
    del_node = node._pointer
    node._element = node._pointer._element
    node._pointer = node._pointer._pointer
    del_node._element = del_node._pointer = None  # 显式地删除该节点
