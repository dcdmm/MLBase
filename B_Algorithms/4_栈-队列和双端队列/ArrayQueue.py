
# 队列是由一系列对象组合的集合,这些对象的插入和删除遵循先进先出(First in Frist out,FIFO)的原则

class Empty(Exception):
    """Error attempting to access an element from an empty container"""
    pass


class ArrayQueue:
    DEFAULT_CAPACITY = 10

    def __init__(self):
        """Create an new queue"""
        self._data = [None] * ArrayQueue.DEFAULT_CAPACITY  # modeerate capacity for all new queues
        self._size = 0  # 当前储存在队列内的元素
        self._front = 0  # 是一个整数,代表_data实例队列中的第一个元素的索引(假定这个队列不为空)

    # O(1)
    # 返回队列中元素的数量
    def __len__(self):
        """Return the number of element in the queues"""
        return self._size

    # O(1)
    # 如果队列没有包含任何元素则返回布尔值"True"
    def is_empty(self):
        """Return True if queue is empty"""
        return self._size == 0

    # O(1)
    # 在不移除的前提下返回队列的第一个元素;如果队列为空,则触发一个错误
    def first(self):
        """Return (but no remove) the element at the front of the queue"""
        if self.is_empty():
            raise Empty('Queue is empty')
        return self._data[self._front]

    # O(1)
    # 从队列中移除并返回第一个元素,如果队列为空,则触发一个错误
    def dequeue(self):
        """Remove and return the first element of the queue"""
        if self.is_empty():
            raise Empty('Queue is empty')
        answer = self._data[self._front]
        # 设为None的原因与Python回收未使用空间的机制有关.在内部,Python对已存的对象维护了一个对其的引用计数的计数器.
        # 如果计数变为0,这个对象实际上就无法访问,那么系统会回收这部分的内存以备将来使用
        self._data[self._front] = None
        # 如果有一个长度为10的列表,并且一个索引为7的首部,可以通过计算(7+1)%10=8来更新首部
        # 同样,更新索引为8的将会进入索引为9的单元.但是当从索引为9(数组的最后一个单元)处更新时,需要计算(9+1)%10,其结果为得到索引为0的位置
        self._front = (self._front + 1) % len(self._data)
        self._size -= 1
        # 当所存储的元素降低到数组总储存能力的1/4时,一个健壮的方法是将这个数组大小缩减到当前容量的一半
        if 0 < self._size < len(self._data) // 4:
            self._resize(len(self._data) // 2)
        return answer

    # O(1)
    # 向队列的队尾添加一个元素(循环使用数组)
    def enqueue(self, e):
        """Add an element to the back of queue"""
        if self._size == len(self._data):
            self._resize(2 * len(self._data))
        # 下一个插入的位置
        # 例如,如果一个储存容量为10的队列,假设的队列有3个元素并且第一个元素在索引8的位置上,通过计算(8+3)%10=1.
        # 这样的结果完全正确,因为3个现有的元素占据索引为8,9和0对应的位置.
        pos = (self._front + self._size) % len(self._data)
        self._data[pos] = e
        self._size += 1

    def _resize(self, cap):
        """Resize to a new list of capacity >= len(self)"""
        old = self._data  # keep track of existing list
        self._data = [None] * cap  # allocate list with new capacity
        walk = self._front
        for k in range(self._size):  # only consider existing elements
            self._data[k] = old[walk]  # intensionally shift indics
            walk = (1 + walk) % len(old)  # use old size as modulus
        self._front = 0  # font has been realigned(首部索引调整为0)

    def printqueue(self):
        for i in range(self._size):
            pos = (self._front + self._size - 1 - i) % len(self._data)  # 队列的最后一个元素(最后一个进入的元素)
            print(self._data[pos], end=" ")
        print()


if __name__ == '__main__':
    myqueue = ArrayQueue()
    print('size was: ', str(len(myqueue)))
    myqueue.enqueue(1)
    myqueue.enqueue(2)
    myqueue.enqueue(3)
    myqueue.enqueue(4)
    myqueue.enqueue(5)
    print('size was: ', str(len(myqueue)))
    myqueue.printqueue()

    myqueue.dequeue()
    myqueue.dequeue()
    print('size was: ', str(len(myqueue)))
    myqueue.printqueue()

    myqueue.enqueue(6)
    myqueue.enqueue(7)
    myqueue.printqueue()
    myqueue.dequeue()
    myqueue.dequeue()
    print('size was: ', str(len(myqueue)))

    myqueue.printqueue()
    myqueue.dequeue()
    myqueue.dequeue()
    myqueue.dequeue()
    print('size was: ', str(len(myqueue)))
    myqueue.printqueue()

    myqueue.dequeue()
