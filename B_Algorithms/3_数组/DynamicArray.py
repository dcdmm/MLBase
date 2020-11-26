
import ctypes

# Python的list类给出了动态数组的一种高度优化的实现
class DynamicArray:
    """A dynamic array akin to a simplified Python list"""
    def __init__(self):
        """Create an empty aray"""
        self._n = 0 # count actual elements
        self._capacity = 1 # default array capacity
        self._A = self._make_array(self._capacity) # low-level array

    def __len__(self):
        return self._n

    def __getitem__(self, k):
        """Return element at index k"""
        if not 0 <= k <= self._n:
            raise IndexError('invalid index')
        return self._A[k] # retrieve form array

    def append(self, obj):
        """Add object to end of array"""
        if self._n == self._capacity: # not emought room
            self._resize(2 * self._capacity) # so double capacity
        self._A[self._n] = obj
        self._n += 1

    def insert(self, k, value):
        """Insert value at index k, shifting subsequent vaues rightward"""
        if self._n == self._capacity:
            self._resize(2 * self._capacity)
        for j in range(self._n, k, -1):
            self._A[j] = self._A[j-1]
        self._A[k] = value
        self._n += 1

    def remove(self, value):
        """Remove first occuruence of value (or rasie ValueError)"""
        for k in range(self._n):
            if self._A[k] == value: # found a match
                for j in range(k, self._n-1):
                    self._A[j] = self._A[j+1]
                self._A[self._n - 1] = None
                self._n -= 1
                return
        raise ValueError('value not found')

    def pop(self, index=None):
        if index is None:
            index = self._n - 1
        """Remove and return item at index (default last)"""
        if not 0 <= index <= self._n:
            raise IndexError('invalid index')
        result = self._A[index]
        for k in range(index, self._n-1):
            self._A[k] = self._A[k+1]
        self._A[self._n - 1] = None
        self._n -= 1

        return result

    def _resize(self, c):
        """Resize internal array to capacity c"""
        B = self._make_array(c) # new (bigger) array
        for k in range(self._n): # for each existing value
            B[k] = self._A[k]
        self._A = B # use the bigger array
        self._capacity = c

    @staticmethod
    def _make_array(c):
        """Return new array with capacity c"""
        return (c * ctypes.py_object)()

    def print_array(self):
        for k in range(self._n):
            print(self._A[k], end=' ')
        print()


if __name__ == '__main__':
    # 测试上述方法
    my_array = DynamicArray()

    for i in range(10):
        my_array.append(i)
    my_array.print_array()

    my_array.insert(3, -1)
    my_array.print_array()

    my_array.remove(-1)
    my_array.print_array()

    my_array.pop()
    my_array.print_array()

    print(my_array.pop(0))
    my_array.print_array()
