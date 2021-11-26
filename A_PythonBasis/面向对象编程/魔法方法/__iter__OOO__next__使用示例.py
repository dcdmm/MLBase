class MyRange(object):
    def __init__(self, end):
        self.start = 0
        self.end = end

    def __iter__(self):
        print('进入__iter__方法')
        return self

    def __next__(self):
        print('进入__next__方法')
        if self.start < self.end:
            ret = self.start
            self.start += 1
            return ret
        else:
            raise StopIteration


if __name__ == '__main__':
    a = MyRange(5)

    for i in a:
        print(i)