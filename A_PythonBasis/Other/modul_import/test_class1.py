from dir_test.test2 import print_hello


class Rect_volumes:
    def __init__(self, area, height):
        self.area = area
        self.height = height

    def volumes(self):
        vo = self.area.area() * self.height
        return vo

    def print_hello(self):
        print_hello("java")
