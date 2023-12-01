"""python模块文档
base_rect_area.py"""

class Rect_Area:
    def __init__(self, length, width):
        self.width = width
        self.length = length

    def area(self):
        rect = self.width * self.length * 3
        return rect


if __name__ == '__main__':
    a = Rect_Area(3, 4)
    print(a.area())
