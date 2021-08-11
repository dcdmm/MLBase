
class Rect_volumes:
    def __init__(self, area, height):
        self.area = area
        self.height = height

    def volumes(self):
        vo = self.area.area() * self.height
        return vo

