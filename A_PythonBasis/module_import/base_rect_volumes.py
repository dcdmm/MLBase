print('base_rect_volumes: __file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__, __name__,
                                                                                            str(__package__)))
from base.print_hello import print_hello


class Rect_volumes:
    def __init__(self, area, height):
        self.area = area
        self.height = height

    def volumes(self):
        vo = self.area.area() * self.height
        return vo

    def print_hello(self):
        print_hello("java")
