from PySide6.QtCore import QPointF


class KPoint:
    def __init__(
        self,
        name: str,
        kpoint: QPointF | list,
    ):
        self.__kpoint = kpoint

        if isinstance(kpoint, QPointF):
            kpoint = [kpoint.x(), kpoint.y()]

        self.__x = kpoint[0]
        self.__y = kpoint[1]
        self.__class_name = name
        self.ClikedFlag = False

    def setkpoint(self, kpoint: QPointF | list):
        self.__kpoint = kpoint
        if isinstance(kpoint, QPointF):
            kpoint = [kpoint.x(), kpoint.y()]
        self.__x = kpoint[0]
        self.__y = kpoint[1]

    def setclass_name(self, name:str):
        self.__class_name = name

    def getkpoint(self):
        return self.__kpoint

    @property
    def x(self):
        return self.__x

    @property
    def y(self):
        return self.__y

    @property
    def xy(self):
        return [self.__x, self.__y]

    def pointf(self):
        return QPointF(self.__x, self.__y)

    @property
    def class_name(self):
        return self.__class_name

    def get_view_list(self):
        return [self.__class_name, self.__x, self.__y]

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def setCliked(self, flag):
        self.ClikedFlag = flag

    def reset(self):
        self.ClikedFlag = False
        return self
