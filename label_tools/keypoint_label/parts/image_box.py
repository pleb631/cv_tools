# -*- coding: utf-8 -*-

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
import numpy as np



class ImageBox(QWidget):
    generatePoint = Signal(tuple)

    def __init__(self):
        super(ImageBox, self).__init__()
        self.img = None
        self.point = QPointF(0, 0)
        self.scale = 1
        self._painter = QPainter()
        self.p = []
        self.centerpos = None

        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_menu)

        self.menu = QMenu(self)
        self.reset_action = QAction("reset", self)
        self.reset_action.triggered.connect(self.reset_viewing_angle)
        self.menu.addAction(self.reset_action)

        
        self.setMouseTracking(True)
    @Slot()
    def reset_viewing_angle(
        self,
    ):
        wh = self.img.size().toTuple()
        wh_max = [self.width(), self.height()]
        self.scale = min(wh_max[0] / wh[0], wh_max[1] / wh[1])
        new_size = (int(wh[0] * self.scale), int(wh[1] * self.scale))
        self.point = QPointF(
            wh_max[0] / 2 - new_size[0] / 2, wh_max[1] / 2 - new_size[1] / 2
        )
        self.repaint()

    @Slot(QPoint)
    def show_menu(self, pos: QPoint):
        if self.img is None:
            return
        global_pos: QPoint = self.mapToGlobal(pos)

        self.menu.exec(global_pos)

    def set_pixmap(self, pixmap: QPixmap):
        self.img: QPixmap = pixmap
        wh = self.img.size().toTuple()
        wh_max = [self.width(), self.height()]
        self.scale = min(wh_max[0] / wh[0], wh_max[1] / wh[1])
        new_size: tuple[int, int] = (int(wh[0] * self.scale), int(wh[1] * self.scale))
        # self.scale = 1
        self.p = []
        self.point = QPointF(
            wh_max[0] / 2 - new_size[0] / 2, wh_max[1] / 2 - new_size[1] / 2
        )
        self.repaint()

    @Slot()
    def setLabel(self, labelList: list):
        self.p = labelList
        self.repaint()
        ...

    def paintEvent(self, e):
        """
        receive paint events
        :param e: QPaintEvent
        :return:
        """
        if self.img:
            painter = self._painter

            painter.begin(self)

            img: QPixmap = self.img.copy()
            img = img.scaled(img.size() * self.scale)
            # img = img.scaled(
            #     QSize(self.width() * self.scale, self.height() * self.scale)
            # )
            painter.drawPixmap(self.point, img)
            
            if self.p:
                for p in self.p:

                    pen: QPen = QPen(Qt.red, 8) if p.ClikedFlag else QPen(Qt.green, 8)

                    pen.setCapStyle(Qt.RoundCap)
                    painter.setPen(pen)
                    painter.drawPoints([(p.pointf() * self.scale + self.point)])
                    
            
            if not self.centerpos is None:
                pen = QPen(Qt.DashLine)
                pen.setColor(Qt.black)
                painter.setPen(pen)


                painter.drawLine(self.centerpos.x(), 0, self.centerpos.x(), self.height())
                painter.drawLine(0, self.centerpos.y(), self.width(), self.centerpos.y())

            painter.end()
            self.update()

    def wheelEvent(self, event):
        
        ##modifiers = event.modifiers()
        ##if modifiers & Qt.AltModifier:
            angle = event.angleDelta() / 8  # 返回QPoint对象，为滚轮转过的数值，单位为1/8度
            
            angleY = angle.y()
            ratio = 1.1 if angleY > 0 else 0.9
            event_pos = np.array(event.position().toTuple())
            start_pos = np.array(self.point.toTuple())
            end_pos = (
                start_pos + np.array((self.img.width(), self.img.height())) * self.scale
                )

            if not (
                start_pos[0] < event_pos[0] < end_pos[0]
                and start_pos[1] < event_pos[1] < end_pos[1]
            ):
                return -1
            s = (event_pos - start_pos) * ratio

            self.point: QPointF = event.position() - QPoint(*s.tolist())
            self.scale *= ratio
            self.repaint()

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            # self.left_click = True
            start_pos = e.pos()
            keypoint = (start_pos.toPointF() - self.point) / self.scale
            if len(self.p) > 0 and keypoint == self.p[-1] or  (self.img is None):
                return 0
            self.generatePoint.emit(keypoint)
        
            self.centerpos = None
            self.setCursor(Qt.ArrowCursor)

    
    def mousePressEvent(self,e):
        if not self.img is None:
            self.setCursor(Qt.CrossCursor)
    
    def mouseMoveEvent(self, event):
        self.centerpos = event.pos()
        self.repaint()

    def leaveEvent(self,event):
        self.centerpos = None
        self.setCursor(Qt.ArrowCursor)
        self.repaint()
