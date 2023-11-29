# -*- coding: utf-8 -*-

from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from .KPoint import KPoint


class NumericItem(QTableWidgetItem):
    pass
    # 定义一个QTableWidgetItem的子类，用来存储数字类型的值
    # def __lt__(self, other):
    #     # 重写__lt__方法，让它能够把值转换成浮点数来比较
    #     try:
    #         return float(self.text()) < float(other.text())
    #     except ValueError:
    #         return super().__lt__(other)


class labelListView(QTableWidget):
    labelChanged = Signal(tuple)

    def __init__(self):
        super(labelListView, self).__init__()
        self.initUI()

        self.__points = []
        self.adjustRow = -1
        self.Clickedrow = -1

        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_menu)
        self.setSortingEnabled(False)

        self.setDragDropMode(QAbstractItemView.InternalMove)  # 允许内部移动
        self.setSelectionMode(QAbstractItemView.SingleSelection)  # 选择一行
        self.setSelectionBehavior(QAbstractItemView.SelectRows)  # 选择整行
        self.setDragDropOverwriteMode(False)

    def initUI(self):
        self.setRowCount(0)
        self.setColumnCount(3)
        self.setColumnWidth(0, self.width()*0.1)
        self.setColumnWidth(1, self.width()*0.15)
        self.setColumnWidth(2, self.width()*0.15)
        self.setHorizontalHeaderLabels(["名称", "X坐标", "Y坐标"])
        self.selected = -1
        self.itemChanged.connect(self.onItemChanged)
        self.cellPressed.connect(self.onCellPressed)

        self.menu = QMenu(self)
        self.delete_action = QAction("删除", self)
        self.delete_action.triggered.connect(self.delete_row)
        self.delete_action.setEnabled(True)

        self.insert_action = QAction("插入(空行)", self)
        self.insert_action.triggered.connect(self.insert_row)

        self.deleteAll_action = QAction("删除(全部)", self)
        self.deleteAll_action.triggered.connect(self.deleteAll)

        self.adjust_action = QAction("修改", self)
        self.adjust_action.triggered.connect(self.adjust)

        self.menu.addAction(self.delete_action)
        self.menu.addAction(self.deleteAll_action)
        self.menu.addAction(self.insert_action)
        self.menu.addAction(self.adjust_action)

    def dropEvent(self, event):
        if event.source() == self and (
            event.dropAction() == Qt.MoveAction
            or self.dragDropMode() == QAbstractItemView.InternalMove
        ):
            row: int = self.rowAt(event.pos().y())
            source_row: int = self.currentRow()
            if row == source_row or source_row - 1 == row:
                return True
            self.insertRow(row + 1)
            if row < source_row:
                self.__points.insert(row + 1, self.__points[source_row].reset())
                self.__points.pop(source_row + 1)
                for col in range(self.columnCount()):  # 遍历每一列
                    item: QTableWidgetItem = self.takeItem(
                        source_row + 1, col
                    )  # 取出原来的项目
                    self.setItem(row + 1, col, item)  # 设置新的项目
                self.removeRow(source_row + 1)
                self.setCurrentCell(row + 1, 0)

            if row > source_row:
                self.__points.insert(row + 1, self.__points[source_row].reset())
                self.__points.pop(source_row)
                for col in range(self.columnCount()):
                    item = self.takeItem(source_row, col)
                    self.setItem(row + 1, col, item)
                self.removeRow(source_row)
                self.setCurrentCell(row, 0)

            self.manualUpdate()

        return True

    def setCurrentCell(self, row, col):
        self.onCellPressed(row, col)
        super().setCurrentCell(row, col)

    @Slot()
    def setPoint(self, pointf: QPointF):
        if self.adjustRow >= 0:
            point: KPoint = self.__points[self.adjustRow]
            point.setkpoint(pointf)
            self.__points[self.adjustRow] = point
            insertrow: int = self.adjustRow
        else:
            point = KPoint(self.rowCount(), [pointf.x(), pointf.y()])
            self.__points.append(point)
            self.insertRow(self.rowCount())
            insertrow = self.rowCount() - 1

        info = point.get_view_list()
        for col in range(self.columnCount()):
            self.setItem(insertrow, col, NumericItem(str(info[col])))
        self.setCurrentCell(insertrow, 0)
        self.manualUpdate()

    def onCellPressed(self, row, col):
        if self.Clickedrow >= 0:
            try:
                self.__points[self.Clickedrow].setCliked(False)
            except:
                self.Clickedrow = -1

        self.Clickedrow = row
        self.__points[row].setCliked(True)
        self.manualUpdate()

    def onItemChanged(self, item):
        row = item.row()
        col = item.column()
        if row >= len(self.__points):
            return
        self.blockSignals(True)
        if col == 0 and len(self.__points) > 0:
            self.__points[row].setclass_name(item.text())
        self.blockSignals(False)

    @Slot(QPoint)
    def show_menu(self, pos: QPoint):
        global_pos: QPoint = self.mapToGlobal(pos)
        row: int = self.rowAt(pos.y())
        if row == -1:
            return

        self.menu.exec(global_pos)

    def delete_row(self):
        row: int = self.currentRow()
        if row == -1:
            return

        self.__points.pop(row)
        self.removeRow(row)
        self.manualUpdate()

    def insert_row(self):
        row: int = self.currentRow()
        if row == -1:
            return

        self.__points.insert(row + 1, KPoint(-1, [-1, -1]))
        self.insertRow(row + 1)
        for col in range(self.columnCount()):
            self.setItem(row + 1, col, NumericItem(str(-1)))

        self.manualUpdate()

    def deleteAll(self):
        self.__points.clear()
        for row in range(self.rowCount() - 1, -1, -1):
            self.removeRow(row)

        self.manualUpdate()

    def adjust(
        self,
    ):
        if self.adjust_action.text() == "修改":
            self.adjust_action.setText("取消(修改)")
            self.delete_action.setEnabled(False)
            self.deleteAll_action.setEnabled(False)
            self.insert_action.setEnabled(False)
            row = self.currentRow()
            if row == -1:
                return
            self.setCurrentCell(row, 0)
            self.adjustRow = row

        else:
            self.delete_action.setEnabled(True)
            self.deleteAll_action.setEnabled(True)
            self.insert_action.setEnabled(True)
            self.adjust_action.setText("修改")
            self.adjustRow = -1

    def get_points(self):
        return self.__points

    def set_points(self, points):
        self.__points = points

        for i in range(len(points)):
            info = points[i].get_view_list()
            self.insertRow(i)
            for col in range(self.columnCount()):
                self.setItem(i, col, NumericItem(str(info[col])))
        self.manualUpdate()

    def manualUpdate(
        self,
    ):
        self.labelChanged.emit(self.__points)

    def setlabel(self, points):
        self.deleteAll()
        if points:
            self.__points = points.copy()
        self.set_points(self.__points)
