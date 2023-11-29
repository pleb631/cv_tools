from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
import glob
import os
from itertools import chain


patterns = [".png", ".jpg", ".jpeg", ".bmp"]


class fileListView(QListView):
    itemDoubleClicked = Signal(str)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.setup_ui()

    def setup_ui(self):
        self.slm = QStringListModel()  # 创建model模型

        self.setModel(self.slm)  # 为视图设置模型
        self.doubleClicked.connect(self.clicked_list)  # 会传出用户点击项的索引

        self.setFlow(QListView.TopToBottom)

    @Slot()
    def clicked_list(self, model_index: QModelIndex):
        # 弹出一个消息提升框，展示用户点击哪个项
        # QMessageBox.information(
        #     self, "QListView", "你选择了: " + self.data_list[model_index.row()]
        # )
        self.itemDoubleClicked.emit(self.data_list[model_index.row()])
        print("点击的是：" + str(model_index.row()))

    def loadFileList(self, path: str | None | list):
        if isinstance(path, str):
            if os.path.isdir(path):
                self.data_list = sorted(
                    chain(
                        *[
                            glob.glob(os.path.join(path, f"**/*{p}"), recursive=True)
                            for p in patterns
                        ]
                    )
                )
                self.slm.setStringList(self.data_list)
            elif os.path.isfile(path):
                if os.path.splitext(path)[1] in patterns:
                    self.data_list = [path]
                    self.slm.setStringList(self.data_list)
        elif isinstance(path, list):
            data_list = []
            for p in path:
                if os.path.splitext(p)[1] in patterns:
                    data_list.append(p)
            self.data_list = data_list
            self.slm.setStringList(self.data_list)
