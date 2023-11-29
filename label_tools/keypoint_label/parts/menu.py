# -*- coding: utf-8 -*-
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *


class topMenu(QMenuBar):
    submitpath = Signal(tuple)
    def __init__(self):
        super(topMenu, self).__init__()
        
        file_menu: QMenu = self.addMenu('&文件')


        openFile_action = QAction(QIcon('open.png'), '&打开(文件)', self)
        openFile_action.setShortcut('Ctrl+O')

        openFile_action.triggered.connect(self.open_file)
                
        openDir_action = QAction(QIcon('open.png'), '&打开(文件夹)', self)
        openDir_action.setShortcut('Ctrl+P')

        openDir_action.triggered.connect(self.open_dir)
        
        
        self.save_action = QAction(QIcon('save.png'), '&保存', self)
        self.save_action.setShortcut('Ctrl+S')
        


        exit_action = QAction(QIcon('exit.png'), '&退出', self)
        exit_action.setShortcut('Ctrl+Q')

        exit_action.triggered.connect(QApplication.quit)

        # 在文件菜单中添加打开文件和退出程序的动作，并用分隔线隔开
        file_menu.addAction(openFile_action)
        file_menu.addAction(openDir_action)
        file_menu.addAction(self.save_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)
        
        
    def open_file(self):

        file_name, _ = QFileDialog.getOpenFileNames(self, '选择图片', '.', '所有文件 (*.*)')
        if file_name:
            print(f'打开了文件：{file_name}')
        self.submitpath.emit(file_name)
            
    
    
    def open_dir(self):

        file_name: str = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if file_name:
            print(f'打开了文件：{file_name}')
        self.submitpath.emit(file_name)