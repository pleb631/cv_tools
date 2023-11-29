import argparse
import sys
from PySide6.QtCore import * 
from PySide6.QtGui import * 
from PySide6.QtWidgets import *


from config import get_config
from app import MainWindow

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--version", "-V", action="store_true", help="show version"
    )
    args = parser.parse_args()

    config_from_args = args.__dict__
    config = get_config(config_from_args=config_from_args)
    app = QApplication(sys.argv)
    win = MainWindow(
        config=config,
)
    
    win.show()
    win.raise_()
    sys.exit(app.exec())


# this main block is required to generate executable by pyinstaller
if __name__ == "__main__":
    main()
