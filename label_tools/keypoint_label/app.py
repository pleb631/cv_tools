from PySide6.QtCore import * 
from PySide6.QtGui import * 
from PySide6.QtWidgets import *
from parts import *
import cv2
from pathlib import Path
import json
import numpy as np

def save_json(json_path, info, indent=4, mode='w', with_return_char=False):

    json_str = json.dumps(info, indent=indent)
    if with_return_char:
        json_str += '\n'
    
    with open(json_path, mode,encoding="UTF-8") as json_file:
        json_file.write(json_str)
    
    json_file.close()
    
    
def read_json(json_path, mode='all'):
    json_data = []
    with open(json_path, 'r',encoding="UTF-8") as json_file:
        if mode == 'all':
            # 把读取内容转换为python字典
            json_data = json.loads(json_file.read())
        elif mode == 'line':
            for line in json_file:
                json_line = json.loads(line)
                json_data.append(json_line)

    return json_data

class MainWindow(QWidget):
    def __init__(self, config,**kwargs) -> None:
        super().__init__(**kwargs)
        self.config = config
    
        self.screenRect = QApplication.primaryScreen().geometry()
        self.screenheight = self.screenRect.height()
        self.screenwidth = self.screenRect.width()
        
        self.resize(int(self.screenwidth),int(self.screenheight))
        self.setWindowTitle("keypoint label_tool")
        self.setWindowIcon(QIcon("./icon/icon.png"))
        self.setup_ui()
        
        self.current_image = None
        self.current_path = None
    
    
    def setup_ui(self):
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(10)
        self.list_img_part = fileListView(parent=self)
        self.list_img_part.setObjectName("list_img_part")
        self.label_view_part = labelListView()
        
        
        self.list_img_part.setFixedWidth(int(self.screenwidth*0.2))
        #list_img_part.setFixedHeight(int(self.screenheight*0.5))
        
                   
        self.label_view_part.setFixedWidth(int(self.screenwidth*0.2))
        #self.label_view_part.setFixedHeight(int(self.screenheight*0.6))
        
           
        self.box = ImageBox()
        self.box.setFixedWidth(int(self.screenwidth*0.8))
        self.box.setFixedHeight(int(self.screenheight*0.8))

        self.list_img_part.itemDoubleClicked.connect(self.process_image_path)
        self.box.generatePoint.connect(self.label_view_part.setPoint)
        self.label_view_part.labelChanged.connect(self.box.setLabel)
        
        self.top_menu_part = topMenu()
        self.top_menu_part.save_action.triggered.connect(self.save_single_label)
        self.top_menu_part.submitpath.connect(self.list_img_part.loadFileList)
        
        self.layout.addWidget(self.top_menu_part,0,0,1,1)
        self.layout.addWidget(self.list_img_part,1,0,1,1)
        self.layout.addWidget(self.label_view_part,2,0,1,1)
        self.layout.addWidget(self.box,1,1,3,1)
        
        
        
        
        
    @Slot(str)
    def process_image_path(self,path):
        
        image = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
        if not self.current_image is None:
            self.save_single_label()
        self.current_image = image
        self.current_path = path
        self.set_image(self.current_image)
        label = self.load_label(path)
        self.label_view_part.setlabel(label)
    
    
    def set_image(self,image):

        #image = cv2.resize(image,(int(self.screenwidth*0.7),int(self.screenheight*0.7)))
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        
        height,width,c= image.shape
        image = QImage(image.data, width, height,c * width,QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.box.set_pixmap(pixmap)

    
    def load_label(self,path):
        save_path = Path(path).with_suffix('.json')
        if save_path.exists():
            points=[]
            labels = read_json(save_path)
            for label in labels:
                name = label['name']
                xy = label['label']
                xy = list(map(float,xy))
                point = KPoint(name,xy)
                points.append(point)
            return points
        return None

    
    
    def save_single_label(self,):
        points = self.label_view_part.get_points()
        #im = cv2.imread(self.current_path)
        labels=[]
        for point in points:
            xy = point.xy
            xy = list(map(str,xy))
            name = point.class_name
            labels.append({"name":name,"label":xy})
        save_path = Path(self.current_path).with_suffix('.json')
        save_json(save_path,labels)
        
        # im = cv2.imread(self.current_path)
        # for point in points:
        #     xy = point.xy
        #     im=cv2.circle(im,(int(xy[0]),int(xy[1])),5,(0,255,0),-1)
        # cv2.imshow("im",im)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    
    
        
