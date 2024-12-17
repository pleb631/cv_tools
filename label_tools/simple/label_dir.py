import tkinter as tk
from PIL import ImageTk, Image
import cv2 
import numpy as np
import os
from functools import partial
import  argparse
import glob
import os


def write_label(checkpoint_path,content):

    checkpoint_file = open(checkpoint_path, "w")
    checkpoint_file.writelines(content)

def read_label(checkpoint_path):
    with open(checkpoint_path, "r") as f:
        return f.readline()
        

        

class markf():
    def __init__(self,args):
        self.args=args
        self.img_root = args.dataset_folder
        dirpaths = []
        for root1,dir1,file1 in os.walk(self.img_root):
            if len(file1)>0:
                dirpaths.append(root1)
        dirpaths = sorted(dirpaths)
        self.dirlist = dirpaths

        self.cache =[]
        self.pointer = 0
        self.imgid = 0
        self.curdir = self.dirlist[0]
        
        self.root = tk.Tk()
       

        
        tk.Button(self.root, text="前一组", command=partial(self.change_MatchIndex, -1)).grid(row=2,column=1, sticky="w")
        tk.Button(self.root, text="后一组", command=partial(self.change_MatchIndex, 1)).grid(row=3,column=1, sticky="w")
    
        
        tk.Button(self.root, text="标记", command=self.label).grid(row=2,column=6, sticky="w")
        #tk.Button(self.root, text="取消合并", command=partial(self.recode, 0)).grid(row=2, sticky="w")
         
        self.root.bind("<KeyPress-a>", lambda event: self.change_imgindex(-1))
        self.root.bind("<KeyPress-d>", lambda event: self.change_imgindex(1))
        self.root.bind("<KeyPress-w>", lambda event: self.change_MatchIndex(-1))
        self.root.bind("<KeyPress-s>", lambda event: self.change_MatchIndex(1))
        self.root.bind("<KeyPress-q>", lambda event: self.label())
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.bind("<KeyPress-Escape>", lambda event: self.root.destroy())
        
        self.image_Label0 = tk.Label(self.root)
        self.image_Label0.grid(row=2, column=2, rowspan=2, columnspan=2,padx='4px', pady='5px')
        self.image_Label0.bind("<MouseWheel>", func=self.processWheel)
        
        
        self.la2=tk.Label(self.root,text=f'当前位置:{self.pointer}/{len(self.dirlist)}')
        self.la2.grid(row=0,column=5, sticky="w")
        
        self.la1=tk.Label(self.root,text=f'标记数:{len(self.cache)}/2')
        self.la1.grid(row=0,column=4, sticky="w")
        self.start()
        
    def on_closing(self):
        # 在这里添加你希望在退出时执行的代码
        print("程序正在退出...")
        self.root.destroy()
    
    def sortlist(func):    # func接收body
        def ware(self, *args, **kwargs):    # self,接收body里的self,也就是类实例

            result = func(self, *args, **kwargs)
            if len(self.cache)>1:
                self.cache = sorted(self.cache)
            self.update_tk()
            self.show_img()
            return result

        return ware
    
    @sortlist
    def label(self):
        if self.imgid in self.cache:
            self.cache.remove(self.imgid)
        else:
            if len(self.cache)<2:
                self.cache.append(self.imgid)
            
           
    def draw(self,img):
        if self.imgid not in self.cache:
            return img
        
        text=f"biaoji-{self.cache.index(self.imgid)}"
        img = cv2.putText(img, str(text), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0),5)

        return img
    
    
    def save_current_label(self,):
        if len(self.cache)==0:
            return
        dirpath = self.dirlist[self.pointer]
        label_path = os.path.join(dirpath,'label')
        content = ' '.join([str(i) for i in self.cache])
        write_label(label_path,content)
        self.cache =[]
    
    def loadAnno(self,):
        dirpath = self.dirlist[self.pointer]
        label_path = os.path.join(dirpath,'label')
        if not os.path.exists(label_path):
            return
        content = read_label(label_path)
        content = content.split()
        self.cache = [int(i) for i in content]
    
    def change_MatchIndex(self,tar):
        
        if tar==-1:
            if self.pointer==0:
                return 0
            self.save_current_label()
            self.pointer-=1
        elif tar ==1:
            if self.pointer==(len(self.dirlist)-1):
                return 0
            self.save_current_label()
            self.pointer+=1
        self.curdir = self.dirlist[self.pointer]
        
        self.loadImgPathList()
        self.loadAnno()
        self.show_img()
        self.update_tk()
        
        
    def processWheel(self,event,):
        if event.delta>0:
            self.change_imgindex(-1)
        else:
            self.change_imgindex(1)
    def show_img(self,draw=False):
        

        img0 = cv2.imdecode(np.fromfile(self.img_list0[self.imgid], dtype=np.uint8), -1)
        img0 = self.draw(img0)

        target_width, target_height = 640, 640
        img_width, img_height = img0.shape[1], img0.shape[0]
        aspect_ratio = img_width / img_height

        if img_width > img_height:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            new_height = target_height
            new_width = int(target_height * aspect_ratio)

        # 调整图片尺寸
        img0 = cv2.resize(img0, (new_width, new_height))
        self.photo0 = ImageTk.PhotoImage(Image.fromarray(img0[:,:,::-1]))

        self.image_Label0.configure(image=self.photo0)

        
    def __del__(self):
        self.save_current_label()
        self.save_anno()
            
    def change_imgindex(self,tar):
        if tar==-1:
            if self.imgid==0:
                return 0
            self.imgid-=1
        elif tar ==1:
            if self.imgid==(len(getattr(self,f'img_list0'))-1):
                return 0
            self.imgid+=1

               
        self.show_img(draw=True)
            
    def start(self,):
        self.read_anno()
        
        self.curdir = self.dirlist[self.pointer]
        self.loadImgPathList()
        self.loadAnno()
        self.update_tk()
        self.show_img(True)
        self.root.mainloop()
    
    def loadImgPathList(self):
        self.imgid = 0
        self.img_list0 = glob.glob(os.path.join(self.img_root,self.curdir,'*.jpg'))+glob.glob(os.path.join(self.img_root,self.curdir,'*.png'))
        self.img_list0 = sorted(self.img_list0)

        
        
    def get_next_img(self,):
        self.change_img(tar=1)
        self.CurImPath = self.imglist[self.pointer]
        
        
    def write_checkpoint(self, checkpoint_path):
        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        checkpoint_file = open(checkpoint_path, "w")
        checkpoint_file.writelines(str(self.pointer))

    def read_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            return -1
        with open(checkpoint_path, "r") as f:
            content = f.readline().strip()
        
        pointer = content
        self.pointer = int(pointer)
        

        
    def save_anno(self,):

        self.write_checkpoint(os.path.join(self.img_root,'.checkpoint'))

    
    def read_anno(self,):
        
        if not os.path.exists(os.path.join(self.img_root,'.checkpoint')):
            return

        self.read_checkpoint(os.path.join(self.img_root,".checkpoint"))
    
    def update_tk(self):

        self.la2['text']=f'当前位置:{self.pointer+1}/{len(self.dirlist)}'
        self.la1['text']=f'标记数:{len(self.cache)}/2'
     
     
     

    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", type=str, default=r'F:\zipall\im',help="the path of reviewed images")

    args = parser.parse_args()
    markf(args)


if __name__ == '__main__':
    main()