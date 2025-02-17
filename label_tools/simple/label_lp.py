import tkinter as tk
from PIL import ImageTk, Image
import cv2 
import numpy as np
import os
from functools import partial
import  argparse
import os
from pathlib import Path
from tkinter.messagebox import showerror
from hyperlpr import HyperLPR_plate_recognition

def write_label(checkpoint_path,content):

    checkpoint_file = open(checkpoint_path, "w",encoding='utf-8')
    checkpoint_file.writelines(content)

def read_label(checkpoint_path):
    with open(checkpoint_path, "r",encoding='utf-8') as f:
        return f.readline()

_chars: list[str] = [
        '京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉',
        '黑', '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂',
        '湘', '粤', '桂', '琼', '川', '贵', '云', '藏', '陕',
        '甘', '青', '宁', '新',"港","澳","使","领","学","警",
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
        'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
        'W', 'X', 'Y', 'Z'
    ]
              
def iscorrentlp(content):
    if len(content)<6 or len(content)>9:
        return False,"输入长度不对"
    for stri in content:
        if stri not in _chars:
            return False,"字符不在字符库中"
    return True,'ok'
        

class markf():
    def __init__(self,args):
        self.args=args
        self.img_root = args.dataset_folder
        items = list(Path(self.img_root).rglob("*.png"))
        self.items = items

        self.cache =[]
        self.pointer = 0
        self.curdir = self.items[0]

        
        self.root = tk.Tk()
       

        
        tk.Button(self.root, text="前一张", command=partial(self.change_MatchIndex, -1)).grid(row=2,column=1, sticky="w")
        tk.Button(self.root, text="后一张", command=partial(self.change_MatchIndex, 1)).grid(row=3,column=1, sticky="w")
        tk.Button(self.root, text="ai生成", command=self.start_async).grid(row=5,column=1, sticky="w")
        tk.Button(self.root, text="reset", command=self.reset_label).grid(row=6,column=1, sticky="w")
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.bind("<KeyPress-Escape>", lambda event: self.root.destroy())
        
        self.image_Label0 = tk.Label(self.root)
        self.image_Label0.grid(row=2, column=2, rowspan=8, columnspan=2,padx='4px', pady='5px')
        # self.image_Label0.bind("<MouseWheel>", func=self.processWheel)
        
        self.entry_var=tk.StringVar()
        tk.Entry(self.root,width=20,textvariable=self.entry_var,font=("Times 30 bold")).grid(row=0,column=3, sticky="w",ipadx=30,ipady=20)

        
        self.la2=tk.Label(self.root,text=f'当前位置:{self.pointer}/{len(self.items)}')
        self.la2.grid(row=0,column=1, sticky="w")

        self.root.bind("<MouseWheel>", func=self.processWheel)
        
        self.text = tk.Text(self.root,height=2,width=40,font=("Georgia 20"))
        self.text.grid(row=10, column=2, columnspan=2)
        self.text.insert("1.0","沪津冀豫苏浙皖渝晋蒙辽吉黑闽赣鲁鄂湘粤桂琼川贵云藏陕甘青宁新港澳使领学警")
        self.text.config(state='disabled')
        self.start()
    

    def reset_label(self,):
        self.entry_var.set('')
        
    def start_async(self,):
        image = cv2.imdecode(np.fromfile(self.items[self.pointer], dtype=np.uint8), 1)
        try:
            result = HyperLPR_plate_recognition(image)
        except:
            print("代码有问题，跳过")
            return 0
        if len(result)==0:
            print("识别不到")
            return
        result = sorted(result,key=lambda x:x[1],reverse=True)
        result = result[0]
        if result[1]>0.3 and iscorrentlp(result[0])[0]:
            self.entry_var.set(result[0])
        else:
            print(f"置信程度不够:{result[0]},conf: {result[1]}")
        
    def on_closing(self):
        # 在这里添加你希望在退出时执行的代码
        print("程序正在退出...")
        self.root.destroy()

    def save_current_label(self,):
        path = self.items[self.pointer]
        label_path = os.path.splitext(path)[0]+'-lp.txt'
        content = self.entry_var.get()
        content = content.upper()

        if len(content)==0:
            if os.path.exists(label_path):
                os.remove(label_path)
            return 0

        ok,message = iscorrentlp(content)
        if not ok:
            showerror("错误",message)
            return -1
        write_label(label_path,content)
        return 0
    
    def loadAnno(self,):
        path = self.items[self.pointer]
        print(path)
        label_path = os.path.splitext(path)[0]+'-lp.txt'
        if not os.path.exists(label_path):
            self.entry_var.set('')
            return
        content = read_label(label_path)
        content = content.strip()

        self.entry_var.set(content)
    
    def change_MatchIndex(self,tar):
        
        if tar==-1:
            if self.pointer==0:
                return 0
            if self.save_current_label():
                return 0
            self.pointer-=1
        elif tar ==1:
            if self.pointer==(len(self.items)-1):
                return 0
            if self.save_current_label():
                return 0
            self.pointer+=1
        self.curdir = self.items[self.pointer]
        

        self.loadAnno()
        self.show_img()
        self.update_tk()
        
        
    def processWheel(self,event,):
        if event.delta>0:
            self.change_MatchIndex(-1)
        else:
            self.change_MatchIndex(1)
            
    def show_img(self,draw=False):
        

        img0 = cv2.imdecode(np.fromfile(self.items[self.pointer], dtype=np.uint8), -1)
        # new_shape = 
        new_shape = [2*self.root.winfo_width()//3,2*self.root.winfo_height()//3] 
        if new_shape[1]<200:
            new_shape = [480,200]
        img_h,img_w = img0.shape[0], img0.shape[1]
        
        r = min(new_shape[0] / img_w, new_shape[1] / img_h)
        interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
        new_unpad = int(round(img_w * r)), int(round(img_h * r))
        if r != 1:
            img0 = cv2.resize(img0, new_unpad,interpolation=interp)
        dh = new_shape[1]-img0.shape[0]
        dw = new_shape[0]-img0.shape[1]
        dw /= 2  # divide padding into 2 sides
        dh /= 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        #top, bottom,left, right = 0,dh,0,dw
        img0 = cv2.copyMakeBorder(img0, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[114,114,114])

        
        self.photo0 = ImageTk.PhotoImage(Image.fromarray(img0[:,:,::-1]))

        self.image_Label0.configure(image=self.photo0)

        
    def __del__(self):
        self.save_current_label()
        self.save_anno()
            
            
    def start(self,):
        self.read_anno()
        
        self.curdir = self.items[self.pointer]

        self.loadAnno()
        self.update_tk()
        self.show_img(True)
        self.root.mainloop()
    

        
        
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

        self.la2['text']=f'当前位置:{self.pointer+1}/{len(self.items)}'
     
     
     

    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_folder", type=str, default=r'F:\data1218\新建文件夹\data\.system\test',help="the path of reviewed images")

    args = parser.parse_args()
    try:
        markf(args)
    except Exception as e:
        print(e)
        input("!")


if __name__ == '__main__':
    main()