import tkinter as tk
from PIL import ImageTk, Image
import cv2 
import numpy as np
import os
from functools import partial
import time
import pickle
import  argparse
import glob
from pathlib import Path
from io import BytesIO


def read_pkl(pkl_path):
    '''读取txt文件

    Args:
        pkl_path: str, pickle文件路径

    Returns:
        pkl_data: list, txt文件内容
    '''
    with open(pkl_path, 'rb') as pkl_file:
        pkl_data = pickle.load(pkl_file)

    return pkl_data


def read_txt(txt_path):

    txt_file = open(txt_path, "r",encoding="UTF-8")
    return [line.replace('\n', '') for line in txt_file]


class markf():
    def __init__(self,args):
        self.args=args
        self.img_root = args.dataset_folder
        self.dirlist = glob.glob(os.path.join(args.dataset_folder,'*'))
        list_path = os.path.join(args.dataset_folder,"list.txt")
        if os.path.exists(list_path):
            items = read_txt(list_path)
            items = list([*i.split(':')[0].split(),int(i.split(':')[-1])] for i in items)
        self.items = items
        
        
        self.cache =set()
        self.pointer = 0
        self.imgid = [0,0]
        self.curMatch = self.items[0]
        
        self.root = tk.Tk()
        #self.checkpoint_path = os.path.join(args.dataset_folder,'.checkpoint')
        self.enable_encrypt  = args.encrypt
        
        tk.Button(self.root, text="前一组", command=partial(self.change_MatchIndex, -1)).grid(row=2,column=1, sticky="w")
        tk.Button(self.root, text="后一组", command=partial(self.change_MatchIndex, 1)).grid(row=3,column=1, sticky="w")
        
        
        tk.Button(self.root, text="前一张", command=partial(self.change_imgindex, -1,0)).grid(row=4,column=2, sticky="w")
        tk.Button(self.root, text="后一张", command=partial(self.change_imgindex, 1,0)).grid(row=4,column=3, sticky="w")
        
        tk.Button(self.root, text="前一张", command=partial(self.change_imgindex, -1,1)).grid(row=4,column=4, sticky="w")
        tk.Button(self.root, text="后一张", command=partial(self.change_imgindex, 1,1)).grid(row=4,column=5, sticky="w")
        
        tk.Button(self.root, text="合并", command=self.label).grid(row=2,column=6, sticky="w")
        #tk.Button(self.root, text="取消合并", command=partial(self.recode, 0)).grid(row=2, sticky="w")
        self.b=tk.Button(self.root, text="保存", command=self.save_anno)
        self.b.grid(row=3,column=6, sticky="w")

        
        self.image_Label0 = tk.Label(self.root)
        self.image_Label0.grid(row=2, column=2, rowspan=2, columnspan=2,padx='4px', pady='5px')
        self.image_Label0.bind("<MouseWheel>", func=partial(self.processWheel,0))
        
        self.image_Label1 = tk.Label(self.root)
        self.image_Label1.grid(row=2, column=4, rowspan=2, columnspan=2,padx='4px', pady='5px')
        self.image_Label1.bind("<MouseWheel>", func=partial(self.processWheel,1))
        
        
        self.la2=tk.Label(self.root,text=f'当前位置:{self.pointer}/{len(self.items)}')
        self.la2.grid(row=0,column=5, sticky="w")
        self.start()
    
    
    
    def label(self):
        if self.pointer in self.cache:
            self.cache.remove(self.pointer)
        else:
            self.cache.add(self.pointer)
            
        self.show_img()
        
        
    def draw(self,img):
        if self.pointer not in self.cache:
            return img
        
        text="merge"
        img = cv2.putText(img, str(text), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0),5)

        return img
    
    def change_MatchIndex(self,tar):
        if tar==-1:
            if self.pointer==0:
                return 0
            self.pointer-=1
        elif tar ==1:
            if self.pointer==(len(self.items)-1):
                return 0
            self.pointer+=1
        self.curMatch = self.items[self.pointer]
        
        self.loadImgPathList()
        self.show_img()
        self.update_tk()
        
        
    def processWheel(self,idx,event,):
        if event.delta>0:
            self.change_imgindex(-1,idx)
        else:
            self.change_imgindex(1,idx)
    def show_img(self,draw=False):
        

        img0 = cv2.imdecode(np.fromfile(self.img_list0[self.imgid[0]], dtype=np.uint8), -1)
        img0 = self.draw(img0)
        img1 = cv2.imdecode(np.fromfile(self.img_list1[self.imgid[1]], dtype=np.uint8), -1)

        self.photo0 = ImageTk.PhotoImage(Image.fromarray(img0[:,:,::-1]).resize((512,512)))
        self.photo1 = ImageTk.PhotoImage(Image.fromarray(img1[:,:,::-1]).resize((512,512)))
       
        

        self.image_Label0.configure(image=self.photo0)
        self.image_Label1.configure(image=self.photo1)
        

            
    def change_imgindex(self,tar,index):
        if tar==-1:
            if self.imgid[index]==0:
                return 0
            self.imgid[index]-=1
        elif tar ==1:
            if self.imgid[index]==(len(getattr(self,f'img_list{index}'))-1):
                return 0
            self.imgid[index]+=1

               
        self.show_img(draw=True)
            
    def start(self,):
        self.read_anno()
       
        self.curMatch = self.items[self.pointer]
        self.loadImgPathList()
        self.show_img(True)
        self.root.mainloop()
    
    def loadImgPathList(self):
        self.imgid = [0,0]
        self.img_list0 = glob.glob(os.path.join(self.img_root,self.curMatch[0],'*.jpg'))+glob.glob(os.path.join(self.img_root,self.curMatch[0],'*.png'))
        self.img_list1 = glob.glob(os.path.join(self.img_root,self.curMatch[1],'*.jpg'))+glob.glob(os.path.join(self.img_root,self.curMatch[1],'*.png'))
        
        
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
        self.update_tk()

        
    def save_anno(self,):
        outname = str(int(time.time()))
        self.write_checkpoint(os.path.join(self.img_root,'result', '.checkpoint'))
        
        self.b['text']="保存中,请勿关闭程序"
        pkllist = glob.glob(os.path.join(self.img_root,'result','*.pkl'))
        with open(os.path.join(self.img_root,'result', outname+'.pkl'), 'wb') as f:
            pickle.dump(self.cache, f)
        self.b['text']="保存"
        for file in pkllist:
            os.remove(file)
        
    
    def read_anno(self,):
        
        if not os.path.exists(os.path.join(self.img_root,'result')):
            return
        timelist = glob.glob(os.path.join(self.img_root,'result','*.pkl'))
        if len(timelist)==0:
            return
        maxtime = str(max([int(os.path.basename(f).split('.')[0]) for f in timelist]))
        
        self.cache = read_pkl(os.path.join(self.img_root,'result',maxtime+'.pkl'))
        self.read_checkpoint(os.path.join(self.img_root,"result",".checkpoint"))
    
    def update_tk(self,pointer=True):
        if pointer:
            self.la2['text']=f'当前位置:{self.pointer+1}/{len(self.items)}'
            
     
     
     

    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", type=str, default=r'D:\data',help="the path of reviewed images")
    parser.add_argument("--task", type=str, default="pedestrain", help="task name")
    parser.add_argument("--encrypt", action="store_true", help="whether to encrypt")
    parser.add_argument("--scale", action="store_true", help="whether to show original images")
    args = parser.parse_args()
    markf(args)


if __name__ == '__main__':
    main()