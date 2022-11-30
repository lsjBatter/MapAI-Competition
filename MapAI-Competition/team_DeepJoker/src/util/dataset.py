import os
import cv2
import torch
import numpy as np
import albumentations as A
from torch.utils.data import Dataset
from torchvision import transforms
import random
from PIL import Image
import numpy as np

def onehot(data,n):
    data = data/255
    data = data.astype('uint8')
    buf = np.zeros(data.shape + (n, ))
    nmsk = np.arange(data.size)*n + data.ravel()
    buf.ravel()[nmsk-1] = 1
    buf = buf.transpose(2,0,1) #不经过transform处理，所以要手动把(H,W,C)转成(C,H,W)
    return buf

class MpiAIDataset(Dataset):
    def __init__(self,
                 img_dir,
                 gt_dir,
                 crop_size,
                 train:bool = None,
                 lidar:bool = None):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.lidar = lidar
        self.img_name = os.listdir(self.img_dir)
        random.seed(42)
        random.shuffle(self.img_name)

        self.train_im_nms = self.img_name[:int(len(self.img_name)*(1-0.2))]
        self.vaild_im_nms = self.img_name[int(len(self.img_name)*(1-0.2)):]

        self.crop_size = crop_size
        self.train = train
        self.palette = []
    def crop(self,im,mask):
        trans = A.RandomCrop(width=self.crop_size, height=self.crop_size)
        transed = trans(image=im,mask=mask)
        img,gt = transed['image'],transed['mask']
        return img,gt

    def Rotate(self,im,mask):
        trans =  A.RandomRotate90(90)
        transed = trans(image=im,mask=mask)
        img,gt = transed['image'],transed['mask']
        return img,gt  

    def Filp(self,im,mask):
        trans =  A.HorizontalFlip(p=0.5)
        transed = trans(image=im,mask=mask)
        img,gt = transed['image'],transed['mask']

        return img,gt    
    def __len__(self):
        if self.train:
            return len(self.train_im_nms)
        else:
            return len(self.vaild_im_nms)

    def __getitem__(self,idx):
        if self.train:
            img = cv2.imread(self.img_dir + '/' + self.train_im_nms[idx],-1)
            gt = cv2.imread(self.gt_dir + '/'+ self.train_im_nms[idx].split('.')[0]+'.png',0)

            img,gt = self.crop(img,gt)
            if self.lidar is not None:
                img,gt = self.Filp(img,gt)
            else:
                img,gt = self.Rotate(img,gt)
        else: 
            img = cv2.imread(self.img_dir + '/' + self.vaild_im_nms[idx],-1)
            gt = cv2.imread(self.gt_dir + '/'+ self.vaild_im_nms[idx].split('.')[0]+'.png',0)
            img,gt = self.crop(img,gt)
            
        img = img/255.0
        if self.lidar is not None:
            img = torch.FloatTensor(img).unsqueeze(0)
        else:
            img = torch.FloatTensor(img.transpose(2,0,1)) 
        gt_onehot = onehot(gt, 2) 
        gt_onehot = torch.FloatTensor(gt_onehot)
        if self.train:
            return img,gt_onehot
        else:
            return img,gt_onehot,self.vaild_im_nms[idx]

def colorful(img,save_path):
    img = Image.fromarray(img) #将图像从numpy的数据格式转为PIL中的图像格式
    palette=[]
    for i in range(256):
        palette.extend((i,i,i))
    palette[:3*21]=np.array([[0, 0, 0],
                            [255,255,255]
                                ], dtype='uint8').flatten()

    img.putpalette(palette)
    img.save(save_path)
    
class MpiAItest(Dataset):
    def __init__(self,
                 img_dir,
                 gt_dir):
        self.img_dir = img_dir
        self.gt_dir = gt_dir
        self.name = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.name)

    def __getitem__(self,idx):
        img = cv2.imread(self.img_dir + '/' + self.name[idx],-1)
        gt = cv2.imread(self.gt_dir + '/'+ self.name[idx].split(".")[0]+'.png',0)
        return img,gt,self.name[idx]

def generate_data(img_dir,
                  gt_dir,
                  crop_size,
                  train:bool = True,
                  lidar:bool = None):
    mpi = MpiAIDataset(img_dir,gt_dir,crop_size,train,lidar)
    if train :
        train_dataset = len(mpi)
        print(f'Number of training : {train_dataset}') 
        return mpi
    else:
        vaild_dataset = len(mpi)
        print(f'Number of vailding : {vaild_dataset}')
        return mpi

if __name__ == "__main__":
    img_dir = '/boot/data1/Li_data/data/competition/MapAI-Competition/train/images'
    gt_dir = '/boot/data1/Li_data/data/competition/MapAI-Competition/train/label'
    save_path = '/boot/data1/Li_data/data/competition/MapAI-Competition/validation/label'
    crop_size = 224
    
    vaild_dataset = generate_data(img_dir,gt_dir,crop_size,train=True,lidar=None)
    a,b= vaild_dataset[0]
    print(type(a),a.shape)
    print(type(b),b.shape)
    # # img = Image.fromarray(c)
    # # img.save(save_path)
  
    # test_dataset = MpiAItest(img_dir,gt_dir)
    # for i in range(len(test_dataset)):
    #     gt,name = test_dataset[i]
    #     colorful(gt, save_path + '/' + name.split(".")[0]+'.png')
    #     # cv2.imwrite(save_path + '/' + name.split(".")[0]+'.png',gt)




