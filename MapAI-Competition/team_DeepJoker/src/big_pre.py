from tqdm import tqdm, trange
import numpy as np
import math
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms as transforms
import cv2
import argparse
import ttach as tta
from PIL import Image
# sys.path.append('../../')
from util.dataset import MpiAItest
from util.eval_functions import iou,biou
from util.big_crop import TifCroppingArray,Result
from util.network import Net

class read_img:
    def __init__(self,dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx): 
        big_image , labels,filename = self.dataset[idx]
        return big_image,labels,filename

def array_list(array,args):
    list = []
    for i in range(len(array)):
        for j in range(len(array[0])):
            image = array[i][j]
            image = image / 255.0
            if args.lidar is not True:
                image = torch.FloatTensor(image.transpose(2,0,1))
            else:
                image = torch.FloatTensor(image).unsqueeze(0)
            list.append(image)
    return list

class img_path:
    def __init__(self,list):
        self.list = list
    def __len__(self):
        return len(self.list)
    def __getitem__(self, idx):
        return self.list[idx]

def pre_path(img,model,device,args):
    predicts = []
    pathLoader = DataLoader(img,batch_size=args.BATCH_SIZE,shuffle=None,num_workers=4)
    model = model.to(device)
    with torch.no_grad():
        for i,data in enumerate(tqdm(pathLoader,total =len(pathLoader),leave=True,ncols=50)):
            srcpath_image = data
            if srcpath_image.shape[0] == args.BATCH_SIZE:
                input = srcpath_image.to(device)
                out = model(input)
                out = torch.argmax(out,axis=1)
                for j in range(srcpath_image.shape[0]):
                    im = out[j]
                    predicts.append(im)
            if srcpath_image.shape[0] < args.BATCH_SIZE:
                input = srcpath_image.to(device)
                out = model(input)
                out = torch.argmax(out,axis=1)
                for j in range(srcpath_image.shape[0]):
                    im = out[j]
                    predicts.append(im)
        return predicts

def path_img(pre,result_shape,RepetitiveLength,ImgArray, RowOver, ColumnOver,args):
    predicts = []
    colormap = [[255],[0]]
    cm = np.array(colormap).astype('uint8')
    for i in pre:
        pred = i
        pred = pred.squeeze(0).cpu().data.numpy()
        pred = cm[pred]
        pred = pred.reshape((args.path_size,args.path_size))
        predicts.append(pred)
    result_data = Result(result_shape, ImgArray, predicts, RepetitiveLength, RowOver, ColumnOver,args.path_size)
    return result_data

def evaluat(name,gt,args):
    out = cv2.imread(args.save_path+'/'+ name.split(".")[0]+'.png',-1)
    out = out/255.0
    gt = gt/255.0
    iou_score = iou(out , gt)
    biou_score = biou(gt , out)
    return iou_score,biou_score

def epoch_time(start_time, end_time): 
    elapsed_time = end_time - start_time 
    elapsed_mins = int(elapsed_time / 60) 
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60)) 
    return elapsed_mins, elapsed_secs

def main(args):
    RepetitiveLength = int((1 - math.sqrt(args.area_perc)) * args.path_size / 2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ## ['efficientnet-b3' , 'mobilenet_v2']
    model = Net(name_backbone=args.backbone,in_ch=args.input,out_ch=args.output)
    model.load_state_dict(torch.load(args.modelp_path))
    test_dataset = MpiAItest(args.img,args.gt)
    img = read_img(test_dataset)
    iou_scores = np.zeros(len(test_dataset))
    biou_scores = np.zeros(len(test_dataset))
    model.eval()
    tta_model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(),merge_mode='mean')
    for i in range(len(img)):
        image,labels,filename = img[i]
        result_shape = (image.shape[0], image.shape[1])

        ImgArray, RowOver, ColumnOver = TifCroppingArray(image, RepetitiveLength,args.path_size)
        list = array_list(ImgArray,args)

        path_image = img_path(list)

        pre = pre_path(path_image,tta_model,device,args)

        result_data = path_img(pre,result_shape,RepetitiveLength,ImgArray, RowOver, ColumnOver,args)

        cv2.imwrite(args.save_path+'/'+filename.split(".")[0]+'.png',result_data)

        iou_score, biou_score = evaluat(filename, labels,args)
        iou_scores[i] = np.round(iou_score,6)
        biou_scores[i] = np.round(biou_score,6)

    print("iou_score:", np.round(iou_scores.mean(), 5), "biou_score:", np.round(biou_scores.mean(), 5))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='predict for test images')
    parser.add_argument('--path_size', default=384, type=int, metavar='SIZE',
                            help='patch size for image cropped from orig image (default: 384)')
    parser.add_argument('--BATCH_SIZE', default=14, type=int, metavar='SIZE',
                            help='batch size for prediction process (default: 14)')
    parser.add_argument('--area_perc', default=0.5, type=float, metavar='SIZE',
                            help='Area percentage (default: 0.5)')
    parser.add_argument('--input', default=3, type=int, metavar='channels',
                            help='input channels (default: 3)')      
    parser.add_argument('--output', default=2, type=int, metavar='channels',
                            help='output channels (default: 2)')  
    parser.add_argument('--lidar', default=None, type=bool, 
                            help='lidar (default: None)') 
    parser.add_argument('--img', default='/boot/data1/Li_data/data/competition/MapAI-Competition/validation/images', type=str,metavar='DIR',
                            help='Test image path')
    parser.add_argument('--gt', default='/boot/data1/Li_data/data/competition/MapAI-Competition/validation/label', type=str,metavar='DIR',
                            help='groundtruth  path')
    parser.add_argument('--backbone', default='efficientnet-b3', type=str,metavar='backbone',
                            help='backbone')
    parser.add_argument('--modelp_path', default='/home/lisijiang/Documents/MapAI/test/model_p/optical/crop+rotate/Unet-efficientnet-b3.pth', type=str,metavar='DIR',
                            help='model_p path')
    parser.add_argument('--save_path', default='/home/lisijiang/Documents/MapAI/test/PNG/Efficientnet/b3/rotate', type=str,metavar='DIR',
                            help='save path')
    args = parser.parse_args()

    main(args)



