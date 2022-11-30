import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import cv2
import torch.nn.functional as F
import time
import argparse
from tqdm import tqdm
from util.save_checkpoint import save_checkpoint
from util.metrics import F1Score, OAAcc
from util.dataset import generate_data
from util.network import Net


def train(model, train_loader,criterion,optimizer,device,epoch): 
    model.train()   
    train_seg = 0.0
    print('Epoch: {:.2f}'.format(epoch+1))
    for i, data in enumerate(tqdm(train_loader,total =len(train_loader),leave=True,ncols=50)):
        images,mask = data
        images,mask = images.to(device),mask.to(device)

        optimizer.zero_grad()  # 清零梯度准备计算

        out,e,d = model(images)  
        loss_seg = criterion(out,mask) 
        loss_seg.backward()
        optimizer.step()
        train_seg += loss_seg.item()  

    print('Train Seg loss: {:.3f}'.format(train_seg/len(train_loader)))

def vaild(model, dataloader,device,epoch):
    val_macc = 0.0
    model.eval() 
    predicts = []
    colormap = [[255],[0]]
    cm = np.array(colormap).astype('uint8')
    print('Epoch: {:.2f}'.format(epoch+1))
    with torch.no_grad(): 
        for i, data in enumerate(tqdm(dataloader,total = len(dataloader),leave=True,ncols=50)):
            images,mask,filename = data
            images,mask = images.to(device),mask.to(device)       
                
            pred = model(images)  
            metric = OAAcc()
            macc , acc = metric(pred,mask)
            val_macc += macc.item()

        print(' .Val mAcc: {:.9f} '
        .format(val_macc/len(dataloader)))
    return val_macc/len(dataloader)

def epoch_time(start_time, end_time): 
    elapsed_time = end_time - start_time 
    elapsed_mins = int(elapsed_time / 60) 
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60)) 
    return elapsed_mins, elapsed_secs

def main(args):

    if args.task == 1:
        args.lidar = None
        args.backbone = 'efficientnet-b3'
        args.input = 3
        args.crop_size = 384
    elif args.task == 2:
        args.lidar = True
        args.backbone = 'efficientnet-b4'
        args.input = 1
        args.crop_size = 224
        args.PATH = './test/Unet-efficientnet-b4.pt'

    train_dataset = generate_data(args.img_dir,args.gt_dir,args.crop_size,train=True,lidar=args.lidar)
    vaild_dataset = generate_data(args.img_dir,args.gt_dir,args.crop_size,train=None,lidar=args.lidar)

    train_dataloader = DataLoader(train_dataset,batch_size=args.BATCH_SIZE,shuffle=True,num_workers=4)
    vaild_dataloader = DataLoader(vaild_dataset,batch_size=args.BATCH_SIZE,shuffle=None,num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(name_backbone=args.backbone,in_ch=args.input,out_ch=args.output)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    save_checkpoint(args.checkpoints_path,args).save()
    vaild_macc = []
    best_acc = 0.0
    for epoch in range(args.EPOCHS):
        start_time = time.monotonic()
        
        train(model, train_dataloader,criterion,optimizer,device,epoch)
        macc = vaild(model, vaild_dataloader,device,epoch)
        vaild_macc.append(macc)

        end_time = time.monotonic()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        if vaild_macc[epoch] >= best_acc:
            best_acc = max(vaild_macc[epoch],best_acc)
            torch.save(model.state_dict(), args.PATH)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Supervised Segmentation with Perfect Labels')
    parser.add_argument('--BATCH_SIZE', default=8, type=int, metavar='SIZE',
                            help='batch size for prediction process (default: 8)')
    parser.add_argument('--crop_size', default=384, type=int, metavar='SIZE',
                            help='parameter of dataset (default: 384)')
    parser.add_argument('--EPOCHS', default=100, type=int, metavar='SIZE',
                            help='epoch (default: 100)')
    parser.add_argument('--input', default=3, type=int, metavar='channels',
                            help='input channels (default: 3)')      
    parser.add_argument('--output', default=2, type=int, metavar='channels',
                            help='output channels (default: 2)')  
    parser.add_argument('--lr', default=0.001, type=int, 
                            help='lr (default: 0.001)')       
    parser.add_argument('--lidar', default=None, type=bool, 
                            help='lidar (default: None)') 
    parser.add_argument("--task", type=int, default=3, help="Which task you are submitting for")           
    parser.add_argument('--img_dir', default='/boot/data1/Li_data/data/competition/MapAI-Competition/train/images', type=str,metavar='DIR',
                            help='Train optical path')
    parser.add_argument('--gt_dir', default='/boot/data1/Li_data/data/competition/MapAI-Competition/train/label', type=str,metavar='DIR',
                            help='groundtruth  path')
            ## ['efficientnet-b3' , 'efficientnet-b4' ]
    parser.add_argument('--backbone', default='efficientnet-b3', type=str,metavar='backbone',
                            help='backbone')
    parser.add_argument('--checkpoints_path', default='./config/data.yaml', type=str,metavar='DIR',
                            help='save checkpoints path')
    parser.add_argument('--PATH', default='./test/Unet-efficientnet-b3.pt', type=str,metavar='DIR',
                            help='save model_p path')
    args = parser.parse_args()
    main(args)

