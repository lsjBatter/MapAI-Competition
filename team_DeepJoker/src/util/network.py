import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import scipy
import math
# from MFF import scSE
class Net(nn.Module):
    """
    Args:
    in_ch : input_channels
    ou_ch : num_classes
    name_backbone:['efficientnet-b3' , 'efficientnet-b4']
                  default: 'resnet101'

    """
    def __init__(self,
                 in_ch:int = 3,
                 out_ch:int = 2,
                 name_backbone:str = 'efficientnet-b3'):
        super(Net, self).__init__()
        self.name_backbone = name_backbone
        self.model = smp.Unet(name_backbone,encoder_weights='imagenet',in_channels=in_ch,classes=out_ch)
        self.encoder = self.model.encoder
        self.segmentation_head = self.model.segmentation_head
        self.model.segmentation_head = nn.Sequential()

    def forward(self,x):
        t_encoder = self.encoder(x)
        decoder_feat = self.model(x)
        out = self.segmentation_head(decoder_feat)
        # return t_encoder
        if self.training:
            return out , t_encoder, decoder_feat 
        else:
            return out 

if __name__ == '__main__':
    x = torch.randn(1,3,384,384)
    model = Net(name_backbone='efficientnet-b3')

    out= model(x)
    print(out[0].shape,out[1].shape,out[2].shape,out[3].shape,out[4].shape,out[5].shape)
