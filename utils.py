import torch
import torch.nn as nn
import math
import os
import torchvision.transforms as tf
from PIL import Image



USE_CUDA = torch.cuda.is_available()
device=torch.device("cuda:4" if USE_CUDA else "cpu")


def makeAttention(low,high):
    B,C,H,W=low.size()
    if C==1:
        A=torch.reshape(torch.div(torch.abs(high-low),high+1e-6),(-1,1,H,W))
        return A
    else:
        maxc_high=torch.max(high,dim=1,keepdim=True)[0]
        maxc_low=torch.max(low,dim=1,keepdim=True)[0]
        A=torch.div(torch.abs(maxc_high-maxc_low),(maxc_high+1e-6))

        thred=torch.ones_like(A)*0.8
        A=torch.min(A,thred)

        return A

def rgb2gray(x):
    res=torch.zeros((x.size()[0],1,x.size()[2],x.size()[3])).type(torch.FloatTensor)
    res=res.to(device)
    R=x[:,0:1,:,:]
    G=x[:,1:2,:,:]
    B=x[:,2:3,:,:]
    res[:,0:1,:,:]=R*0.299+0.587*G+0.114*B
    return res

def decompose(input_tensor):
    if len(input_tensor.size())==4:
        Y=torch.sqrt((input_tensor[:,0:1,:,:]+1e-6)**2+(input_tensor[:,1:2,:,:]+1e-6)**2+(input_tensor[:,2:3,:,:]+1e-6)**2)/math.sqrt(3)
        refletence=torch.cat((input_tensor[:,0:1,:,:]/(math.sqrt(3)*Y),
                              input_tensor[:, 1:2, :, :] / (math.sqrt(3) * Y),
                              input_tensor[:, 2:3, :, :] / (math.sqrt(3) * Y)),dim=1)
        return Y,refletence
    if len(input_tensor.size())==3:
        Y=torch.sqrt((input_tensor[0:1,:,:]+1e-6)**2+(input_tensor[1:2,:,:]+1e-6)**2+(input_tensor[2:3,:,:]+1e-6)**2)/math.sqrt(3)
        refletence=torch.cat((input_tensor[0:1,:,:]/(math.sqrt(3)*Y),
                              input_tensor[1:2, :, :] / (math.sqrt(3) * Y),
                              input_tensor[2:3, :, :] / (math.sqrt(3) * Y)),dim=0)
        return Y,refletence


def PSNR(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(1.0 ** 2 / mse.item())


def weights_init(m):
    classname=m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal(m.weight.data)
        nn.init.xavier_normal(m.bias.data)


