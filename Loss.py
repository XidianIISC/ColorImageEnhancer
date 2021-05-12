import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models
from utils import rgb2gray
import pytorch_ssim as pytorch_ssim


class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()

    def gradient(self,x):

        h_x = x.size()[-2]
        w_x = x.size()[-1]
        l = x
        r = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
        t = x
        b = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

        dx, dy = torch.abs(r - l), torch.abs(b - t)
        dx[:, :, :, -1] = 0
        dy[:, :, -1, :] = 0

        return dx, dy


    def forward(self,hat,gt):
        if hat.size()[1]==3:
            hat_gray=rgb2gray(hat)
            high_gray=rgb2gray(gt)
            grad_hat=self.gradient(hat_gray)
            grad_high=self.gradient(high_gray)
            x_loss=(grad_hat[0]-grad_high[0])**2
            y_loss=(grad_hat[1]-grad_high[1])**2

            grad_loss_all=torch.mean(x_loss+y_loss)
            return grad_loss_all
        else:
            grad_hat=self.gradient(hat)
            grad_high=self.gradient(gt)
            x_loss = (grad_hat[0] - grad_high[0]) ** 2
            y_loss = (grad_hat[1] - grad_high[1]) ** 2

            grad_loss_all = torch.mean(x_loss + y_loss)
            return grad_loss_all

class SquareLoss(nn.Module):
    def __init__(self):
        super(SquareLoss, self).__init__()
    def forward(self,hat,gt):
        return F.mse_loss(hat,gt)

class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()
    def forward(self, hat,gt):
        return 1-pytorch_ssim.ssim(hat,gt)


class RecoveryLoss(nn.Module):
    def __init__(self):
        super(RecoveryLoss, self).__init__()
    def forward(self,hat,gt):
        return F.mse_loss(hat,gt)

class RegionLoss(nn.Module):
    def __init__(self):
        super(RegionLoss,self).__init__()
    def forward(self,attention,hat,gt):
        norm1 = F.l1_loss(attention * hat, attention * gt)
        ssim = 1 - pytorch_ssim.ssim(attention * hat, attention * gt)
        return norm1 + ssim


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()
    def forward(self,x,gt):
        return F.l1_loss(x,gt)


class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()
    def forward(self,x,gt):
        return F.mse_loss(x,gt)


class MSE_in_refletence(nn.Module):
    def __init__(self):
        super(MSE_in_refletence,self).__init__()
    def forward(self,x,gt):
        x_Y=torch.sqrt((x[0:1,:,:]+1e-6)**2+(x[1:2,:,:]+1e-6)**2+(x[2:3,:,:]+1e-6)**2)/math.sqrt(3)
        x_r,x_g,x_b=(x[0:1,:,:]/(math.sqrt(3)*x_Y),x[1:2, :, :] / (math.sqrt(3) * x_Y),x[2:3, :, :] / (math.sqrt(3) * x_Y))

        gt_Y=torch.sqrt((gt[0:1,:,:]+1e-6)**2+(gt[1:2,:,:]+1e-6)**2+(gt[2:3,:,:]+1e-6)**2)/math.sqrt(3)
        gt_r,gt_g,gt_b=(gt[0:1,:,:]/(math.sqrt(3)*gt_Y),gt[1:2, :, :] / (math.sqrt(3) * gt_Y),gt[2:3, :, :] / (math.sqrt(3) * gt_Y))

        return 0.3477*F.mse_loss(x_r,gt_r)+0.3369*F.mse_loss(x_g,gt_g)+0.3153*F.mse_loss(x_b,gt_b)



class TVlOSS(nn.Module):
    def __init__(self):
        super(TVlOSS, self).__init__()

    def forward(self,prediction):
        shape=prediction.size()
        x_tv=torch.mean((prediction[:,:,1:,:]-prediction[:,:,:shape[2]-1,:])**2)
        y_tv = torch.mean((prediction[:, :, :, 1:] - prediction[:, :, :, :shape[3]-1]) ** 2)
        loss_tv=x_tv+y_tv
        return loss_tv

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

