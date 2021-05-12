import argparse
import os
import torch
import torch.nn as nn
import time
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

from torchvision.utils import make_grid,save_image
from torch.utils.data import DataLoader


from Datasets import TrainSet,TestSet
from Model import U_net_CBAM
from utils import *

from Loss import GradLoss,SquareLoss,SSIMLoss,MSE_in_refletence


if __name__=='__main__':



    parser=argparse.ArgumentParser()
    parser.add_argument('--BATCH',type=int,default=32)
    parser.add_argument('--EPOCH',type=int,default=800)
    parser.add_argument('--initLR',type=float,default=0.001)
    parser.add_argument('--step_size',type=int,default=300)
    parser.add_argument('--decray_weight',type=float,default=0.5)
    parser.add_argument('--device_id',type=list,default=[4,5,6,7,8])
    args=parser.parse_args()

    USE_CUDA=torch.cuda.is_available()
    device=torch.device("cuda:4" if USE_CUDA else "cpu")
    Tensor=torch.FloatTensor

    saved_models='../saved_models'+'_'+'Uet'+'-'.join(map(lambda i:str(i),time.localtime(time.time())[0:5]))+'/'
    saved_images='../saved_images'+'_'+'Uet'+'-'.join(map(lambda i:str(i),time.localtime(time.time())[0:5]))+'/'
    learn_param_curve=os.path.join(saved_images,'learn_param_curve/')


    model=U_net_CBAM(in_colors=3,out_colors=3)

    model=nn.DataParallel(model,device_ids=args.device_id)
    model.to(device)
    print("model prepared..")

    trainset=TrainSet()
    loader=DataLoader(dataset=trainset,batch_size=args.BATCH)
    print("trainset prepared..")

    optimizer=optim.Adam(model.parameters(),lr=args.initLR,betas=(0.5,0.99))
    sche=StepLR(optimizer,step_size=args.step_size,gamma=args.decray_weight)


    saved_models_dir=saved_models + 'EPOCH_%d' % args.EPOCH + '_initlr_%.6f' % args.initLR + '/'
    saved_images_dir=saved_images + 'EPOCH_%d' % args.EPOCH + '_initlr_%.6f' % args.initLR + '/'
    saved_models_paramCurve_dir=learn_param_curve+ 'EPOCH_%d' % args.EPOCH + '_initlr_%.6f' % args.initLR + '/'
    print("necessary dirs and optimizer prepared..")

    testset=TestSet()
    testloader=DataLoader(dataset=testset,batch_size=1)

    print("testset prepared..")

    if not os.path.exists(saved_models_dir):
        os.makedirs(saved_models_dir)
    if not os.path.exists(saved_images_dir):
        os.makedirs(saved_images_dir)
    if not os.path.exists(saved_models_paramCurve_dir):
        os.makedirs(saved_models_paramCurve_dir)

    criterion_grad=GradLoss()
    criterion_square=SquareLoss()
    criterion_ssim=SSIMLoss()

    criterion_mse_in_refletence=MSE_in_refletence()

    loss_all=[]

    R_grad_all=[]
    R_square_all=[]
    R_ssim_all=[]
    R_mse_in_refletence_all=[]
    R_vgges=[]

    psnr_all=[]
    lr_all=[]

    print("start_time:", time.asctime(time.localtime(time.time())))
    for epoch in range(args.EPOCH):
        losses=[]

        R_grades=[]
        R_squarees=[]
        R_ssimes=[]

        R_mse_in_refletencees=[]


        psnres=[]

        for i,data in enumerate(loader):


            lowRGB = data['orginal'][0].to(device)
            highRGB = data['orginal'][1].to(device)

            orginallow = data['orginal'][0].to(device)
            orginalhigh = data['orginal'][1].to(device)

            optimizer.zero_grad()

            res=model(lowRGB)


            r_grad=criterion_grad(res,highRGB)
            r_square=criterion_square(res,highRGB)
            r_mse_in_ref=criterion_mse_in_refletence(res,highRGB)
            r_ssim=criterion_ssim(res,highRGB)
            r_loss=r_square+r_ssim*0.01+r_grad*10+r_mse_in_ref*10.0


            loss=r_loss


            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            R_grades.append(r_grad.item())
            R_squarees.append(r_square.item())
            R_ssimes.append(r_ssim.item())
            R_mse_in_refletencees.append(r_mse_in_ref.item())


            psnr_rechat_rec_gt=PSNR(res,highRGB)
            psnres.append(psnr_rechat_rec_gt)
            print('Epoch[{}/{}],Batch[{}/{}],loss:{:.8f}'.format(epoch,
                                                                 args.EPOCH,
                                                                 i,
                                                                 462//args.BATCH,
                                                                 loss.item()))

        loss_all.append(np.mean(losses))

        R_grad_all.append(np.mean(R_grades))
        R_square_all.append(np.mean(R_squarees))
        R_ssim_all.append(np.mean(R_ssimes))
        R_mse_in_refletence_all.append(np.mean(R_mse_in_refletencees))
        psnr_all.append(np.mean(psnres))

        lr_all.append(optimizer.state_dict()['param_groups'][0]['lr'])

        sche.step(epoch)

        print('======' + 'Epoch [{}/{}],loss:{:.6f},lr:{:,.6f},r_grad:{:.6f},mse:{:.6f},r_ssim:{:.6f},mse_in_ref:{:.6f}'.format(
                epoch,
                args.EPOCH,
                np.mean(losses),
                optimizer.state_dict()['param_groups'][0]['lr'],
                np.mean(R_grades),
                np.mean(R_squarees),
                np.mean(R_ssimes),
                np.mean(R_mse_in_refletencees))+ '======')
        print("time: ",time.asctime(time.localtime(time.time())))


        if epoch % 50 == 0:
            torch.save(model.state_dict(), saved_models_dir + "epoch_%d" % epoch + '_.pth')


    epoches = list(range(args.EPOCH))
    plt.xlabel('epoch')
    plt.ylabel('lr')
    plt.plot(epoches, lr_all)
    plt.savefig(saved_models_paramCurve_dir + 'lr.jpg')
    plt.clf()

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(epoches, loss_all)
    plt.savefig(saved_models_paramCurve_dir + 'loss.jpg')
    plt.clf()

    plt.xlabel('epoch')
    plt.ylabel('psnr_train')
    plt.plot(epoches, psnr_all)
    plt.savefig(saved_models_paramCurve_dir + 'psnr_train.jpg')
    plt.clf()

    plt.xlabel('epoch')
    plt.ylabel('r_grad')
    plt.plot(epoches, R_grad_all)
    plt.savefig(saved_models_paramCurve_dir + 'r_grad.jpg')
    plt.clf()

    plt.xlabel('epoch')
    plt.ylabel('mse')
    plt.plot(epoches, R_square_all)
    plt.savefig(saved_models_paramCurve_dir + 'mse.jpg')
    plt.clf()

    plt.xlabel('epoch')
    plt.ylabel('r_ssim')
    plt.plot(epoches, R_ssim_all)
    plt.savefig(saved_models_paramCurve_dir + 'r_ssim.jpg')
    plt.clf()

    plt.xlabel('epoch')
    plt.ylabel('mse_in_ref')
    plt.plot(epoches, R_mse_in_refletence_all)
    plt.savefig(saved_models_paramCurve_dir + 'mse_in_ref.jpg')
    plt.clf()


    print("End_time:", time.asctime(time.localtime(time.time())))
