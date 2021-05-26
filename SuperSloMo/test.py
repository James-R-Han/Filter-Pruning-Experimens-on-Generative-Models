
#[Super SloMo] Validation function

import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import math
import model
import dataloader
import datetime
import tensorflow as tf
import numpy as np
import torch.quantization
#from skimage.measure import compare_psnr, compare_ssim



def evaluate(model1, flowComp, validationloader, neval_batches):
    model1.eval()
    cnt = 0
    vgg16 = torchvision.models.vgg16(pretrained=True)
    vgg16_conv_4_3 = nn.Sequential(*list(vgg16.children())[0][:22])
    vgg16_conv_4_3.to(device)

    validationFlowBackWarp = model.backWarp(640, 352, device)
    validationFlowBackWarp = validationFlowBackWarp.to(device)

    tloss = 0

    for param in vgg16_conv_4_3.parameters():
        param.requires_grad = False
    with torch.no_grad():
        for validationIndex, (validationData, validationFrameIndex) in enumerate(validationloader, 0):
            frame0, frameT, frame1 = validationData

            I0 = frame0.to(device)
            I1 = frame1.to(device)
            IFrame = frameT.to(device)

            flowOut = flowComp(torch.cat((I0, I1), dim=1))
            F_0_1 = flowOut[:,:2,:,:]
            F_1_0 = flowOut[:,2:,:,:]

            fCoeff = model.getFlowCoeff(validationFrameIndex, device)

            F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
            F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

            g_I0_F_t_0 = validationFlowBackWarp(I0, F_t_0)
            g_I1_F_t_1 = validationFlowBackWarp(I1, F_t_1)
            
            intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))
                
            F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
            F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
            V_t_0   = torch.sigmoid(intrpOut[:, 4:5, :, :]) 
            V_t_1   = 1 - V_t_0
                
            g_I0_F_t_0_f = validationFlowBackWarp(I0, F_t_0_f)
            g_I1_F_t_1_f = validationFlowBackWarp(I1, F_t_1_f)
            
            wCoeff = model.getWarpCoeff(validationFrameIndex, device)
            
            Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)
            
            
            L1_lossFn = nn.L1Loss()
            MSE_LossFn = nn.MSELoss()
            recnLoss = L1_lossFn(Ft_p, IFrame)
        
            prcpLoss = MSE_LossFn(vgg16_conv_4_3(Ft_p), vgg16_conv_4_3(IFrame))    
            warpLoss = L1_lossFn(g_I0_F_t_0, IFrame) + L1_lossFn(g_I1_F_t_1, IFrame) + L1_lossFn(validationFlowBackWarp(I0, F_1_0), I1) + L1_lossFn(validationFlowBackWarp(I1, F_0_1), I0)
            loss_smooth_1_0 = torch.mean(torch.abs(F_1_0[:, :, :, :-1] - F_1_0[:, :, :, 1:])) + torch.mean(torch.abs(F_1_0[:, :, :-1, :] - F_1_0[:, :, 1:, :]))
            loss_smooth_0_1 = torch.mean(torch.abs(F_0_1[:, :, :, :-1] - F_0_1[:, :, :, 1:])) + torch.mean(torch.abs(F_0_1[:, :, :-1, :] - F_0_1[:, :, 1:, :]))
            loss_smooth = loss_smooth_1_0 + loss_smooth_0_1
            
            
            loss = 204 * recnLoss + 102 * warpLoss + 0.005 * prcpLoss + loss_smooth
            #tloss += loss.item()
            tloss = loss.item()

            cnt += 1
            
            if cnt >= neval_batches:
                 return

    return




# provide:
#--dataset_root  and --checkpoint_dir and --checkpoint /path/to/model

#For parsing commandline arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_root", type=str, required=False, help='path to dataset folder containing train-test-validation folders')
parser.add_argument("--checkpoint_dir", type=str, required=False, help='path to folder for saving checkpoints')
parser.add_argument("--checkpoint", type=str, help='path of checkpoint for pretrained model')
parser.add_argument("--train_continue", type=bool, default=False, help='If resuming from checkpoint, set to True and set `checkpoint` path. Default: False.')
parser.add_argument("--epochs", type=int, default=200, help='number of epochs to train. Default: 200.')
parser.add_argument("--train_batch_size", type=int, default=6, help='batch size for training. Default: 6.')
parser.add_argument("--validation_batch_size", type=int, default=1, help='batch size for validation. Default: 10.')
parser.add_argument("--init_learning_rate", type=float, default=0.0001, help='set initial learning rate. Default: 0.0001.')
parser.add_argument("--milestones", type=list, default=[100, 150], help='Set to epoch values where you want to decrease learning rate by a factor of 0.1. Default: [100, 150]')
parser.add_argument("--progress_iter", type=int, default=100, help='frequency of reporting progress and validation. N: after every N iterations. Default: 100.')
parser.add_argument("--checkpoint_epoch", type=int, default=5, help='checkpoint saving frequency. N: after every N epochs. Each checkpoint is roughly of size 151 MB.Default: 5.')
args = parser.parse_args()



###Initialize flow computation and arbitrary-time flow interpolation CNNs.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"
flowComp = model.UNet(6, 4)
flowComp.to(device)
ArbTimeFlowIntrp = model.UNet(20, 5)
ArbTimeFlowIntrp.to(device)

###Initialze backward warpers for train and validation datasets
validationFlowBackWarp = model.backWarp(640, 352, device)
validationFlowBackWarp = validationFlowBackWarp.to(device)

###Load Datasets
# Channel wise mean calculated on adobe240-fps training dataset
mean = [0,0,0]
std  = [1, 1, 1]
normalize = transforms.Normalize(mean=mean, std=std)
#normalize = None

transform = transforms.Compose([transforms.ToTensor(), normalize])

#cahnged '/validation' to '/test'
validationset = dataloader.SuperSloMo(root=args.dataset_root + '/test', transform=transform, randomCropSize=(640, 352), train=False)
#validationset = dataloader.SuperSloMo(root='data' + '/train_subset', randomCropSize=(640, 352), train=False)
validationloader = torch.utils.data.DataLoader(validationset, batch_size=args.validation_batch_size, shuffle=False)

print(validationset) 

###Create transform to display image from tensor
negmean = [x * -1 for x in mean]
revNormalize = transforms.Normalize(mean=negmean, std=std)
TP = transforms.Compose([revNormalize, transforms.ToPILImage()])
#TP = transforms.Compose([None, transforms.ToPILImage()])

###Utils
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

###Loss and Optimizer
L1_lossFn = nn.L1Loss()
MSE_LossFn = nn.MSELoss()

params = list(ArbTimeFlowIntrp.parameters()) + list(flowComp.parameters())

optimizer = optim.Adam(params, lr=args.init_learning_rate)
# scheduler to decrease learning rate by a factor of 10 at milestones.
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

### Validation function
# 
def validate():
    # For details see training.
    psnr = 0
    tloss = 0
    flag = 1
    ssim = 0
    ie = 0
    max_num = 0
    max_num2 = 0
    with torch.no_grad():
       for validationIndex, (validationData, validationFrameIndex) in enumerate(validationloader, 0):
            frame0, frameT, frame1 = validationData

            I0 = frame0.to(device)
            I1 = frame1.to(device)
            IFrame = frameT.to(device)
            num = torch.max(IFrame)
            if max_num < num:
                max_num = num
            
            flowOut = flowComp(torch.cat((I0, I1), dim=1))
            F_0_1 = flowOut[:,:2,:,:]
            F_1_0 = flowOut[:,2:,:,:]

            fCoeff = model.getFlowCoeff(validationFrameIndex, device)

            F_t_0 = fCoeff[0] * F_0_1 + fCoeff[1] * F_1_0
            F_t_1 = fCoeff[2] * F_0_1 + fCoeff[3] * F_1_0

            g_I0_F_t_0 = validationFlowBackWarp(I0, F_t_0)
            g_I1_F_t_1 = validationFlowBackWarp(I1, F_t_1)
            
            intrpOut = ArbTimeFlowIntrp(torch.cat((I0, I1, F_0_1, F_1_0, F_t_1, F_t_0, g_I1_F_t_1, g_I0_F_t_0), dim=1))
                
            F_t_0_f = intrpOut[:, :2, :, :] + F_t_0
            F_t_1_f = intrpOut[:, 2:4, :, :] + F_t_1
            V_t_0   = torch.sigmoid(intrpOut[:, 4:5, :, :]) # can use F.sigmoid too
            V_t_1   = 1 - V_t_0
                
            g_I0_F_t_0_f = validationFlowBackWarp(I0, F_t_0_f)
            g_I1_F_t_1_f = validationFlowBackWarp(I1, F_t_1_f)
            
            wCoeff = model.getWarpCoeff(validationFrameIndex, device)
            
            Ft_p = (wCoeff[0] * V_t_0 * g_I0_F_t_0_f + wCoeff[1] * V_t_1 * g_I1_F_t_1_f) / (wCoeff[0] * V_t_0 + wCoeff[1] * V_t_1)
            
            # For tensorboard
            if (flag):
                retImg = torchvision.utils.make_grid([revNormalize(frame0[0]), revNormalize(frameT[0]), revNormalize(Ft_p.cpu()[0]), revNormalize(frame1[0])], padding=10)
                flag = 0
            
            num = torch.max(Ft_p)
            if max_num2 < num:
                max_num2 = num
            #loss
            recnLoss = L1_lossFn(Ft_p, IFrame)
            
            warpLoss = L1_lossFn(g_I0_F_t_0, IFrame) + L1_lossFn(g_I1_F_t_1, IFrame) + L1_lossFn(validationFlowBackWarp(I0, F_1_0), I1) + L1_lossFn(validationFlowBackWarp(I1, F_0_1), I0)
        
            loss_smooth_1_0 = torch.mean(torch.abs(F_1_0[:, :, :, :-1] - F_1_0[:, :, :, 1:])) + torch.mean(torch.abs(F_1_0[:, :, :-1, :] - F_1_0[:, :, 1:, :]))
            loss_smooth_0_1 = torch.mean(torch.abs(F_0_1[:, :, :, :-1] - F_0_1[:, :, :, 1:])) + torch.mean(torch.abs(F_0_1[:, :, :-1, :] - F_0_1[:, :, 1:, :]))
            loss_smooth = loss_smooth_1_0 + loss_smooth_0_1
            
            
            
            tloss = 0

            #psnr
            
            MSE_val = MSE_LossFn(Ft_p, IFrame)
            psnr += (10 * math.log10(1 / MSE_val.item()))

            c = Ft_p.cpu().numpy()
            d= IFrame.cpu().numpy()
            err = c - d
            ie = np.mean(np.sqrt(np.sum(err * err, axis=2)))
            #ie += math.pow(MSE_val,0.5)

            a = Ft_p.reshape(352,640,3)
            b = IFrame.reshape(352,640,3)

            
            a= a.cpu() #.numpy()
            b = b.cpu() #.numpy()
            ssim += tf.image.ssim(a,b,1)
           
        
    psnr = psnr / len(validationloader)
    tloss = tloss / len(validationloader)
    ssim = ssim / len(validationloader)
    ie = ie / len(validationloader)
            
    return psnr, tloss, ssim, ie, max_num, max_num2


### Initialization
import time

result_file  = open("result_file.txt", "w")

CHECKPOINT_PATH = args.checkpoint

dict1 = torch.load(CHECKPOINT_PATH)
ArbTimeFlowIntrp.load_state_dict(dict1['state_dictAT'])
flowComp.load_state_dict(dict1['state_dictFC'])


start = time.time()
# model is treated as a global variable
psnr, vLoss, ssim, ie, max_number, max_number2= validate()
end = time.time()
val_time = end - start

result_file.write("name: " + CHECKPOINT_PATH)
result_file.write("\n")
result_file.write("time required: " + str(val_time))
result_file.write("\n")
result_file.write("psnr: " + str(psnr))
result_file.write("\n")
result_file.write("ssim: " + str(ssim))
result_file.write("\n")
result_file.write("ie: " + str(ie))
result_file.write("\n\n\n")

result_file.close() 



    
