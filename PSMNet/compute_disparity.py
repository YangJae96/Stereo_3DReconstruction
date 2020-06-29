from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import time
import math
from .models import *
import cv2
from PIL import Image

# 2012 data /media/jiaren/ImageNet/data_scene_flow_2012/testing/

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] ="2,3"   


maxdisp = 192 
model = stackhourglass(maxdisp)

if torch.cuda.is_available():
    device=torch.device('cuda')

model = nn.DataParallel(model)
model.to(device=device)

print('load PSMNet')
state_dict = torch.load('./PSMNet/trained/pretrained_model_KITTI2015.tar')
model.load_state_dict(state_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

def test(imgL,imgR):
        model.eval()

        imgL = imgL.cuda()
        imgR = imgR.cuda()     

        with torch.no_grad():
            disp = model(imgL,imgR)

        disp = torch.squeeze(disp)
        pred_disp = disp.data.cpu().numpy()

        return pred_disp


def get_disparity(leftimg, rightimg):

        normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                            'std': [0.229, 0.224, 0.225]}
        infer_transform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(**normal_mean_var)])    

        imgL = infer_transform(leftimg)
        imgR = infer_transform(rightimg) 

        # pad to width and hight to 16 times
        if imgL.shape[1] % 16 != 0:
            times = imgL.shape[1]//16       
            top_pad = (times+1)*16 -imgL.shape[1]
        else:
            top_pad = 0

        if imgL.shape[2] % 16 != 0:
            times = imgL.shape[2]//16                       
            right_pad = (times+1)*16-imgL.shape[2]
        else:
            right_pad = 0    

        imgL = F.pad(imgL,(0,right_pad, top_pad,0)).unsqueeze(0)
        imgR = F.pad(imgR,(0,right_pad, top_pad,0)).unsqueeze(0)


        start_time = time.time()
        
        pred_disp = test(imgL,imgR)


        print('Disparity Estimation time = %.2f' %(time.time() - start_time))
        img = pred_disp
        img = (img*256).astype('uint16')
        # img.save('Test_disparity.png')
        
        # img = Image.fromarray(img)
        # img.save('Test_disparity.png')

        return img

if __name__ == '__main__':
   img = get_disparity
   print(img)





