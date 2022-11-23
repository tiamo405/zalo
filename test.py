import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch import nn

import torchvision
from torchvision import  transforms
from utils.cropvideo import crop_test
from utils.utils import load_checkpoint, save_csv, predict
import cv2

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='nam')
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('-j', '--workers', type=int, default=2)
    parser.add_argument('--load_height', type=int, default=360)
    parser.add_argument('--load_width', type=int, default=224)
    parser.add_argument('--replicate', type=int, default=5)
    parser.add_argument('--shuffle', action='store_true')
    
    #
    parser.add_argument('--dataset_dir', type=str, default='zalo/dataset')
    parser.add_argument('--test_dir', type=str, default='zalo/dataset/test')
    parser.add_argument('--save_dir', type=str, default='zalo/results/')
    # parser.add_argument('--vid_test_dir', type=str, default='zalo/dataset/public_2/videos')
    parser.add_argument('--public', type=str, default='public')
    
    #checkpoints, test
    parser.add_argument('--name_model', type= str, default= 'mobilenet_v2')
    parser.add_argument('--checkpoint_dir', type=str, default='zalo/checkpoints/')
    parser.add_argument('--num_point', type= str, default= 'best_epoch.pth')
    parser.add_argument("--gpu", type=str, default=1, help="choose gpu device.")
    
    

    opt = parser.parse_args()
    return opt
# ------------------------------------------------
def trans(img, opt) :
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    img = cv2.resize(img, (opt.load_width,opt.load_height))
    img = transform(img)
    return img

def test(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # --------------resnet50-----------
    # model =  torchvision.models.resnet50(pretrained=False)
    
    # num_ftrs = model.fc.in_features
    # model.fc = nn.Linear(num_ftrs, 2)
    #--------------mobilenetv2------------
    # model = MobileNetv2()
    #---------------------mobilenet_v3_small
    model = torchvision.models.mobilenet_v3_small(pretrained = False)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)

    #-----------mobilenet v3 lagre
    # model = torchvision.models.mobilenet_v3_large(pretrained=False) 
    # model.classifier.append (nn.Linear(model.classifier[-1].out_features, 2))

#-----------------------------------
    load_checkpoint(model, os.path.join(opt.checkpoint_dir,opt.name_model, opt.num_point))
    model = model.to(device)
    model.eval()
    fnames = os.listdir(os.path.join(opt.test_dir, str(opt.replicate), opt.public))
    fnames.sort()
    index = 0
    res = []
    x =[]
    y = []
    for fname in fnames :
        index +=1
        path_img = os.path.join(opt.test_dir, str(opt.replicate),opt.public ,fname)
        img = cv2.imread(path_img)
        img = trans(img, opt)
        img = img.to(device).unsqueeze(0)
        output = model(img)
        if index == opt.replicate :
            print(abs(np.mean(output.cpu().detach().numpy())))
            res.append(np.argmax(output.cpu().detach().numpy()))
            x.append(fname.split("_")[0]+".mp4")

            y.append(predict(res))
            # y.append(abs(np.mean(output.cpu().detach().numpy())))
            # print(res)
            res = []
            index = 0
        else :
            res.append(np.argmax(output.cpu().detach().numpy()))
    save_csv(x, y, opt)
    
    
#-------------------------------------------

if __name__ == '__main__':
    opt = get_opt()
    if os.path.exists(opt.save_dir) == False:
        os.mkdir(opt.save_dir)
    if os.path.exists(opt.test_dir) == False:
        os.mkdir(opt.test_dir)
    crop_test(opt= opt)
    if len(os.listdir(os.path.join(opt.dataset_dir, opt.public, "videos"))) * opt.replicate == \
                    len(os.listdir(os.path.join(opt.test_dir, str(opt.replicate), opt.public))) : #vid * replicate=img crop
        test(opt=opt)
    else :print("crop image erorr")
    print("done")

    #python zalo/test.py --replicate 11 --name_model mobilenet_v3_small --public public_2