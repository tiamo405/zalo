import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
import json
import torchvision
from torchvision import  transforms
from utils.cropvideo import crop_test
from utils.utils import load_checkpoint, save_csv, predict
import cv2
from model import MobileNetv2
def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='nam')
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('-j', '--workers', type=int, default=2)
    parser.add_argument('--load_height', type=int, default=480)
    parser.add_argument('--load_width', type=int, default=360)
    parser.add_argument('--replicate', type=int, default=11)
    parser.add_argument('--shuffle', action='store_true')
    
    #
    parser.add_argument('--dataset_dir', type=str, default='zalo/dataset')
    parser.add_argument('--test_dir', type=str, default='zalo/dataset/test')
    parser.add_argument('--save_dir', type=str, default='zalo/results/')
    # parser.add_argument('--vid_test_dir', type=str, default='zalo/dataset/public_2/videos')
    parser.add_argument('--public', type=str, choices=['public', 'public2'], default='public2')
    
    #checkpoints, test
    parser.add_argument('--name_model', type= str,choices=['resnet50', 'mobilenet_v2', \
        'mobilenet_v3_small', 'mobilenet_v3_large', 'alexnet'] , default= 'mobilenet_v2')
    parser.add_argument('--checkpoint_dir', type=str, default='zalo/saved_models/')
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
    if opt.name_model == 'resnet50':
        model =  torchvision.models.resnet50(pretrained=False)
        
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    #--------------mobilenetv2------------
    if opt.name_model == 'mobilenet_v2' :
        model = MobileNetv2(type= False)

    #---------------------mobilenet_v3_small
    if opt.name_model == 'mobilenet_v3_small' :
        model = torchvision.models.mobilenet_v3_small(pretrained = False)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)

    #-----------mobilenet v3 lagre
    if opt.name_model == 'mobilenet_v3_large':
        model = torchvision.models.mobilenet_v3_large(pretrained=False) 
        model.classifier.append (nn.Linear(model.classifier[-1].out_features, 2))
       #---------------torchvision.models.alexnet 
    if opt.name_model =='alexnet' :
        model = torchvision.models.alexnet(pretrained = False)
        # model.classifier.append(nn.Linear(model.classifier[-1].out_features, 2))
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
#-----------------------------------
    load_checkpoint(model, os.path.join(opt.checkpoint_dir,opt.name_model, opt.num_point))
    model = model.to(device)
    model.eval()
    index = 0
    res = []
    x =[]
    y = []
    if opt.public == 'public' :
        fnames = os.listdir(os.path.join(opt.test_dir, str(opt.replicate), opt.public))
        fnames.sort()
        
        for fname in fnames :
            index +=1
            path_img = os.path.join(opt.test_dir, str(opt.replicate),opt.public ,fname)
            img = cv2.imread(path_img)
            img = trans(img, opt)
            img = img.to(device).unsqueeze(0)
            output = model(img)
            # print(output)
            res.append(np.argmax(output.cpu().detach().numpy()))
            # res.append(output.cpu().detach().numpy()[0])

            if index == opt.replicate :
                res = np.array(res)
                print(os.path.join(opt.test_dir, str(opt.replicate), opt.public, fname.split("_")[0]))
                x.append(fname.split("_")[0]+".mp4")
                y.append(predict(res))
                # y.append(abs(abs(np.sum(res[:,0])) - abs(np.sum(res[:,1])))/len(res))

                res = []
                index = 0

        save_csv(x, y, opt)
    else :
        path = os.path.join(opt.test_dir, "test2")
        fnames = os.listdir(path)
        fnames.sort()
        for fname in fnames :
            path_imgs = os.path.join(path, fname)
            print(path_imgs)
            if len(os.listdir(path_imgs) ) > 0 :
                for f_img in os.listdir(path_imgs) :
                    if ".jpg" in f_img :
                        index +=1
                        path_img = os.path.join(path_imgs, f_img)
                        a = json.load(open(os.path.join(path_imgs, f_img.split(".")[0] + ".json")))['bbox']
                        image = cv2.imread(path_img)
                        img = image[abs(a[1]): abs(a[3]), abs(a[0]): abs(a[2])]
                        img = cv2.resize(img, (opt.load_width,opt.load_height))
                        img = trans(img, opt).to(device).unsqueeze(0)
                        output = model(img)
                        res.append(np.argmax(output.cpu().detach().numpy()))
                        # res.append(output.cpu().detach().numpy()[0])

                        # res.append(abs(output.cpu().detach().numpy()[0][0] - \
                            #             output.cpu().detach().numpy()[0][1]))
                        if index == (len(os.listdir(path_imgs)) // 2 ):
                            x.append(fname+".mp4")
                            res = np.array(res)
                            # y.append(abs(abs(np.sum(res[:,0])) - abs(np.sum(res[:,1])))/len(res))
                            y.append(predict(res))

                            res = []
                            index = 0
            else :
                x.append(fname + ".mp4")
                y.append(0.1)
        save_csv(x, y, opt)
#-------------------------------------------

if __name__ == '__main__':
    opt = get_opt()
    if os.path.exists(opt.save_dir) == False:
        os.mkdir(opt.save_dir)
    if os.path.exists(opt.test_dir) == False:
        os.mkdir(opt.test_dir)
    # crop_test(opt = opt)
    test(opt)
 
    print("done")

    #python zalo/predict.py --replicate 11 --name_model mobilenet_v2 --public public2