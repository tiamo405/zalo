import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
from datasets import ZaloDataset
from utils.cropvideo import crop_train
from model import MobileNetv2

import copy
def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='nam')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('-j', '--workers', type=int, default=2)
    parser.add_argument('--load_height', type=int, default=480)
    parser.add_argument('--load_width', type=int, default=360)
    parser.add_argument('--replicate', type=int, default=11)
    parser.add_argument('--shuffle', action='store_true')
    
    #
    parser.add_argument('--dataset_dir', type=str, default='dataset')
    parser.add_argument('--train_dir', type=str, default='zalo/dataset/train')
    parser.add_argument('--val_dir', type=str, default='zalo/dataset/val')
    parser.add_argument('--save_dir', type=str, default='zalo/results')
    
    #checkpoints, train
    parser.add_argument('--name_model', type= str,choices=['resnet50', 'mobilenet_v2', \
        'mobilenet_v3_small', 'mobilenet_v3_large', 'alexnet', 'convnext_tiny'] , default= 'resnet50')
    parser.add_argument('--checkpoint_dir', type=str, default='zalo/saved_models/')
    parser.add_argument("--gpu", type=str, default='1', help="choose gpu device.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.05)

    opt = parser.parse_args()
    return opt
# -------------------------------------------------
def train(opt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device :", device)
    # ----------------Net_1------------------------
    # model = Net_1()
    # optimizer = optim.SGD(model.parameters(), lr=3e-2)

    # loss_fn = nn.CrossEntropyLoss()
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # ----------------resnet50------------------------
    if opt.name_model == 'resnet50' :
        model = torchvision.models.resnet50(pretrained=True)
        num_ftrs = model.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        model.fc = nn.Linear(num_ftrs, 2)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum = 0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    #-------------MOBILENET_V2------------
    elif opt.name_model =='mobilenet_v2' :
        model = MobileNetv2()
  
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        # optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum = 0.9)
        optimizer = torch.optim.Adam(model.parameters(),lr= opt.lr, betas=(0.9, 0.999))
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  
     
    #-----------------------torchvision.models.mobilenet_v3_small
    elif opt.name_model == 'mobilenet_v3_small' :
        model = torchvision.models.mobilenet_v3_small(pretrained = True)  
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum = 0.9)
        # optimizer = torch.optim.Adam(model.parameters(),lr= opt.lr, betas=(0.9, 0.999))
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  
    #-----------------------mobilenet_v3_large
    elif opt.name_model == 'mobilenet_v3_large' :
        model = torchvision.models.mobilenet_v3_large(pretrained=True) 
        model.classifier.append (nn.Linear(model.classifier[-1].out_features, 2))
        # print(model)
        # linear_model = nn.Linear(1000, 2,device= device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum = 0.9)
        # optimizer = torch.optim.Adam(lr= opt.lr, beta_1=0.9, beta_2=0.999)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  
        # trans = transforms.Compose([transforms.ToTensor()])
       #---------------torchvision.models.alexnet 
    elif opt.name_model =='alexnet' :
        model = torchvision.models.alexnet(pretrain = True)
        # model.classifier.append(nn.Linear(model.classifier[-1].out_features, 2))
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 2)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum = 0.9)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) 
    # ---------------
     
    #-------------------Train the model----------------------------
    print(model)
    model = model.to(device)
    if not os.path.exists(os.path.join(opt.checkpoint_dir, opt.name_model)) :
        os.mkdir(os.path.join(opt.checkpoint_dir, opt.name_model))
    best_acc = 0.0
    best_epoch = 0
    for epoch in range(1, opt.epochs+1):
        print(f'Epoch {epoch}/{opt.epochs}')
        print('-' * 15)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            for inputs in tqdm(dataLoader[phase]):
                input = inputs['img'].to(device)
                labels = inputs['label'].to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(input)
                    # print(outputs)
                    _, preds = torch.max(outputs, 1)
                    # print(preds)
                    loss = criterion(outputs, labels)
                    # print(loss)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * input.size(0)
                
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()
            # print(running_loss)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_epoch = epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, os.path.join(opt.checkpoint_dir, opt.name_model, (str(epoch)+".pth")))
        print()
    print(f'Best val Acc: {best_acc:4f}, epoch: {best_epoch}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_epoch
#-------------------------------------------

if __name__ == '__main__':
    opt = get_opt()
    #----------------- tach anh tu video
    # crop_train(opt= opt)

    if os.path.exists(opt.checkpoint_dir) == False:
        os.mkdir(opt.checkpoint_dir)

    # #----------------------dataloader---------------------
    Dataset = ZaloDataset (opt=opt)
    train_size = int(0.8 * len(Dataset))
    val_size = len(Dataset) - train_size
    trainDataset, valDataset = torch.utils.data.random_split(Dataset, [train_size, val_size])
    # trainLoader = ZaloDataLoader(opt, trainDataset)
    # valLoader = ZaloDataLoader(opt, valDataset)
    trainLoader = DataLoader(trainDataset, batch_size=opt.batch_size, shuffle= True, num_workers= opt.workers)
    valLoader = DataLoader(valDataset, batch_size=opt.batch_size, shuffle= True, num_workers= opt.workers)
    dataset_sizes = {
        'train' : len(trainDataset),
        'val' : len(valDataset),
    }
    dataLoader = {
        'train' : trainLoader,
        'val': valLoader
    }
    
    model, best_epoch = train(opt)
    torch.save({
        'epoch': best_epoch,
        'model_state_dict': model.state_dict(),
    }, os.path.join(opt.checkpoint_dir, opt.name_model, ("best_epoch"+".pth")))

    print("done")
#python zalo/train.py --name_model mobilenet_v3_smaill --epochs 5 --lr 0.01