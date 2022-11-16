import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader

import torchvision.models as models
# import PIL
# from PIL import Image 
import torch.optim as optim
import cv2
from model import Net_1 
from datasets import ZaloDataset, ZaloDataLoader
from utils.cropvideo import crop
from model import VGG16
def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='nam')
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('-j', '--workers', type=int, default=2)
    parser.add_argument('--load_height', type=int, default=224)
    parser.add_argument('--load_width', type=int, default=224)
    parser.add_argument('--replicate', type=int, default=5)
    parser.add_argument('--shuffle', action='store_true')
    
    #
    parser.add_argument('--dataset_dir', type=str, default='dataset')
    parser.add_argument('--train_dir', type=str, default='zalo/dataset/train')
    parser.add_argument('--val_dir', type=str, default='zalo/dataset/val')
    parser.add_argument('--save_dir', type=str, default='results/')
    
    #checkpoints, train
    parser.add_argument('--checkpoint_dir', type=str, default='zalo/checkpoints/')
    parser.add_argument("--gpu", type=str, default='1', help="choose gpu device.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--learning_rate", type=int, default=0.005)

    opt = parser.parse_args()
    return opt
# -------------------------------------------------
def train(opt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # ----------------Net_1------------------------
    # model = Net_1()
    # optimizer = optim.SGD(model.parameters(), lr=3e-2)

    # loss_fn = nn.CrossEntropyLoss()
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # ----------------VGG16------------------------
    model = VGG16().to(device)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=opt.learning_rate, weight_decay = 0.005, momentum = 0.9)
    #----------------------dataloader---------------------
    Dataset = ZaloDataset (opt=opt)
    train_size = int(0.8 * len(Dataset))
    val_size = len(Dataset) - train_size
    trainDataset, valDataset = torch.utils.data.random_split(Dataset, [train_size, val_size])
    # trainLoader = ZaloDataLoader(opt, trainDataset)
    trainLoader = DataLoader(trainDataset, batch_size=opt.batch_size, shuffle= True, num_workers= opt.workers)
    valLoader = DataLoader(valDataset, batch_size=opt.batch_size, shuffle= True, num_workers= opt.workers)
    # Train the model
    total_step = len(trainLoader)
    #------------------------train-------------------------
    losses_train = []
    losses_valid = []
    accu_valid = []
    for epoch in range(opt.epochs):
        running_train_loss = 0.0
        running_valid_loss = 0.0
        running_valid_accu = 0.0

        model.train() #gradient optimize

        for inputs in tqdm(trainLoader):
            optimizer.zero_grad()
            img = inputs['img'].to(device)
            label = inputs['label'].to(device)

            output = model(img)
            loss = criterion(output, label)
            optimizer.zero_grad() 
            loss.backward()            
            optimizer.step()

            running_train_loss += loss.item() * img.size(0) 
        epoch_train_loss = running_train_loss / len(trainLoader)
        losses_train.append(epoch_train_loss)
        print("")
        print('Training, Epoch {} - Loss {}'.format(epoch+1, epoch_train_loss))

        model.eval()

        for inputs in tqdm(valLoader):
            img = inputs['img'].to(device)
            label = inputs['label'].to(device)
 
            with torch.no_grad():
                output = model(img)

            pred_label = torch.argmax(output, dim=1)
            pred_label = pred_label.cpu().numpy()

            accuracy = np.count_nonzero(pred_label == label) / opt.batch_size
            running_valid_accu += accuracy

            loss = criterion(output, label)
            running_valid_loss += loss.item() * img.size(0)

            epoch_valid_loss = running_valid_loss / len(valLoader)
            epoch_valid_accu = running_valid_accu / len(valLoader)
            accu_valid.append(epoch_valid_accu)
            optimizer.step()
            losses_valid.append(epoch_valid_loss)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, os.path.join(opt.checkpoint_dir, str(epoch)))
    print("")
    print('Validated, Epoch {} - Loss {} - Acc {}'.format( epoch+1, epoch_valid_loss, epoch_valid_accu))

#------------------------------------------------

if __name__ == '__main__':
    opt = get_opt()
    
    crop(opt= opt)
    if os.path.exists(opt.checkpoint_dir) == False:
        os.mkdir(opt.checkpoint_dir)
    train(opt)
    print("done")