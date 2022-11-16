
from os import path as osp
from torch.utils import data
from torchvision import transforms
import cv2
import pandas as pd


class ZaloDataset(data.Dataset):
    def __init__(self, opt):
        super(ZaloDataset, self).__init__()
        self.load_height = opt.load_height
        self.load_width = opt.load_width
        self.replicate = opt.replicate
        self.data_path = osp.join(opt.train_dir, str(opt.replicate))
        self.label_path = osp.join(opt.train_dir, "label.csv")
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # load data list
        img_names = []
        labels = []
        df  = pd.read_csv(self.label_path)
        for i in range(len(df)):
            img_name = df['fname'][i].split(".")[0]
            label = df['liveness_score'][i]
            for j in range(self.replicate) :
                img_names.append(img_name + '_' + str(j))
                labels.append(label)

        self.img_names = img_names
        self.labels = labels

    def __getitem__(self, index):
        img_name = self.img_names[index]
        label = self.labels[index]
        img = cv2.imread(osp.join(self.data_path, (img_name+ ".jpg")))
        img = cv2.resize(img, (self.load_width, self.load_height))
        img = self.transform(img)
        
        result = {
            'img_name': img_name,
            'label' : label,
            'img': img,
        }
        
        return result

    def __len__(self):
        return len(self.img_names)


class ZaloDataLoader:
    def __init__(self, opt, dataset):
        super(ZaloDataLoader, self).__init__()

        if opt.shuffle:
            train_sampler = data.sampler.RandomSampler(dataset)
        else:
            train_sampler = None

        self.data_loader = data.DataLoader(
                dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                num_workers=opt.workers, pin_memory=True, drop_last=True, sampler=train_sampler
        )
        self.dataset = dataset
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch

