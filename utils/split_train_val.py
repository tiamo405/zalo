import cv2
import os
def split(opt) :
    folder_path = os.path.join(opt.val_dir, str(opt.replicate))
    if os.path.exists(folder_path) == False:
        os.mkdir(folder_path)
        fnames = os.listdir(os.path.join(opt.train_dir, str(opt.replicate)))
        for i in range(int(len(fnames)/5)) :
            img = cv2.imread(os.path.join(opt.train_dir, str(opt.replicate), fnames[i]))
            cv2.imwrite(os.path.join(folder_path, fnames[i]), img)
            os.remove(os.path.join(opt.train_dir, str(opt.replicate), fnames[i]))
