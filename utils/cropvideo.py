import cv2
import os
def crop(opt):
    folder_path = os.path.join(opt.train_dir, str(opt.replicate))
    if os.path.exists(folder_path) == False:
        os.mkdir(folder_path)
        fnames = os.listdir(os.path.join(opt.train_dir, "videos"))
        # print(fnames)
        for fname in fnames :
            path = os.path.join(opt.train_dir, "videos", fname)
            cap=cv2.VideoCapture(path)

            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            index = []
            for j in range(1, opt.replicate + 1) :
                index.append(int(frame_count / opt.replicate * j - 5))
            i = 0
            vt = 0
            while True :
                ret, frame = cap.read()
                if ret :
                    if i in index :
                        
                        path_save = os.path.join(folder_path,fname.split(".")[0] + '_'+ (str(vt)+".jpg" ))
                        cv2.imwrite(path_save, frame)
                        vt +=1
                    i += 1
                else : break
            
            cap.release ()
