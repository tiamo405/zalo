import cv2
import os
import pandas as pd
import json
import numpy as np
def testtrain():
    df = pd.read_csv("zalo/dataset/train/label.csv")
    img_names = []
    labels = []
    for i in range(len(df)):
        img_name = df['fname'][i].split(".")[0] 
        label = df['liveness_score'][i]
        # for j in range(self.replicate) :
        for fname in os.listdir(os.path.join("zalo/dataset/train", "data", img_name )) :
            if "jpg" in fname:
                img_names.append(img_name + '_' + fname.split(".")[0])
                labels.append(label)
        print(img_names, labels)
        break
# testtrain()
def testjson():
    f = open("zalo/dataset/train/data/1886/024.json")
    img = cv2.imread("zalo/dataset/train/data/1886/024.jpg")
    data = json.load(f)
    a = data['bbox']
    img_new = img[a[1]:a[3], a[0]:a[2]]
def save():
    i = 1
    f = ''.join(os.listdir("zalo/results"))
    while str(i) in f :
        i+=1
    print(i)
save()
print(len(os.listdir("zalo/dataset/test/test2")))
def testcrop():
    if not os.path.exists("test"):
        os.mkdir("test")
    path = "zalo/dataset/test/test2"
    for f_name in os.listdir(path) :
        print(f_name)
        for f_img in os.listdir(os.path.join(path, f_name)):
            if ".jpg" in f_img :
                image = cv2.imread(os.path.join(path, f_name, f_img))
                a = json.load(open(os.path.join(path, f_name, f_img.split(".")[0] + ".json")))['bbox']
                img = image[abs(a[1]): abs(a[3]), abs(a[0]): abs(a[2])]
                if not os.path.exists(os.path.join("test", f_name)) :
                    os.mkdir(os.path.join("test", f_name))
                path_save = os.path.join("test", f_name, f_img)
                cv2.imwrite(path_save, img)
def res() :
    res = []
    for i in range(5) :
        tmp =[]
        tmp.append(i)
        tmp.append(i+5)
        res.append(tmp)
    res = np.array(res)
    print(abs(np.sum(res[:,0])) - np.sum(res[:,1]))
    print(res)
res()