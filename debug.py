import cv2
import os
import pandas as pd
import json
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
