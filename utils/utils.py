import torch
import os
import pandas as pd
def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise ValueError("'{}' is not a valid checkpoint path".format(checkpoint_path))
    model.load_state_dict(torch.load(checkpoint_path)['model_state_dict'])

def predict(res):
    dem = 0
    for i in res :
        if i == 0 :
            dem+=1
    if dem > int(len(res) /2 ) :
        # return 1 - dem / len(res)
        return 0.1
    # return int(len(res) -dem) / len(res)
    return 0.2

def save_csv(x, y, opt):
    z= []
    for i in range (len(x)) :
        tmp=[]
        tmp.append(int(x[i].split(".")[0]))
        tmp.append(y[i])
        z.append(tmp)
    x = []
    y = []
    z = sorted(z)
    for i in range(len(z)) :
        x.append(str(z[i][0])+".mp4")
        y.append(z[i][1])
    df = pd.DataFrame({
        'fname' : x,
        'liveness_score': y
    })
    i = 1
    f = ''.join(os.listdir(opt.save_dir))
    while str(i) in f :
        i+=1
    os.mkdir(os.path.join(opt.save_dir, str(i)+ "_"+ opt.public+ "_"+ opt.name_model))
    df.to_csv(os.path.join(opt.save_dir,str(i)+ "_"+ opt.public+ "_"+ opt.name_model, "predict.csv"), index=False)
    