import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import os
from model import  resnet_vit as resnet
import pandas as pd
import scipy.io as scio

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.762, 0.774, 0.864],
        std=[0.249, 0.254, 0.166],
    )
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path='model/fungus_image_level.pth'
res = resnet.resnet50(num_classes=1, pretrained=False)
model_dict=torch.load(model_path)
res.load_state_dict(model_dict)

if torch.cuda.is_available():
    res = res.cuda()
res.training = False
res.eval()

num_thre=3
score_thre=0.93



def preprocess(pil_img, scale,transform):
    w, h = pil_img.size
    newW, newH = int(scale * w), int(scale * h)
    assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
    pil_img = pil_img.resize((1024, 1024), resample=Image.BICUBIC)
    pil_img = transform(pil_img)
    return pil_img


def read_img(img_path):
    image = Image.open(img_path) # or convert('L')
    image = preprocess(image, scale=1.0,transform=transform)
    image = image.unsqueeze(0)
    image = image.to('cuda')
    return image

def start_inference(sample_code, path): 
    try:
        inputs=read_img(path)
    except:
        return 0
    with torch.no_grad():
        score_img,cam_img,feature_img=res(inputs)
        
    return score_img[0][1].cpu().numpy(),feature_img.cpu().numpy()

if __name__ == '__main__':
    all_names = []
    all_scores = []
    all_img_scores=[]
    all_data=[]
    all_features=[]
    predict=[]
    final_scores=[]
    src_dir=r'D:\\fungus/sample200/fungus/'
    pos_path=os.listdir(src_dir)
    for sample in pos_path:
        sample_dir=os.path.join(src_dir,sample,'torch')
        sample_code=sample
        try:
            sample_path=os.listdir(sample_dir)
        except:
            print(sample_path," is not a path")
            continue
        scores=[]
        predict=0
        features=[]
        for idx in range(len(sample_path)):
            if idx>300:
                break
            img_path=sample_path[idx]
            path=os.path.join(sample_dir,img_path)
            score_img,feature_img=start_inference(sample_code, path)
            scores.append(score_img)
            features.append(feature_img)
        idxs=sorted(range(len(scores)),key= lambda k:scores[k],reverse=True)
        sample_features=[]#20*2048
        print(sample_code,scores[idxs[0]])
        for i in range(20):
            sample_features.append(features[idxs[i]])
        sample_features=np.array(sample_features)
        
        file_name='features/'+sample_code+'.mat'
        scores.sort(reverse=True)
        final_score=np.array(scores[0:20])
        scio.savemat(file_name,{'data':sample_features,'label':1,'predict':final_score})
    
    
    
    all_names = []
    all_scores = []
    all_img_scores=[]
    all_data=[]
    all_features=[]
    predict=[]
    final_scores=[]
    src_dir=r'D:\\fungus/sample200/no_fungus/'
    pos_path=os.listdir(src_dir)
    for sample in pos_path:
        sample_dir=os.path.join(src_dir,sample,'torch')
        sample_code=sample
        try:
            sample_path=os.listdir(sample_dir)
        except:
            print(sample_path," is not a path")
            continue
        scores=[]
        predict=0
        features=[]
        for idx in range(len(sample_path)):
            if idx>300:
                break
            img_path=sample_path[idx]
            path=os.path.join(sample_dir,img_path)
            score_img,feature_img=start_inference(sample_code, path)
            scores.append(score_img)
            features.append(feature_img)
        idxs=sorted(range(len(scores)),key= lambda k:scores[k],reverse=True)
        sample_features=[]#20*2048
        print(sample_code,scores[idxs[0]])
        for i in range(20):
            sample_features.append(features[idxs[i]])
        sample_features=np.array(sample_features)
        
        file_name='features/'+sample_code+'.mat'
        scores.sort(reverse=True)
        final_score=np.array(scores[0:20])
        scio.savemat(file_name,{'data':sample_features,'label':0,'predict':final_score})
    