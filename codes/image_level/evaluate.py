from cProfile import label
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def compute_confusion_matrix(precited,expected):           
    part = np.logical_xor(precited, expected)
    pcount = np.bincount(part)             
    tp_list = list(precited & expected)    
    fp_list = list(precited & ~expected)   
    tp = tp_list.count(1)                  
    fp = fp_list.count(1)                  
    tn = pcount[0] - tp                   
    fn = pcount[1] - fp                    
    return tp, fp, tn, fn


def compute_indexes(tp, fp, tn, fn):
    accuracy = (tp+tn) / (tp+tn+fp+fn)     
    sensitivity = tp / (tp+fn)
    specificity = tn / (tn+fp)
    try:
        precision = tp / (tp+fp)         
    except:
        precision=0
    try:      
        recall = tp / (tp+fn)     
    except:
        recall=0             
    F1 = (2*precision*recall) / (precision+recall)    
    return accuracy, sensitivity, specificity, F1
        

def get_one_hot(label, num_cls):
    label = label.reshape(-1)
    label = torch.eye(num_cls, device=label.device)[label]
    return label




def evaluate(net, dataloader, device,batch_size):
    net.eval()
    num_val_batches = len(dataloader)
    prob_all=[]
    label_all=[]
    softmax = torch.nn.Softmax(dim=1)
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, clss = batch['image2'],batch['clss']
        image = image.to(device=device, dtype=torch.float32)
        clss = clss.to(device=device,dtype=torch.float)
        
        
        with torch.no_grad():
            out1,_,_= net(image)
            out1=softmax(out1)
            
            out1=out1.cpu().numpy()
            clss=clss.cpu().numpy()
            prob_all.extend(list(out1[:,1]))
            label_all.extend(list(clss[:,1]))
            
    
    net.train()
    net.freeze_bn()
    
    roc_auc=roc_auc_score(label_all,prob_all)
    
    prob_all = np.array(prob_all).round().astype(int)
    label_all = np.array(label_all).astype(int)
    tp, fp, tn, fn = compute_confusion_matrix(prob_all, label_all)
    
    print("tp, fp, tn, fn:",tp, fp, tn, fn)
    
    accuracy, sensitivity, specificity, F1 = compute_indexes(tp, fp, tn, fn)
    
    print("accuracy, sensitivity, specificity, F1,roc_auc:",accuracy, sensitivity, specificity, F1,roc_auc)
    
    return accuracy
    



