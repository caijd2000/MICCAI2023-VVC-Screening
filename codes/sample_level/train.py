import argparse
import logging
import sys
import scipy.io as scio
from pathlib import Path
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
import os
import random
import cv2 as cv
from evaluate import evaluate
from trans import  transformer
from torchvision import transforms
import numpy as np
import torch.nn as nn
import torchvision.models as models



datasets=[]
scores=[]

dir='features/'
dir_checkpoint = Path(r'checkpoints/')


filename=os.listdir(dir)
filename.sort()
for name in filename:
    file_path=os.path.join(dir,name)
    file=scio.loadmat(file_path)
    file['name']=name
    k=10
    
    if file['label']==1:
        file['label']=torch.tensor([0,1])
    else:
        file['label']=torch.tensor([1,0])     
    file['predict']=file['predict'][:,:k]
    pre=np.array(torch.tensor(file['predict']).squeeze(0))
    data=np.array(torch.tensor(file['data']).squeeze(1))
    now=data
    file['data']=torch.tensor(now[:k,:,:,:])
    datasets.append(file)



def train_net(net,
              device,
              train_loader,
              val_loader,
              epochs: int = 50,
              batch_size: int = 1,
              learning_rate: float = 0.001,
              save_checkpoint: bool = True,
              amp: bool = False,
              fold=1
              ):
    best_auc,best_f1,best_acc,best_sen,best_spe=0,0,0,0,0
    optimizer = optim.Adam(filter(lambda p:p.requires_grad,net.parameters()), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    weights=torch.tensor([0.5,0.5]).to(device=device, dtype=torch.float32)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean',weight=weights)
    global_step = 0
    for epoch in range(epochs):
        net.train()
        n_train=len(train_loader)
        loss=None
        random.shuffle(train_loader)
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='sample') as pbar:
            for i in range(n_train):
                
                x=train_loader[i]['data'].to(device=device, dtype=torch.float32)
                score=torch.tensor(train_loader[i]['predict']).to(device=device, dtype=torch.float32)
                
                y=train_loader[i]['label'].to(device=device, dtype=torch.float32)
                y=y.unsqueeze(0)
                #loss = 0
                
                with torch.cuda.amp.autocast(enabled=amp):
                    

                    out=net(x,score,True)
                    loss=loss_fn(out,y)
                    grad_scaler.scale(loss).backward(retain_graph=True)
                    if (i+1)%batch_size==0:
                        grad_scaler.step(optimizer)
                        grad_scaler.update()
                        pbar.update(batch_size)
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1
                        pbar.set_postfix(**{'loss(batch)': loss.item()})
                        
                        # Evaluation round
                        division_step = (n_train // (batch_size))
                        if division_step > 0:
                            if global_step % division_step == 0:
                                
                                print()
                                print("eopch",epoch,":")
                                print()
                                tra_auc,tra_f1,tra_acc,tra_sen,tra_spe = evaluate(net, train_loader, device,batch_size)
                                val_auc,val_f1,val_acc,val_sen,val_spe = evaluate(net, val_loader, device,batch_size)
                                
                                scheduler.step(val_f1)
                                print("lr:",optimizer.param_groups[0]['lr'])
                                print("tra_auc,tra_f1,tra_acc,tra_sen,tra_spe:",tra_auc,tra_f1,tra_acc,tra_sen,tra_spe)
                                print("val_auc,val_f1,val_acc,val_sen,val_spe:",val_auc,val_f1,val_acc,val_sen,val_spe)
                                logging.info('Validation score: {}'.format(val_f1))
                                if val_acc>best_acc:
                                    best_auc,best_f1,best_acc,best_sen,best_spe=val_auc,val_f1,val_acc,val_sen,val_spe
                        

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'fold{}_checkpoint_epoch{}.pth'.format(fold,epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')
    return best_auc,best_f1,best_acc,best_sen,best_spe


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')


    try:
        all_auc,all_f1,all_acc,all_sen,all_spe=[],[],[],[],[]
        l=len(datasets)
        split=5
        each=int(l/split)
        random.shuffle(datasets)
        for i in range(5):
            
            config=transformer.Config()
            
            net = transformer.trans(config)
            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                net = nn.DataParallel(net)
            if torch.cuda.is_available():
                net.cuda()
            train_loader=[]
            val_loader=[]
            if i>0:
                train_loader.extend(datasets[0:each*i])
            if i<4:
                train_loader.extend(datasets[each*(i+1):l])
            val_loader=datasets[each*i:each*(i+1)]
            best_auc,best_f1,best_acc,best_sen,best_spe\
            =train_net(net=net,
                device=device,train_loader=train_loader,
                val_loader=val_loader,
                epochs= args.epochs,
                batch_size= args.batch_size,
                learning_rate=args.lr,
                amp=args.amp,
                fold=i
                )
            all_auc.append(best_auc)
            all_f1.append(best_f1)
            all_acc.append(best_acc)
            all_sen.append(best_sen)
            all_spe.append(best_spe)

        print("auc:",np.mean(all_auc),' ',np.std(all_auc),' ',all_auc)
        print("f1 :",np.mean(all_f1 ),' ',np.std(all_f1 ),' ',all_f1 )
        print("acc:",np.mean(all_acc),' ',np.std(all_acc),' ',all_acc)
        print("sen:",np.mean(all_sen),' ',np.std(all_sen),' ',all_sen)
        print("spe:",np.mean(all_spe),' ',np.std(all_spe),' ',all_spe)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
