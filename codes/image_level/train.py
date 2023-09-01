import argparse
import logging
import sys
from pathlib import Path
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from utils.data_loading import BasicDataset, CarvanaDataset
from evaluate import evaluate
from model import  resnet_vit as resnet
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import torch.nn as nn
import torchvision.models as models
import random
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


seed=2023
torch.manual_seed(seed) 
torch.cuda.manual_seed(seed) 
torch.cuda.manual_seed_all(seed)  
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.	
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True






model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def compute_confusion_matrix(precited,expected):
    part = precited ** expected             
    pcount = np.bincount(part)             
    tp_list = list(precited & expected)    
    fp_list = list(precited & ~expected)   
    tp = tp_list.count(1)                  
    fp = fp_list.count(1)                  
    tn = pcount[0] - tp                   
    fn = pcount[1] - fp                    
    return tp, fp, tn, fn



#normal training
dir_img_train = Path(r'D:\\fungus\\multi-resnet\\train_img/')
dir_img_test = Path(r'D:\\fungus\\multi-resnet\\test_img/')
dir_img_val = Path(r'D:\\fungus\\multi-resnet\\val_img/')
dir_checkpoint = Path(r'D:\\fungus\\multi-resnet\\checkpoints/')


    
    
class loss_am_focus(nn.Module):
    def __init__(self):
        super(loss_am_focus, self).__init__()
    def forward(self, output,limit):
        am = limit + output[:,1]
        return torch.mean(am)


transform1 = transforms.Compose([
    transforms.RandomGrayscale(p=0.1),
    transforms.RandomEqualize(1.0),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.ToTensor(),
    transforms.Normalize(#using param of our dataset
        mean=[0.762, 0.774, 0.864],
        std=[0.249, 0.254, 0.166],
    )
])


transform2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize( 
        mean=[0.762, 0.774, 0.864], 
        std=[0.249, 0.254, 0.166],
    )
])
transform3 = transforms.Compose([
    transforms.Normalize(
        mean=[0.762, 0.774, 0.864],
        std=[0.249, 0.254, 0.166],
    )
])


def train_net(net,
              device,
              epochs: int = 50,
              batch_size: int = 1,
              learning_rate: float = 0.001,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 1,
              amp: bool = False):
    img_size=1024
    # 1. Create dataset
    try:
        #dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
        train_set = CarvanaDataset(img_size,dir_img_train, img_scale,transform1)
        val_set = CarvanaDataset(img_size,dir_img_val,  img_scale,transform=transform2)
        test_set = CarvanaDataset(img_size,dir_img_test, img_scale,transform=transform2)
    except (AssertionError, RuntimeError):
        train_set = CarvanaDataset(img_size,dir_img_train,  img_scale,transform=transform1)
        val_set = CarvanaDataset(img_size,dir_img_val,  img_scale,transform=transform2)
        test_set = CarvanaDataset(img_size,dir_img_test,  img_scale,transform=transform2)
    
    n_val=len(val_set)
    n_train=len(train_set)
    

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=8, pin_memory=True)
    
    #shan
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True,**loader_args)
    val_loader = DataLoader(val_set, shuffle=True, drop_last=True, **loader_args)
    test_loader = DataLoader(test_set, shuffle=True, drop_last=True, **loader_args)
    

    

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(filter(lambda p:p.requires_grad,net.parameters()), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
    loss_AM = loss_am_focus()
    loss_tri=torch.nn.TripletMarginLoss(margin=1.0)
    global_step = 0
    sigmoid=torch.nn.Sigmoid()
    softmax = torch.nn.Softmax(dim=1)

    for epoch in range(epochs):
        net.train()
        net.freeze_bn()
        epoch_loss = 0
        predict_train=[0,0,0,0]
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch_idx,batch in enumerate(train_loader):
                images_tra = batch['image']
                images_pre = batch['image2']#only normal
                clss = batch['clss']
                
                loss = 0
                
                optimizer.zero_grad(set_to_none=True)
                images_tra = images_tra.to(device=device, dtype=torch.float32)
                images_pre = images_pre.to(device=device, dtype=torch.float32)
                clss=clss.to(device=device, dtype=torch.float32)
                
                
                with torch.cuda.amp.autocast(enabled=amp):
                    out_pre,cam_pre,out1_pre=net(images_pre)

                    out_tra,cam_tra,out1_tra=net(images_tra)
                    
                    
                    #get masked image
                    
                    
                    soft_mask = sigmoid(10*(cam_pre[:,1:2,:,:]-0.5))
                    soft_mask = soft_mask.repeat(1,3,1,1)
                    orig=batch['orig'].to(device=device, dtype=torch.float32)
                    masked_images=orig-orig*soft_mask
                    
                    
                    masked_images = transform3(masked_images)
                    new_input = masked_images.to(device=device, dtype=torch.float32)
                    
                    out_new,cam_new,out1_new=net(new_input)  
                    out_new=softmax(out_new)
                    
                    
                    
                    limit=torch.mean(torch.mean(cam_pre[:,1,:,:],dim=-1),dim=-1) #mean of attention map
                    limit = limit.to(device=device, dtype=torch.float32)
                    
                    
                    loss1=loss_fn(out_tra,clss)
                    loss2=loss_tri(out1_pre,out1_tra,out1_new)
                    loss3=loss_AM(out_new,limit)
                    loss = loss1 + 0.1*loss2 + 0.1*loss3
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    
                    pbar.update(images_tra.shape[0])
                    global_step += 1
                    epoch_loss += loss.item()
                
                
                
                predict_batch = torch.round(softmax(out_tra)[:,1]).cpu().detach().numpy().astype(int)
                expected_batch = clss[:,1].cpu().detach().numpy().astype(int)
                
                tp, fp, tn, fn = compute_confusion_matrix(predict_batch, expected_batch)
                
                predict_train += np.array([tp, fn, tn, fp])
                
                
                pbar.set_postfix(**{'loss1(batch)': loss1.item(),'loss2(batch)': loss2.item(),'loss3(batch)': loss3.item()})
                # Evaluation round
                division_step = (n_train // (batch_size))
                if batch_idx%100==0:
                    print("train tp,fn,tn,fp:",predict_train)
                    
                  
                
                if division_step > 0:
                    if global_step % division_step == 0:
                        
                        print(epoch,":")
                        print("lr:",optimizer.param_groups[0]['lr'])
                        
                        print("train tp,fn,tn,fp:",predict_train)
                        
                        val_score = evaluate(net, val_loader, device,batch_size)
                        scheduler.step(val_score)
                        print("val_score:",val_score)
                        
                        test_score = evaluate(net, test_loader, device,batch_size)
                        print("test_score:",test_score)
                        

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch + 1)))
            logging.info(f'Checkpoint {epoch + 1} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=3e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    net = resnet.resnet50(num_classes=1, pretrained=False)
    
    model = torch.load('pretrained_model/retinanet_pretrain.pth').state_dict()
    net.load_state_dict(model,strict=False)
    net.freeze_bn()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)
    if torch.cuda.is_available():
        net.cuda()
    
    
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
