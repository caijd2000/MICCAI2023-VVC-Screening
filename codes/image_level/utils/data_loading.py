import logging
from os import listdir
from os.path import splitext
from pathlib import Path
import cv2 as cv
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
transform1 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.762, 0.774, 0.864],
        std=[0.249, 0.254, 0.166],
    )
])
transform2 = transforms.Compose([
    transforms.ToTensor()
])



class BasicDataset(Dataset):
    def __init__(self, img_size:int,images_dir: str,  scale: float = 1.0, mask_suffix: str = '',transform=None):
        self.images_dir = Path(images_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.img_size=img_size
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, img_size,pil_img, scale, is_mask,transform):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'

        pil_img = pil_img.resize((img_size, img_size), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        pil_img_orig = pil_img.copy()
        
        pil_img = transform(pil_img)
        pil_img2 = transform1(pil_img_orig)#only normal

        pil_img_orig = transform2(pil_img_orig)
        return pil_img,pil_img2,pil_img_orig#img1,img2,orig


    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        elif ext in ['.jpg','.png']:
            return Image.open(filename)



    def __getitem__(self, idx):
        name = self.ids[idx]
        
        if (name[0]=='p'):
            clss = [0.0,1.0]
        else:
            clss = [1.0,0.0]
        
        img_file = list(self.images_dir.glob(name + '.*'))
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        img = self.load(img_file[0])


        img1,img2,orig = self.preprocess(img_size=self.img_size,pil_img=img, scale=self.scale, is_mask=False,transform=self.transform)
        return {
            'image': img1,#tran
            'image2':img2,#just normal
            'clss':torch.as_tensor(clss),
            'orig':orig
        }


class CarvanaDataset(BasicDataset):
    def __init__(self,img_size,images_dir,  scale=1,transform=None):
        super().__init__(img_size,images_dir,  scale,transform=transform)
